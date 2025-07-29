"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from jsonargparse import ActionParser, ArgumentParser

from ...utils import SegmentSet
from ...utils.misc import filter_func_args
from .hyp_sampler import HypSampler
from .seg_sampler import LengthSamplingMethod, SegSampler


class SegChunkSampler(HypSampler):
    """
    A sampler that generates fixed or variable-length overlapping chunks from a set of segments.

    This is useful for tasks such as speaker recognition or speech contrastive learning,
    where fixed-duration windows are needed but input segments vary in length.

    Each segment is split into chunks using the following logic:
    - Chunks are drawn with random or fixed length in [min_chunk_length, max_chunk_length].
    - Adjacent chunks may overlap, with overlap chosen randomly in [min_chunk_overlap, max_chunk_overlap].
    - Chunks are guaranteed to cover the full segment, with special handling for short tails.
    - Chunks are wrapped in a SegmentSet and sampled using a base sampler like `SegSampler`.

    Attributes:
        segments (SegmentSet): Input segments to be chunked.
        min_chunk_length (float): Minimum chunk length (in seconds or frames).
        max_chunk_length (Optional[float]): Maximum chunk length. If None, equals min_chunk_length.
        min_chunk_overlap (float): Minimum overlap between adjacent chunks.
        max_chunk_overlap (Optional[float]): Maximum overlap. If None, equals min_chunk_overlap.
        base_sampler (Type[HypSampler]): Sampler class used to sample chunked data (e.g., SegSampler).
        length_sampling_method (LengthSamplingMethod): Strategy for chunk length (e.g., uniform or max).
        length_name (str): Column in `segments` representing the segment length (usually "duration").
        max_batches_per_epoch (Optional[int]): Maximum number of batches per training epoch.
        shuffle (bool): Whether to shuffle chunks before sampling.
        seed (int): Random seed used for reproducibility.
        **base_kwargs: Additional arguments passed to the base sampler.
    """

    def __init__(
        self,
        segments: SegmentSet,
        min_chunk_length: float,
        max_chunk_length: Optional[float] = None,
        min_chunk_overlap: float = 0.0,
        max_chunk_overlap: Optional[float] = None,
        base_sampler: Type[HypSampler] = SegSampler,
        length_sampling_method: LengthSamplingMethod = LengthSamplingMethod.UNIFORM,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        **base_kwargs,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.segments = segments
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )
        if length_sampling_method == LengthSamplingMethod.UNIFORM:
            self.avg_chunk_length = (self.max_chunk_length + self.min_chunk_length) / 2
        elif length_sampling_method == LengthSamplingMethod.MAXIMUM:
            self.avg_chunk_length = self.max_chunk_length
        self.min_chunk_overlap = min(min_chunk_overlap, self.min_chunk_length * 0.5)
        self.max_chunk_overlap = (
            min_chunk_overlap if max_chunk_overlap is None else max_chunk_overlap
        )
        self.avg_chunk_overlap = (self.max_chunk_overlap + self.min_chunk_overlap) / 2
        self.chunk_set = None
        self.length_name = length_name
        self.chunk_sampler = base_sampler
        self.length_sampling_method = length_sampling_method

        logging.info("Initializing SegChunkSampler with %d segments", len(segments))
        logging.info(
            "Chunk length range: [%.2f, %.2f] (avg=%.2f), Overlap range: [%.2f, %.2f] (avg=%.2f)",
            self.min_chunk_length,
            self.max_chunk_length,
            self.avg_chunk_length,
            self.min_chunk_overlap,
            self.max_chunk_overlap,
            self.avg_chunk_overlap,
        )
        logging.info("Length sampling strategy: %s", length_sampling_method)

        # Allow nested use: "subbase_sampler" can override deeper base
        if "subbase_sampler" in base_kwargs:
            base_kwargs["base_sampler"] = base_kwargs.pop("subbase_sampler")

        # Filter args that apply to the base_sampler
        self.base_kwargs = base_sampler.filter_args(**base_kwargs)
        self.base_kwargs["seed"] = seed
        self.base_kwargs["shuffle"] = shuffle
        self.base_kwargs["max_batches_per_epoch"] = max_batches_per_epoch

        logging.debug("Base sampler: %s", base_sampler.__name__)
        logging.debug("Base sampler args: %s", self.base_kwargs)

        # Build chunk set and base sampler
        logging.info("Creating initial chunk set and initializing base sampler...")
        # Build the initial chunk set and sampler, to get avg_batch_size
        self._create_chunks()
        self._seg_sampler = self.chunk_sampler(self.chunk_set, **self.base_kwargs)
        self.avg_batch_size = self._seg_sampler.avg_batch_size
        logging.info("Sampler ready: avg_batch_size = %.2f", self.avg_batch_size)

    def __len__(self):
        return len(self._seg_sampler)

    @property
    def duration_is_random(self):
        """Returns True if chunk lengths are randomized within [min, max]."""
        return self.min_chunk_length != self.max_chunk_length

    def get_random_duration(self):
        """Returns a single random chunk length in the valid range."""
        if self.duration_is_random:
            return (
                torch.rand(size=(1,), generator=self.rng).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )
        else:
            return self.min_chunk_length

    def get_random_overlap(self) -> float:
        if self.min_chunk_overlap == self.max_chunk_overlap:
            return self.min_chunk_overlap
        return (
            torch.rand(size=(1,), generator=self.rng).item()
            * (self.max_chunk_overlap - self.min_chunk_overlap)
            + self.min_chunk_overlap
        )

    def _create_chunks(self):
        """
        Slices the original segments into smaller chunks.
        Guarantees at least one chunk per segment.
        """
        if self.chunk_set is not None:
            del self.chunk_set
            self.chunk_set = None

        num_segments = len(self.segments)
        num_skipped = 0

        logging.info("Creating chunks from %d segments...", num_segments)
        chunks = []
        for id, length in zip(self.segments["id"], self.segments[self.length_name]):
            if length < self.min_chunk_length:
                # discard too short sequences
                num_skipped += 1
                continue

            # Using avg_chunk_length gives consistent chunk count across epochs,
            # but chunk positions vary due to random overlap.
            denom = max(self.avg_chunk_length - self.avg_chunk_overlap, 1e-4)
            num_chunks = math.ceil((length - self.avg_chunk_overlap) / denom)
            start = 0
            for i in range(num_chunks - 1):
                remainder = length - start
                if remainder < self.min_chunk_length:
                    # force minimum length for last piece
                    remainder = self.min_chunk_length
                    dur = remainder
                    start = length - dur
                else:
                    if self.length_sampling_method == LengthSamplingMethod.UNIFORM:
                        dur = self.get_random_duration()
                    else:
                        dur = self.max_chunk_length

                    dur = min(dur, remainder)

                chunk = (f"{id}-{i}", id, start, dur)
                chunks.append(chunk)
                increment = dur - self.get_random_overlap()
                if increment <= 0:
                    # Failsafe for numerical errors or unlucky overlap
                    increment = max(0.01, self.min_chunk_length * 0.1)
                start += increment

            # special treatment for last chunk we get from the recording
            remainder = length - start
            chunk_id = f"{id}-{num_chunks - 1}"
            if remainder > self.max_chunk_length:
                # here we discard part of the end
                chunk = (chunk_id, id, start, self.max_chunk_length)
            elif remainder < self.min_chunk_length:
                # here we overlap with second last chunk
                chunk = (
                    chunk_id,
                    id,
                    length - self.min_chunk_length,
                    self.min_chunk_length,
                )
            else:
                # use whatever is left as the last chunk
                chunk = (chunk_id, id, start, remainder)

            chunks.append(chunk)

        if not chunks:
            raise ValueError(
                f"No valid chunks were created. "
                f"All segments may be shorter than min_chunk_length={self.min_chunk_length}."
            )

        logging.info(
            "Chunking complete: %d total chunks from %d segments (skipped %d segments)",
            len(chunks),
            num_segments - num_skipped,
            num_skipped,
        )
        chunk_set = pd.DataFrame(
            chunks, columns=["id", "seg_id", "chunk_start", self.length_name]
        )
        self.chunk_set = SegmentSet(chunk_set)

    def __iter__(self):
        """Creates chunk set and wraps it with a new base sampler instance."""
        super().__iter__()
        self._create_chunks()
        self._seg_sampler = self.chunk_sampler(self.chunk_set, **self.base_kwargs)
        self._seg_sampler.set_epoch(self.epoch, self.init_batch)
        self._seg_sampler = iter(self._seg_sampler)

        return self

    def __next__(self):
        """Returns the next batch of chunked segments from the base sampler."""
        return next(self._seg_sampler)

    @staticmethod
    def filter_args(**kwargs):
        return kwargs
