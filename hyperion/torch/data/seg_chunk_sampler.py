"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch

from ...utils import SegmentSet
from .hyper_sampler import HyperSampler
from .seg_sampler import LengthSamplingMethod, SegSampler


class SegChunkSampler(HyperSampler):
    """
    A sampler that generates fixed or variable-length overlapping chunks from a set of segments.

    This is useful for tasks such as speaker recognition or speech contrastive learning,
    where fixed-duration windows are needed but input segments vary in length.

    Each segment is split into chunks using the following logic:
    - Chunks are drawn with random or fixed length in [min_chunk_length, max_chunk_length].
    - Adjacent chunks may overlap, with overlap chosen randomly in [min_chunk_overlap, max_chunk_overlap].
    - Short tails are covered by shifting the final chunk backward; tails longer than
      ``max_chunk_length`` may be discarded.
    - Chunks are wrapped in a SegmentSet and sampled using a base sampler like `SegSampler`.

    Attributes:
        segments (SegmentSet): Input segments to be chunked.
        min_chunk_length (float): Minimum chunk length (in seconds or frames).
        max_chunk_length (float): Maximum chunk length. If None, equals min_chunk_length.
        min_chunk_overlap (float): Minimum overlap between adjacent chunks.
        max_chunk_overlap (float): Maximum overlap. If None, equals min_chunk_overlap.
        avg_chunk_length (float): Average chunk length used to estimate chunk count.
        avg_chunk_overlap (float): Average chunk overlap used to estimate chunk count.
        base_sampler (Type[HyperSampler]): Sampler class used to sample chunked data (e.g., SegSampler).
        length_sampling_method (LengthSamplingMethod): Strategy for chunk length (e.g., uniform or max).
        length_name (str): Column in `segments` representing the segment length (usually "duration").
        _lengths (np.ndarray): Validated positional source-segment lengths.
        chunk_set (Optional[SegmentSet]): Current generated chunk metadata.
        chunk_sampler (Type[HyperSampler]): Base sampler class used for chunk sampling.
        base_kwargs (Dict[str, Any]): Arguments passed to the base sampler.
        avg_batch_size (float): Average batch size reported by the base sampler.
    """

    def __init__(
        self,
        segments: SegmentSet,
        min_chunk_length: float,
        max_chunk_length: Optional[float] = None,
        min_chunk_overlap: float = 0.0,
        max_chunk_overlap: Optional[float] = None,
        base_sampler: Type[HyperSampler] = SegSampler,
        length_sampling_method: LengthSamplingMethod = LengthSamplingMethod.UNIFORM,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        **base_kwargs: Any,
    ) -> None:
        """
        Initialize the chunking sampler.

        Args:
            segments: Segment metadata table to chunk.
            min_chunk_length: Minimum generated chunk length.
            max_chunk_length: Optional maximum generated chunk length. If ``None``,
                uses ``min_chunk_length``.
            min_chunk_overlap: Minimum overlap between adjacent chunks.
            max_chunk_overlap: Optional maximum overlap between adjacent chunks. If
                ``None``, uses ``min_chunk_overlap``.
            base_sampler: Sampler class used to sample from generated chunks.
            length_sampling_method: Strategy used to choose chunk lengths.
            length_name: Column in ``segments`` containing segment lengths.
            max_batches_per_epoch: Optional maximum number of batches per epoch.
            shuffle: Whether to shuffle chunks in the base sampler.
            seed: Base random seed.
            **base_kwargs: Additional arguments for the base sampler.
        """
        super().__init__(shuffle=shuffle, seed=seed)
        if len(segments) == 0:
            raise ValueError("segments must contain at least one row.")
        try:
            min_chunk_length = float(min_chunk_length)
            min_chunk_overlap = float(min_chunk_overlap)
            if max_chunk_length is not None:
                max_chunk_length = float(max_chunk_length)
            if max_chunk_overlap is not None:
                max_chunk_overlap = float(max_chunk_overlap)
        except (TypeError, ValueError) as error:
            raise ValueError("Chunk lengths and overlaps must be numeric.") from error
        chunk_values = [min_chunk_length, min_chunk_overlap]
        if max_chunk_length is not None:
            chunk_values.append(max_chunk_length)
        if max_chunk_overlap is not None:
            chunk_values.append(max_chunk_overlap)
        if not all(math.isfinite(value) for value in chunk_values):
            raise ValueError("Chunk lengths and overlaps must be finite.")
        if min_chunk_length <= 0:
            raise ValueError(
                f"min_chunk_length must be positive, got {min_chunk_length}."
            )
        self.segments = segments
        self.length_name = length_name
        try:
            self._lengths = np.asarray(
                self.segments[self.length_name].values, dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{self.length_name} must contain numeric segment lengths."
            ) from error
        if not np.all(np.isfinite(self._lengths)) or np.any(self._lengths <= 0):
            raise ValueError(
                f"{self.length_name} must contain finite, positive segment lengths."
            )
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )
        if self.max_chunk_length < self.min_chunk_length:
            raise ValueError(
                "max_chunk_length must be greater than or equal to min_chunk_length."
            )
        length_sampling_method = LengthSamplingMethod(length_sampling_method)
        if length_sampling_method == LengthSamplingMethod.UNIFORM:
            self.avg_chunk_length = (self.max_chunk_length + self.min_chunk_length) / 2
        elif length_sampling_method == LengthSamplingMethod.MAXIMUM:
            self.avg_chunk_length = self.max_chunk_length
        if min_chunk_overlap < 0:
            raise ValueError(
                f"min_chunk_overlap must be non-negative, got {min_chunk_overlap}."
            )
        self.min_chunk_overlap = min(min_chunk_overlap, self.min_chunk_length * 0.5)
        self.max_chunk_overlap = (
            min_chunk_overlap if max_chunk_overlap is None else max_chunk_overlap
        )
        if self.max_chunk_overlap < self.min_chunk_overlap:
            raise ValueError(
                "max_chunk_overlap must be greater than or equal to min_chunk_overlap."
            )
        max_allowed_overlap = self.min_chunk_length * 0.5
        if self.max_chunk_overlap > max_allowed_overlap:
            logging.warning(
                "Clamping max_chunk_overlap=%.2f to %.2f.",
                self.max_chunk_overlap,
                max_allowed_overlap,
            )
            self.max_chunk_overlap = max_allowed_overlap
        self.avg_chunk_overlap = (self.max_chunk_overlap + self.min_chunk_overlap) / 2
        self.chunk_set = None
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
        self._set_seed()
        self._create_chunks()
        self._seg_sampler = self.chunk_sampler(self.chunk_set, **self.base_kwargs)
        self.avg_batch_size = self._seg_sampler.avg_batch_size
        logging.info("Sampler ready: avg_batch_size = %.2f", self.avg_batch_size)

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches yielded by the current base sampler.
        """
        return len(self._seg_sampler)

    @property
    def duration_is_random(self) -> bool:
        """
        Whether chunk durations are sampled randomly.

        Returns:
            ``True`` when minimum and maximum chunk lengths differ.
        """
        return self.min_chunk_length != self.max_chunk_length

    def get_random_duration(self) -> float:
        """
        Return one chunk length.

        Returns:
            Random duration in the valid range, or the fixed duration when min and
            max chunk lengths are equal.
        """
        if self.duration_is_random:
            return (
                torch.rand(size=(1,), generator=self.rng).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )
        else:
            return self.min_chunk_length

    def get_random_overlap(self) -> float:
        """
        Return one chunk overlap.

        Returns:
            Random overlap in the valid range, or the fixed overlap when min and max
            overlap are equal.
        """
        if self.min_chunk_overlap == self.max_chunk_overlap:
            return self.min_chunk_overlap
        return (
            torch.rand(size=(1,), generator=self.rng).item()
            * (self.max_chunk_overlap - self.min_chunk_overlap)
            + self.min_chunk_overlap
        )

    def _create_chunks(self) -> None:
        """
        Slices the original segments into smaller chunks.
        Segments shorter than ``min_chunk_length`` are skipped.
        """
        if self.chunk_set is not None:
            del self.chunk_set
            self.chunk_set = None

        num_segments = len(self.segments)
        num_skipped = 0

        logging.info("Creating chunks from %d segments...", num_segments)
        chunks: List[Tuple[str, str, float, float]] = []
        for seg_id, length in zip(self.segments["id"], self._lengths):
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

                chunk = (f"{seg_id}-{i}", seg_id, start, dur)
                chunks.append(chunk)
                increment = dur - self.get_random_overlap()
                if increment <= 0:
                    # Failsafe for numerical errors or unlucky overlap
                    increment = max(0.01, self.min_chunk_length * 0.1)
                start += increment

            # special treatment for last chunk we get from the recording
            remainder = length - start
            chunk_id = f"{seg_id}-{num_chunks - 1}"
            if remainder > self.max_chunk_length:
                # here we discard part of the end
                chunk = (chunk_id, seg_id, start, self.max_chunk_length)
            elif remainder < self.min_chunk_length:
                # here we overlap with second last chunk
                chunk = (
                    chunk_id,
                    seg_id,
                    length - self.min_chunk_length,
                    self.min_chunk_length,
                )
            else:
                # use whatever is left as the last chunk
                chunk = (chunk_id, seg_id, start, remainder)

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

    def __iter__(self) -> "SegChunkSampler":
        """
        Create a chunk set and wrap it with a new base sampler instance.

        Returns:
            This sampler instance.
        """
        init_batch = self.init_batch
        super().__iter__()
        self._create_chunks()
        self._seg_sampler = self.chunk_sampler(self.chunk_set, **self.base_kwargs)
        if init_batch != 0:
            logging.info(
                "Replaying %d batches to resume SegChunkSampler base sampler.",
                init_batch,
            )
            replay_start = time.monotonic()
        self._seg_sampler.set_epoch(self.epoch, init_batch)
        self._seg_sampler = iter(self._seg_sampler)
        if init_batch != 0:
            logging.info(
                "Finished replaying %d batches to resume SegChunkSampler base sampler in %.2f seconds.",
                init_batch,
                time.monotonic() - replay_start,
            )

        return self

    def __next__(self) -> Union[List[str], List[Tuple[str, float, float]]]:
        """
        Return the next batch of chunked segments from the base sampler.

        Returns:
            Segment IDs or ``(segment_id, chunk_start, length)`` tuples.
        """
        batch = next(self._seg_sampler)
        self.batch += 1
        return batch

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Return keyword arguments for nested sampler construction.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Unfiltered keyword arguments. The selected base sampler filters them
            later.
        """
        return kwargs
