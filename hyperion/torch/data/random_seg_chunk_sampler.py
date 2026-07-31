"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import SegmentSet
from ...utils.misc import filter_func_args
from .hyper_sampler import EmptySamplerError, HyperSampler


class RandomSegChunkSampler(HyperSampler):
    """Randomly samples chunks from source segments without materializing chunks.

    Every batch independently samples source segments with replacement, then
    samples a valid start position for each chunk. Consequently, this sampler
    does not guarantee that each possible chunk is visited once per epoch.

    Attributes:
        segments (SegmentSet): Source segment table.
        min_chunk_length (float): Minimum sampled chunk duration.
        max_chunk_length (float): Maximum sampled chunk duration.
        min_batch_size (int): Minimum number of chunks per GPU batch.
        max_batch_size (int): Maximum number of chunks per GPU batch.
        avg_batch_size (float): Average per-GPU batch size used to estimate the
            epoch length.
        var_batch_size (bool): Whether sampled chunk duration changes batch size.
        num_chunks_per_seg_epoch (Union[str, int, float]): Target number of
            chunks per eligible source segment and epoch.
        num_chunks_per_seg (int): Number of chunks sampled from each selected
            source segment in a batch.
        length_name (str): Name of the source segment length column.
        _seg_ids (np.ndarray): Cached source segment IDs.
        _seg_lengths (np.ndarray): Cached source segment lengths.
        _sorted_length_idxs (np.ndarray): Source row indices sorted by length.
        _sorted_lengths (np.ndarray): Cached source lengths in ascending order.
        _num_sampleable_segments (int): Number of segments long enough for the
            minimum chunk duration.
    """

    def __init__(
        self,
        segments: SegmentSet,
        min_chunk_length: float,
        max_chunk_length: Optional[float] = None,
        *,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_batch_length: Optional[float] = None,
        num_chunks_per_seg_epoch: Union[str, int, float] = "auto",
        num_chunks_per_seg: int = 1,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        iters_per_epoch: Optional[Union[str, int, float]] = None,
        batch_size: Optional[int] = None,
        seed: int = 1234,
    ) -> None:
        """Initialize the random chunk sampler.

        Args:
            segments: Source segment table to sample from.
            min_chunk_length: Minimum sampled chunk duration.
            max_chunk_length: Maximum sampled chunk duration. If ``None``, uses
                ``min_chunk_length``.
            min_batch_size: Minimum number of chunks per GPU batch.
            max_batch_size: Optional maximum number of chunks per GPU batch.
            max_batch_length: Optional maximum total chunk duration per GPU batch.
            num_chunks_per_seg_epoch: Target chunks per eligible source segment
                and epoch, or ``"auto"`` to infer it from average duration.
            num_chunks_per_seg: Number of chunks sampled per selected source
                segment within a batch.
            length_name: Name of the column containing source segment lengths.
            max_batches_per_epoch: Optional maximum number of yielded batches.
            shuffle: Whether to vary the random stream by epoch.
            iters_per_epoch: Deprecated alias for ``num_chunks_per_seg_epoch``.
            batch_size: Deprecated alias for ``min_batch_size``.
            seed: Base random seed.
        """
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        if len(segments) == 0:
            raise EmptySamplerError("segments must contain at least one row.")

        try:
            min_chunk_length = float(min_chunk_length)
            if max_chunk_length is not None:
                max_chunk_length = float(max_chunk_length)
            if max_batch_length is not None:
                max_batch_length = float(max_batch_length)
        except (TypeError, ValueError) as error:
            raise ValueError("Chunk and batch lengths must be numeric.") from error

        length_values = [min_chunk_length]
        if max_chunk_length is not None:
            length_values.append(max_chunk_length)
        if max_batch_length is not None:
            length_values.append(max_batch_length)
        if not all(math.isfinite(value) for value in length_values):
            raise ValueError("Chunk and batch lengths must be finite.")
        if min_chunk_length <= 0:
            raise ValueError(
                f"min_chunk_length must be positive, got {min_chunk_length}."
            )

        self.segments = segments
        self.length_name = length_name
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )
        if self.max_chunk_length < self.min_chunk_length:
            raise ValueError(
                "max_chunk_length must be greater than or equal to min_chunk_length."
            )
        if max_batch_length is not None and max_batch_length <= 0:
            raise ValueError(
                f"max_batch_length must be positive, got {max_batch_length}."
            )
        if num_chunks_per_seg <= 0:
            raise ValueError(
                f"num_chunks_per_seg must be positive, got {num_chunks_per_seg}."
            )

        try:
            self._seg_lengths = np.asarray(segments[length_name].values, dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{length_name} must contain numeric source segment lengths."
            ) from error
        if not np.all(np.isfinite(self._seg_lengths)) or np.any(self._seg_lengths <= 0):
            raise ValueError(
                f"{length_name} must contain finite, positive source segment lengths."
            )
        max_segment_length = float(np.max(self._seg_lengths))
        if self.max_chunk_length > max_segment_length:
            logging.warning(
                "Clamping max_chunk_length=%.2f to the longest segment length %.2f.",
                self.max_chunk_length,
                max_segment_length,
            )
            self.max_chunk_length = max_segment_length
        self._seg_ids = np.asarray(segments["id"].values)
        self._sorted_length_idxs = np.argsort(self._seg_lengths)
        self._sorted_lengths = self._seg_lengths[self._sorted_length_idxs]
        first_sampleable_idx = np.searchsorted(
            self._sorted_lengths, self.min_chunk_length, side="left"
        )
        self._num_sampleable_segments = len(self._seg_lengths) - first_sampleable_idx
        if self._num_sampleable_segments == 0:
            raise EmptySamplerError(
                "No segment is long enough for min_chunk_length="
                f"{self.min_chunk_length}."
            )

        if batch_size is not None:
            if min_batch_size != 1:
                logging.warning(
                    "Both batch_size and min_batch_size provided; using batch_size as min_batch_size."
                )
            min_batch_size = batch_size
        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be positive, got {min_batch_size}.")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

        if max_batch_length is None:
            max_batch_size_from_length = int(
                min_batch_size * self.max_chunk_length / self.min_chunk_length
            )
        else:
            min_batch_size = int(max_batch_length / self.max_chunk_length)
            max_batch_size_from_length = int(max_batch_length / self.min_chunk_length)
        max_batch_size = (
            max_batch_size_from_length
            if max_batch_size is None
            else min(max_batch_size_from_length, max_batch_size)
        )
        if min_batch_size <= 0:
            raise ValueError(
                "max_batch_length must be at least max_chunk_length to fit one chunk."
            )
        if min_batch_size > max_batch_size:
            raise ValueError(
                "min_batch_size must be less than or equal to max_batch_size, "
                f"got min_batch_size={min_batch_size}, max_batch_size={max_batch_size}."
            )

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.avg_batch_size = (min_batch_size + max_batch_size) / 2
        self.var_batch_size = min_batch_size != max_batch_size
        self.num_chunks_per_seg = num_chunks_per_seg

        num_chunks_per_seg_epoch = (
            iters_per_epoch if iters_per_epoch is not None else num_chunks_per_seg_epoch
        )
        self._set_num_chunks_per_seg_epoch(num_chunks_per_seg_epoch)
        self._compute_len()
        logging.info(
            "RandomSegChunkSampler: batches/epoch=%d min-batch-size=%d "
            "max-batch-size=%d avg-batch-size/gpu=%.2f "
            "chunks/(eligible-seg*epoch)=%s",
            self._len,
            self.min_batch_size,
            self.max_batch_size,
            self.avg_batch_size,
            self.num_chunks_per_seg_epoch,
        )

    def _set_seed(self) -> None:
        """Initialize the random generator for the current epoch and rank."""
        epoch_offset = 10 * self.epoch if self.shuffle else 0
        self.rng.manual_seed(self.seed + epoch_offset + 100 * self.rank)

    def _set_num_chunks_per_seg_epoch(
        self, num_chunks_per_seg_epoch: Union[str, int, float]
    ) -> None:
        """Set the target number of chunks sampled per eligible segment and epoch.

        Args:
            num_chunks_per_seg_epoch: Positive target count, or ``"auto"`` to
                infer it from average eligible segment duration.
        """
        if num_chunks_per_seg_epoch == "auto":
            avg_seg_length = np.mean(
                self._sorted_lengths[-self._num_sampleable_segments :]
            )
            avg_chunk_length = (self.min_chunk_length + self.max_chunk_length) / 2
            self.num_chunks_per_seg_epoch = math.ceil(avg_seg_length / avg_chunk_length)
            return
        if not isinstance(num_chunks_per_seg_epoch, (int, float)):
            raise ValueError(
                "num_chunks_per_seg_epoch must be a positive number or 'auto', "
                f"got {num_chunks_per_seg_epoch!r}."
            )
        if not math.isfinite(num_chunks_per_seg_epoch) or num_chunks_per_seg_epoch <= 0:
            raise ValueError(
                "num_chunks_per_seg_epoch must be finite and positive, "
                f"got {num_chunks_per_seg_epoch}."
            )
        self.num_chunks_per_seg_epoch = num_chunks_per_seg_epoch

    def _compute_len(self) -> None:
        """Compute the estimated per-rank number of batches in an epoch."""
        self._len = math.ceil(
            self.num_chunks_per_seg_epoch
            * self._num_sampleable_segments
            / self.avg_batch_size
            / self.world_size
        )
        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)

    def __len__(self) -> int:
        """Return the per-rank number of batches in an epoch.

        Returns:
            Number of batches yielded by this rank.
        """
        return self._len

    def _sample_chunk_length(self) -> float:
        """Sample the duration shared by all chunks in the next batch.

        Returns:
            Sampled chunk duration.
        """
        if not self.var_batch_size:
            return self.min_chunk_length
        return (
            self.min_chunk_length
            + (self.max_chunk_length - self.min_chunk_length)
            * torch.rand((), generator=self.rng).item()
        )

    def _compute_batch_size(self, chunk_length: float) -> int:
        """Compute the batch size allowed by a sampled chunk duration.

        Args:
            chunk_length: Shared duration of chunks in the batch.

        Returns:
            Number of chunks to target for the batch.
        """
        batch_size = int(self.min_batch_size * self.max_chunk_length / chunk_length)
        return min(batch_size, self.max_batch_size)

    def _sample_segment_idxs(
        self, num_segments: int, chunk_length: float
    ) -> np.ndarray:
        """Sample source segment row indices that can contain a chunk.

        Args:
            num_segments: Number of source segments to sample.
            chunk_length: Required chunk duration.

        Returns:
            Positional indices into ``segments``.
        """
        first_eligible_idx = np.searchsorted(
            self._sorted_lengths, chunk_length, side="left"
        )
        eligible_idxs = self._sorted_length_idxs[first_eligible_idx:]
        sampled_positions = torch.randint(
            len(eligible_idxs), (num_segments,), generator=self.rng
        ).numpy()
        return eligible_idxs[sampled_positions]

    def _sample_chunks(
        self, batch_size: int, chunk_length: float
    ) -> List[Tuple[str, float, float]]:
        """Sample chunks from uniformly selected eligible source segments.

        Args:
            batch_size: Target number of chunks for the batch.
            chunk_length: Shared duration of all chunks in the batch.

        Returns:
            Tuples of ``(segment_id, start_time, chunk_length)``.
        """
        num_segments = math.ceil(batch_size / self.num_chunks_per_seg)
        seg_idxs = self._sample_segment_idxs(num_segments, chunk_length)
        lengths = torch.as_tensor(self._seg_lengths[seg_idxs], dtype=torch.float32)
        starts = (lengths - chunk_length) * torch.rand(
            (self.num_chunks_per_seg, num_segments), generator=self.rng
        )
        return [
            (segment_id, start.item(), chunk_length)
            for segment_id, segment_starts in zip(self._seg_ids[seg_idxs], starts.T)
            for start in segment_starts
        ]

    def __next__(self) -> List[Tuple[str, float, float]]:
        """Sample and return a batch of chunks.

        Returns:
            Tuples of ``(segment_id, start_time, chunk_length)``.
        """
        if self.batch >= self._len:
            raise StopIteration
        chunk_length = self._sample_chunk_length()
        batch_size = self._compute_batch_size(chunk_length)
        chunks = self._sample_chunks(batch_size, chunk_length)
        if self.batch == 0:
            logging.info("batch 0 chunks=%s", chunks[:10])
        self.batch += 1
        return chunks

    def __iter__(self) -> "RandomSegChunkSampler":
        """Return the sampler as its own iterator and replay resumed batches.

        Returns:
            This sampler instance.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        if resume_batch != 0:
            logging.info(
                "Replaying %d batches to resume RandomSegChunkSampler.", resume_batch
            )
            replay_start = time.monotonic()
            for _ in range(min(resume_batch, self._len)):
                next(self)
            logging.info(
                "Finished replaying %d batches to resume RandomSegChunkSampler in %.2f seconds.",
                self.batch,
                time.monotonic() - replay_start,
            )
        return self

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by the sampler constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dictionary containing only constructor-compatible arguments.
        """
        return filter_func_args(RandomSegChunkSampler.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Add command-line arguments for configuring the sampler.

        Args:
            parser: Argument parser to populate.
            prefix: Optional key under which to nest these arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--min-chunk-length", type=float, default=4.0)
        parser.add_argument("--max-chunk-length", type=float, default=None)
        parser.add_argument("--min-batch-size", type=int, default=1)
        parser.add_argument("--max-batch-size", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument(
            "--max-batch-length",
            "--max-batch-duration",
            dest="max_batch_length",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--iters-per-epoch",
            type=lambda value: value if value == "auto" else float(value),
            default=None,
        )
        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            type=lambda value: value if value == "auto" else float(value),
            default="auto",
        )
        parser.add_argument("--num-chunks-per-seg", type=int, default=1)
        parser.add_argument("--length-name", default="duration")
        parser.add_argument("--max-batches-per-epoch", type=int, default=None)
        parser.add_argument("--shuffle", action=ActionYesNo)
        parser.add_argument("--seed", type=int, default=1234)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
