"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils import SegmentSet
from ...utils.misc import filter_func_args
from .hyper_sampler import EmptySamplerError, HyperSampler


class LengthSamplingMethod(str, Enum):
    """
    Enum for length sampling methods used in segment-based sampling.

    Attributes:
        UNIFORM (str): Sample lengths uniformly from the allowed interval.
        MAXIMUM (str): Always use the maximum allowed length.
    """

    UNIFORM = "uniform"
    MAXIMUM = "maximum"

    def __str__(self) -> str:
        """
        Return the enum value.

        Returns:
            String value for the sampling method.
        """
        return self.value

    @staticmethod
    def choices() -> list[str]:
        """
        Returns a list of valid choices for length sampling methods.

        Returns:
            Valid command-line choices for length sampling.
        """
        return [method.value for method in LengthSamplingMethod]


class SegSampler(HyperSampler):
    """
    Segment-based sampler for PyTorch DataLoaders with support for:

    - Fixed or variable batch sizes.
    - Maximum total length per batch (e.g., max total duration).
    - Optional shuffling.
    - Distributed data loading (multi-GPU training).
    - Support for segment-chunk indexing (if "chunk_start" present in `segments`).

    Batching modes:
      - With ``sample_all_segments=True``, a finite per-rank plan covers every
        segment. Variable batching uses singleton batches for overlong segments;
        fixed batching includes final partial batches.
      - Variable batches with ``min_batch_size > 1`` and
        ``sample_all_segments=False`` use a finite candidate queue and yield only
        batches that meet the minimum size.
      - Variable batches with ``min_batch_size == 1`` use a cursor-based packing
        strategy, where singleton batches are valid.
      - Fixed-size batches use contiguous rank-strided segment positions; only this
        mode applies ``drop_last``.

    Attributes:
        segments (SegmentSet): The set of segments to sample from.
        min_batch_size (int): Minimum number of samples per batch.
        max_batch_size (Optional[int]): Maximum number of samples per batch.
        max_batch_length (Optional[float]): Maximum accumulated value for
            ``length_name`` per batch.
        length_name (str): Name of the column in `segments` that defines length (e.g., "duration").
        max_batches_per_epoch (Optional[int]): Optional limit on the number of batches per epoch.
        shuffle (bool): Whether to shuffle segment order each epoch.
        drop_last (bool): Whether to drop the last partial batch in fixed-size mode.
            Ignored in variable-size mode, where ``min_batch_size`` is enforced for
            every yielded batch.
        sample_all_segments (bool): Whether to cover every segment at least once per
            epoch. Shorter distributed ranks repeat batches to match the maximum
            rank-local batch count.
        sort_by_length (bool): Whether to sort batch items by length descending.
        seed (int): Random seed for reproducibility.
        avg_batch_size (float): Estimated average number of samples per batch.
        var_batch_size (bool): Whether batches are constrained by max batch length.
        _num_segments (int): Number of segments in the input table.
        _lengths (np.ndarray): Positional segment lengths used for batch packing.
        _has_chunk_start (bool): Whether segments contain chunk metadata.
        _use_variable_min_batch (bool): Whether variable batching uses the finite
            queue that enforces ``min_batch_size``.
        _variable_batch_availability_validated (bool): Whether unshuffled variable
            batch availability has already been validated.
        _variable_remaining_idxs (Deque[int]): Unconsumed rank-assigned segment
            indices for variable batching with ``min_batch_size > 1``.
        h (Deque[int]): Candidates that cannot extend an
            undersized batch and are discarded after the epoch queue is empty.
        _sample_all_batch_plan (list[list[int]]): Finite sequence of batches used
            when ``sample_all_segments`` is enabled.
    """

    def __init__(
        self,
        segments: SegmentSet,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_batch_length: Optional[float] = None,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sample_all_segments: bool = False,
        sort_by_length: bool = True,
        seed: int = 1234,
    ) -> None:
        """
        Initialize the segment sampler.

        Args:
            segments: Segment metadata table.
            min_batch_size: Minimum number of segments in each batch.
            max_batch_size: Optional maximum number of segments in each batch.
            max_batch_length: Optional maximum padded batch length. When set, batches
                grow until ``max(segment_length) * batch_size`` would exceed it.
            length_name: Column in ``segments`` containing segment lengths.
            max_batches_per_epoch: Optional maximum number of batches per epoch.
                Ignored when ``sample_all_segments`` is enabled.
            shuffle: Whether to shuffle segment order each epoch.
            drop_last: Whether to drop the final partial batch in fixed-size mode.
                Ignored in variable-size mode, where all yielded batches satisfy
                ``min_batch_size``.
            sample_all_segments: Whether to cover every segment at least once per
                epoch. In distributed mode, shorter ranks repeat batches to match
                the maximum rank-local batch count.
            sort_by_length: Whether to sort returned IDs by descending length.
            seed: Base random seed.
        """
        if sample_all_segments:
            max_batches_per_epoch = None
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=shuffle, seed=seed
        )
        if len(segments) == 0:
            raise ValueError("segments must contain at least one row.")
        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be positive, got {min_batch_size}.")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
        if max_batch_length is not None and max_batch_length <= 0:
            raise ValueError(
                f"max_batch_length must be positive, got {max_batch_length}."
            )
        if sample_all_segments and max_batch_length is not None:
            min_batch_size = 1
        if sample_all_segments and max_batch_length is None:
            drop_last = False
        self.segments = segments
        self._num_segments = len(segments)
        self._has_chunk_start = "chunk_start" in segments
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_batch_length = max_batch_length
        self.var_batch_size = max_batch_length is not None
        self.length_name = length_name
        self.drop_last = drop_last
        self.sample_all_segments = sample_all_segments
        self.sort_by_length = sort_by_length
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
        if max_batch_size is not None and min_batch_size > max_batch_size:
            raise ValueError(
                "min_batch_size must be less than or equal to max_batch_size, "
                f"got min_batch_size={min_batch_size}, max_batch_size={max_batch_size}."
            )

        num_segs = self._num_segments
        logging.info(f"Initializing SegSampler with {num_segs} segments.")
        if (
            self.var_batch_size
            and not self.sample_all_segments
            and num_segs < self.world_size
        ):
            raise EmptySamplerError(
                "Variable batching without sample_all_segments requires at least "
                "one segment per distributed rank. Increase the number of segments, "
                "reduce world_size, or enable sample_all_segments."
            )

        if self.var_batch_size:
            logging.info("Variable batch size mode enabled.")
            if not self.sample_all_segments:
                logging.info(
                    f"Filtering segments longer than max_batch_length={max_batch_length}."
                )
                lengths = self._lengths[self._lengths <= max_batch_length]
                num_segs = len(lengths)
                if num_segs == 0:
                    raise EmptySamplerError(
                        f"No segments fit within max_batch_length={max_batch_length}."
                    )
                avg_batch_size = max_batch_length / np.mean(lengths)
                logging.info(
                    f"Average batch size estimated from filtered segments: {avg_batch_size:.2f}"
                )
            else:
                lengths = self._lengths
                avg_batch_size = max_batch_length / np.mean(lengths)
                logging.info(
                    f"Average batch size estimated from all segments: {avg_batch_size:.2f}"
                )
            if max_batch_size is not None and avg_batch_size > max_batch_size:
                logging.warning(
                    f"Average batch size {avg_batch_size:.2f} exceeds max_batch_size={max_batch_size}. "
                    "Adjusting to max_batch_size."
                )
                avg_batch_size = max_batch_size
        else:
            logging.info("Fixed batch size mode enabled.")
            avg_batch_size = min_batch_size
            logging.info(f"Using fixed min_batch_size = {min_batch_size}")
            lengths = self._lengths

        self.avg_batch_size = avg_batch_size

        if self.var_batch_size or drop_last:
            self._len = int(num_segs / (avg_batch_size * self.world_size))
        else:
            self._len = int(math.ceil((num_segs // self.world_size) / avg_batch_size))

        self._permutation = None
        self._sample_all_batch_plan: list[list[int]] = []
        if self.sample_all_segments:
            initial_plan = self._build_sample_all_plan(shuffled=False)
            self._set_sample_all_len(len(initial_plan), synchronize=True)

        logging.info(
            f"Batches per epoch (before max_batches_per_epoch limit): {self._len}"
        )

        if self.max_batches_per_epoch is not None:
            self._len = min(self._len, self.max_batches_per_epoch)
            logging.info(
                f"Limiting batches per epoch to max_batches_per_epoch = {self.max_batches_per_epoch}"
            )

        if self._len == 0:
            raise EmptySamplerError(
                "SegSampler would yield zero batches per rank. Increase the number "
                "of segments, reduce world_size or min_batch_size, or increase "
                "max_batch_length."
            )

        self._use_variable_min_batch = (
            self.var_batch_size
            and self.min_batch_size > 1
            and not self.sample_all_segments
        )
        self._variable_batch_availability_validated = False
        logging.info(
            f"Sampler final configuration: "
            f"min_batch_size={self.min_batch_size}, max_batch_size={self.max_batch_size}, "
            f"min_length={lengths.min():.2f}, max_length={lengths.max():.2f}, "
            f"avg_batch_size={self.avg_batch_size:.2f}, shuffle={shuffle}, "
            f"drop_last={drop_last}, sort_by_length={sort_by_length}, seed={seed}"
        )

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        Returns:
            Number of batches yielded by this rank.
        """
        return self._len

    def _shuffle_segs(self) -> None:
        """Shuffles the segment indices using internal RNG."""
        self._permutation = torch.randperm(
            self._num_segments, generator=self.rng
        ).numpy()

    def _init_variable_min_batch_queue(self) -> None:
        """Initialize the finite per-rank candidate queue for variable batching."""
        self._variable_remaining_idxs = deque(
            int(idx) for idx in self._rank_indices(shuffled=self.shuffle)
        )

    def _rank_indices(self, shuffled: bool = False, pad: bool = False) -> np.ndarray:
        """
        Return the complete, non-overlapping set of segment indices for this rank.

        Args:
            shuffled: Whether to shuffle indices using the current epoch seed.
            pad: Whether to repeat initial indices so every rank receives the same
                number of indices.

        Returns:
            Segment row indices assigned to this rank.
        """
        num_segments = self._num_segments
        if shuffled:
            if self._permutation is None:
                generator = torch.Generator()
                generator.manual_seed(self.seed + 10 * self.epoch)
                indices = torch.randperm(num_segments, generator=generator).numpy()
            else:
                indices = self._permutation
        else:
            indices = np.arange(num_segments)

        if pad:
            num_rank_segs = math.ceil(num_segments / self.world_size)
            total = num_rank_segs * self.world_size
            if total > num_segments:
                indices = np.concatenate((indices, indices[: total - num_segments]))
        else:
            num_rank_segs = num_segments // self.world_size
            total = num_rank_segs * self.world_size
            indices = indices[:total]

        return indices[self.rank : total : self.world_size]

    def _build_sample_all_plan(self, shuffled: bool) -> list[list[int]]:
        """Build this rank's finite coverage plan.

        Args:
            shuffled: Whether to shuffle the segment order for this epoch.

        Returns:
            Batches that cover the rank-assigned padded segment sequence.
        """
        indices = self._rank_indices(shuffled=shuffled, pad=True)
        if not self.var_batch_size:
            return [
                [int(idx) for idx in indices[i : i + self.min_batch_size]]
                for i in range(0, len(indices), self.min_batch_size)
            ]

        batches: list[list[int]] = []
        batch: list[int] = []
        max_length = 0.0
        for idx in indices:
            idx = int(idx)
            seg_length = self._lengths[idx]
            if seg_length > self.max_batch_length:
                if batch:
                    batches.append(batch)
                    batch = []
                    max_length = 0.0
                batches.append([idx])
                continue

            candidate_max_length = max(max_length, seg_length)
            is_full = (
                self.max_batch_size is not None and len(batch) >= self.max_batch_size
            )
            if batch and (
                is_full
                or candidate_max_length * (len(batch) + 1) > self.max_batch_length
            ):
                batches.append(batch)
                batch = []
                max_length = 0.0
                candidate_max_length = seg_length

            batch.append(idx)
            max_length = candidate_max_length

        if batch:
            batches.append(batch)
        return batches

    def _synchronize_max_batch_count(self, count: int) -> int:
        """Return the maximum batch count across distributed ranks.

        Args:
            count: Local batch count.

        Returns:
            Maximum batch count across ranks, or ``count`` outside DDP.
        """
        if self.world_size == 1 or not dist.is_initialized():
            return count
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        value = torch.tensor([count], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return int(value.item())

    def _set_sample_all_len(self, count: int, synchronize: bool) -> None:
        """Set the coverage-mode batch-count target.

        Args:
            count: Local coverage-plan batch count.
            synchronize: Whether to synchronize the plan length across ranks.
        """
        self._len = count
        if synchronize:
            self._len = self._synchronize_max_batch_count(self._len)

    def _refresh_sample_all_plan(self) -> None:
        """Build the current epoch's coverage plan for iteration."""
        self._sample_all_batch_plan = self._build_sample_all_plan(shuffled=self.shuffle)
        self._set_sample_all_len(len(self._sample_all_batch_plan), synchronize=True)

    def _take_variable_min_batch(self, remaining_idxs: Deque[int]) -> list[int]:
        """
        Remove one greedily packed batch candidate from a finite index queue.

        Args:
            remaining_idxs: Segment indices that have not yet been assigned or
                skipped.

        Returns:
            Segment indices in the candidate batch. The result can be smaller than
            ``min_batch_size`` when no remaining segment can extend it.
        """
        idxs = []
        max_length = 0.0
        num_candidates = len(remaining_idxs)

        for _ in range(num_candidates):
            if self.max_batch_size is not None and len(idxs) >= self.max_batch_size:
                break

            idx = remaining_idxs.popleft()
            seg_length = self._lengths[idx]
            if seg_length > self.max_batch_length:
                continue

            candidate_max_length = max(max_length, seg_length)
            if candidate_max_length * (len(idxs) + 1) > self.max_batch_length:
                if len(idxs) >= self.min_batch_size:
                    # The current batch is valid. Keep this candidate at the front
                    # so the next batch considers it before later segments.
                    remaining_idxs.appendleft(idx)
                    break
                remaining_idxs.append(idx)
                continue

            idxs.append(idx)
            max_length = candidate_max_length

        return idxs

    def _can_sample_variable_min_batch(self) -> bool:
        """Check whether this rank can form one valid variable-size batch.

        Returns:
            ``True`` when the current rank-assigned queue contains a batch with at
            least ``min_batch_size`` segments.
        """
        remaining_idxs = deque(self._variable_remaining_idxs)
        while remaining_idxs:
            idxs = self._take_variable_min_batch(remaining_idxs)
            if len(idxs) >= self.min_batch_size:
                return True
        return False

    def _can_sample_variable_singleton_batch(self) -> bool:
        """Check whether this rank can sample one segment within the length cap.

        Returns:
            ``True`` when the cursor sequence contains a non-overlong segment.
        """
        start = self.rank
        for _ in range(self._num_segments):
            idx = self._permutation[start] if self.shuffle else start
            if self._lengths[idx] <= self.max_batch_length:
                return True
            start = (start + self.world_size) % self._num_segments
        return False

    def _validate_variable_batch_availability(self) -> None:
        """Raise when any rank cannot form a valid variable-size batch.

        Raises:
            ValueError: When a rank cannot form a valid variable-size batch.
        """
        can_sample = (
            self._can_sample_variable_min_batch()
            if self._use_variable_min_batch
            else self._can_sample_variable_singleton_batch()
        )
        if self.world_size > 1 and dist.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if dist.get_backend() == "nccl"
                else torch.device("cpu")
            )
            value = torch.tensor([int(can_sample)], device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            can_sample = bool(value.item())

        if not can_sample:
            raise EmptySamplerError(
                "Variable batching cannot form a valid batch on every rank. "
                "Increase max_batch_length, reduce min_batch_size, or enable "
                "sample_all_segments."
            )

    def _sample_variable_min_batch(self) -> list[int]:
        """
        Sample one variable-size batch that satisfies ``min_batch_size``.

        Returns:
            Segment row indices for the next valid batch.

        Raises:
            StopIteration: When no remaining candidates can form a full batch.
        """
        while self._variable_remaining_idxs:
            idxs = self._take_variable_min_batch(self._variable_remaining_idxs)
            if len(idxs) >= self.min_batch_size:
                return idxs

            # No remaining candidate can complete this partial batch, so discard it.

        raise StopIteration

    def __iter__(self) -> "SegSampler":
        """
        Initialize the iterator for batching and replay any requested resume offset.

        Returns:
            This sampler instance.

        Raises:
            ValueError: When variable batching cannot form a valid batch on every
                distributed rank.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        if self.sample_all_segments:
            self._refresh_sample_all_plan()
        elif self.shuffle:
            self._shuffle_segs()

        self.start = self.rank
        if self._use_variable_min_batch:
            self._init_variable_min_batch_queue()
        if (
            self.var_batch_size
            and not self.sample_all_segments
            and (self.shuffle or not self._variable_batch_availability_validated)
        ):
            self._validate_variable_batch_availability()
            self._variable_batch_availability_validated = True
        if resume_batch != 0:
            logging.info("Replaying %d batches to resume SegSampler.", resume_batch)
            replay_start = time.monotonic()
            for _ in range(resume_batch):
                try:
                    next(self)
                except StopIteration:
                    break
            logging.info(
                "Finished replaying %d batches to resume SegSampler in %.2f seconds.",
                self.batch,
                time.monotonic() - replay_start,
            )

        return self

    def _sample_variable_singleton_batch(self) -> list[int]:
        """
        Sample one variable-size batch when singleton batches are valid.

        Returns:
            Segment row indices for the next batch.

        Raises:
            StopIteration: When no segment can be added to the batch.
        """
        idxs = []
        max_length = 0.0
        batch_size = 0
        attempts = 0

        while attempts < self._num_segments and (
            batch_size < self.max_batch_size if self.max_batch_size else True
        ):
            if self.shuffle:
                idx = self._permutation[self.start]
            else:
                idx = self.start

            seg_length = self._lengths[idx]
            candidate_max_length = max(max_length, seg_length)
            if candidate_max_length * (batch_size + 1) > self.max_batch_length:
                if seg_length > self.max_batch_length:
                    self.start = (self.start + self.world_size) % self._num_segments
                    attempts += 1
                    continue
                break

            idxs.append(idx)
            max_length = candidate_max_length
            self.start = (self.start + self.world_size) % self._num_segments
            batch_size += 1
            attempts += 1

        if not idxs:
            raise StopIteration

        return idxs

    def _sample_fixed_batch(self) -> Union[np.ndarray, slice]:
        """
        Sample one fixed-size batch from consecutive rank-strided positions.

        Returns:
            Segment row indices or a slice selecting the next fixed-size batch.
        """
        stop = min(
            self.start + self.world_size * self.min_batch_size, self._num_segments
        )
        if self.shuffle:
            idxs = self._permutation[self.start : stop : self.world_size]
        else:
            idxs = slice(self.start, stop, self.world_size)

        self.start += self.world_size * self.min_batch_size
        return idxs

    def __next__(self) -> Union[np.ndarray, list[tuple[str, float, float]]]:
        """
        Generate the next batch of segment IDs or chunk tuples.

        Returns:
            Segment IDs, or ``(segment_id, chunk_start, length)`` tuples when the
            input segment table contains chunk metadata.
        """
        if self.batch >= self._len:
            raise StopIteration

        if self.sample_all_segments:
            idxs = self._sample_all_batch_plan[
                self.batch % len(self._sample_all_batch_plan)
            ]

        elif self._use_variable_min_batch:
            try:
                idxs = self._sample_variable_min_batch()
            except StopIteration:
                self._init_variable_min_batch_queue()
                idxs = self._sample_variable_min_batch()

        elif self.var_batch_size:
            idxs = self._sample_variable_singleton_batch()

        else:
            idxs = self._sample_fixed_batch()

        # Extract segment IDs from selected indices
        ids = self.segments.iloc[idxs].id.values

        # Optionally sort the batch by segment length (descending)
        if self.sort_by_length:
            sort_idx = np.argsort(self._lengths[idxs])[::-1]
            ids = ids[sort_idx]

        # If this is chunked data, return (seg_id, chunk_start, length) triplets
        if self._has_chunk_start:
            chunks = self.segments.loc[ids]
            seg_ids = [
                (id, s, d)
                for id, s, d in zip(
                    chunks.seg_id, chunks.chunk_start, chunks[self.length_name]
                )
            ]
        else:
            seg_ids = ids

        # Log a sample batch for inspection
        if self.batch == 0:
            logging.info("batch 0 seg_ids=%s", str(seg_ids[:10]))

        self.batch += 1
        return seg_ids

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters keyword arguments relevant to the SegSampler constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Filtered keyword arguments.
        """
        return filter_func_args(SegSampler.__init__, kwargs, skip={"segments"})

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """
        Adds SegSampler-specific arguments to a parser.

        Args:
            parser (ArgumentParser): The parser to extend.
            prefix (Optional[str]): If provided, adds a subparser group under this prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help="Minimum number of samples (segments) per batch on each GPU. "
            "Required for both fixed and variable batch sizing.",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help="Optional cap on the number of segments per batch. "
            "Useful when using variable-length batching. If not set, batching is only constrained by max-batch-length.",
        )

        parser.add_argument(
            "--max-batch-length",
            type=float,
            default=None,
            help="Maximum total duration (sum of segment lengths) allowed per batch, in seconds. "
            "If set, enables variable-length batching where batch size adapts based on segment duration.",
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="If enabled, drops the final partial batch in fixed-size mode. "
            "Ignored by variable-length batching, which always enforces --min-batch-size.",
        )
        parser.add_argument(
            "--sample-all-segments",
            default=False,
            action=ActionYesNo,
            help="Cover every segment at least once per epoch. With variable "
            "batching, overlong segments become singleton batches; with fixed "
            "batching, final partial batches are retained.",
        )
        parser.add_argument(
            "--sort-by-length",
            default=True,
            action=ActionYesNo,
            help="If enabled, sorts batch items by duration in descending order.",
        )
        parser.add_argument(
            "--max-batches-per-epoch",
            type=int,
            default=None,
            help="Optional cap on the total number of batches per epoch. "
            "Useful for debugging or limiting training duration.",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="Shuffle the dataset at the start of each epoch. "
            "Recommended for training to ensure randomness.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Random seed used for shuffling. Ensures reproducibility across runs.",
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help="Name of the column in the segment table that represents segment length (e.g., 'duration'). "
            "This is used to compute variable-length batches.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
