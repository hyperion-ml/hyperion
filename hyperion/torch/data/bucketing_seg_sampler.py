"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from ...utils import SegmentSet
from .hyper_sampler import EmptySamplerError, HyperSampler
from .seg_sampler import SegSampler


class BucketingSegSampler(HyperSampler):
    """
    A sampler that groups segments into buckets based on length and samples batches
    from each bucket using a base sampler (e.g., SegSampler). This improves efficiency
    and minimizes padding when using variable-length inputs.

    Supports distributed training and randomly selects from buckets per batch.

    Attributes:
        segments (SegmentSet): Segments to bucket after optional maximum-length
            filtering.
        base_sampler (Type[HyperSampler]): The sampler class used within each bucket.
        num_buckets (int): Number of buckets to divide the data into by length.
        length_name (str): Name of the column in `segments` that holds duration/length.
        max_batch_length (Optional[float]): Maximum segment length for regular
            buckets and child samplers. Overlength segments are discarded unless
            coverage mode is enabled.
        sample_all_segments (bool): Whether every segment, including overlength
            segments, must be sampled during an epoch.
        base_kwargs (Dict[str, Any]): Keyword arguments passed to the base sampler.
        bucket_samplers (List[HyperSampler]): Sampler instance for each non-empty bucket.
        _active_bucket_idxs (List[int]): Non-depleted bucket indices available for
            random selection.
        _overlength_segments (Optional[SegmentSet]): Segments longer than
            ``max_batch_length`` that form an extra coverage-mode bucket.
    """

    def __init__(
        self,
        segments: SegmentSet,
        base_sampler: Type[HyperSampler] = SegSampler,
        num_buckets: int = 10,
        length_name: str = "duration",
        max_batch_length: Optional[float] = None,
        sample_all_segments: bool = False,
        max_batches_per_epoch: Optional[int] = None,
        seed: int = 1234,
        **base_kwargs: Any,
    ) -> None:
        """
        Initialize the bucketing sampler.

        Args:
            segments: Segment metadata table to bucket.
            base_sampler: Sampler class used inside each bucket.
            num_buckets: Requested number of length buckets.
            length_name: Column in ``segments`` containing segment lengths.
            max_batch_length: Optional maximum segment length for regular buckets.
                Longer segments are removed unless coverage mode is enabled, and the
                value is passed to child samplers.
            sample_all_segments: Whether all segments must be sampled. Overlength
                segments form an extra bucket when coverage is enabled.
            max_batches_per_epoch: Optional maximum number of batches per epoch.
                Ignored when ``sample_all_segments`` is enabled.
            seed: Base random seed.
            **base_kwargs: Additional arguments for the base sampler.
        """
        if sample_all_segments:
            max_batches_per_epoch = None
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=True, seed=seed
        )
        if num_buckets <= 0:
            raise ValueError(f"num_buckets must be positive, got {num_buckets}.")
        if len(segments) == 0:
            raise ValueError("segments must contain at least one row.")
        try:
            lengths = np.asarray(segments[length_name].values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{length_name} must contain numeric segment lengths."
            ) from error
        valid_lengths = np.isfinite(lengths)
        if not np.all(valid_lengths):
            logging.warning(
                "Discarding %d segments with non-finite %s values.",
                np.count_nonzero(~valid_lengths),
                length_name,
            )
            segments = segments.iloc[np.flatnonzero(valid_lengths)]
            if len(segments) == 0:
                raise ValueError(
                    f"No segments have finite {length_name} values."
                )
            lengths = lengths[valid_lengths]
        if np.any(lengths <= 0):
            raise ValueError(
                f"{length_name} must contain positive segment lengths."
            )
        if max_batch_length is not None and max_batch_length <= 0:
            raise ValueError(
                f"max_batch_length must be positive, got {max_batch_length}."
            )
        self.sample_all_segments = sample_all_segments
        self._overlength_segments: Optional[SegmentSet] = None
        if max_batch_length is not None:
            in_range_segments = segments.loc[
                segments[length_name] <= max_batch_length
            ]
            if self.sample_all_segments:
                self._overlength_segments = segments.loc[
                    segments[length_name] > max_batch_length
                ]
            segments = in_range_segments
            if (
                len(segments) == 0
                and self._overlength_segments is not None
                and len(self._overlength_segments) == 0
            ):
                raise ValueError(
                    f"No segments fit within max_batch_length={max_batch_length}."
                )
        self.segments = segments
        self.base_sampler = base_sampler
        base_kwargs["seed"] = seed
        base_kwargs["length_name"] = length_name
        base_kwargs["sample_all_segments"] = self.sample_all_segments
        if max_batch_length is not None:
            base_kwargs["max_batch_length"] = max_batch_length
        if "subbase_sampler" in base_kwargs:
            base_kwargs["base_sampler"] = base_kwargs.pop("subbase_sampler")
        self.base_kwargs = base_sampler.filter_args(**base_kwargs)
        self.num_buckets = num_buckets
        self.length_name = length_name
        self.max_batch_length = max_batch_length
        logging.info(
            "Initializing BucketingSegSampler with %d segments into %d buckets",
            len(self.segments),
            self.num_buckets,
        )
        # Create bucketed samplers and compute epoch length
        self._create_bucket_samplers()
        self._compute_len()
        self._active_bucket_idxs = list(range(len(self.bucket_samplers)))

    def create_buckets(self) -> List[SegmentSet]:
        """
        Sorts segments by length and divides them into approximately equal total-length buckets.

        Returns:
            Non-empty segment buckets.
        """
        if len(self.segments) == 0:
            return []

        sort_idx = np.argsort(self.segments[self.length_name].values)
        sorted_segments = self.segments.iloc[sort_idx]
        cum_lengths = np.cumsum(sorted_segments[self.length_name].values, axis=0)
        bucket_length = cum_lengths[-1] / self.num_buckets
        buckets = []
        current_start = 0
        for i in range(self.num_buckets):
            current_end = np.searchsorted(
                cum_lengths, (i + 1) * bucket_length, side="right"
            )
            bucket = sorted_segments.iloc[current_start:current_end]
            logging.info(
                "Bucket %d: %d segments, total %s = %.2f",
                i,
                len(bucket),
                self.length_name,
                bucket[self.length_name].sum(),
            )
            if len(bucket) > 0:
                buckets.append(bucket)
            current_start = current_end

        return buckets

    def _create_bucket_samplers(self) -> None:
        """
        Creates a base sampler (e.g., SegSampler) for each bucket.
        """
        buckets = self.create_buckets()
        if self._overlength_segments is not None and len(self._overlength_segments):
            logging.info(
                "Overlength bucket: %d segments, total %s = %.2f",
                len(self._overlength_segments),
                self.length_name,
                self._overlength_segments[self.length_name].sum(),
            )
            buckets.append(self._overlength_segments)
        bucket_samplers = []
        for bucket in buckets:
            try:
                sampler_i = self.base_sampler(bucket, **self.base_kwargs)
            except EmptySamplerError as error:
                logging.info("Discarding bucket with zero batches: %s", error)
                continue
            bucket_samplers.append(sampler_i)

        if not bucket_samplers:
            raise EmptySamplerError("No buckets can yield a batch on every rank.")
        self.bucket_samplers = bucket_samplers
        self.num_buckets = len(bucket_samplers)

    def __len__(self) -> int:
        """
        Return the target total number of batches in an epoch across all buckets.

        Returns:
            Number of batches yielded by this rank.
        """
        return self._len

    def _compute_len(self, bucket_idxs: Optional[List[int]] = None) -> None:
        """
        Compute the target total number of batches for the current epoch.

        Args:
            bucket_idxs: Optional child bucket indices to include.
        """
        self._len = 0
        if bucket_idxs is None:
            bucket_idxs = list(range(len(self.bucket_samplers)))
        for i in bucket_idxs:
            sampler = self.bucket_samplers[i]
            bucket_len = len(sampler)
            self._len += bucket_len
            logging.info("Bucket %d contributes %d batches", i, bucket_len)

        logging.info("Total batches across all buckets: %d", self._len)
        if self.max_batches_per_epoch is not None:
            if self._len > self.max_batches_per_epoch:
                logging.info(
                    "Truncating total batches to max_batches_per_epoch=%d",
                    self.max_batches_per_epoch,
                )
                self._len = self.max_batches_per_epoch

        logging.info("Batches per epoch: %d", self._len)

    def set_epoch(self, epoch: int, batch: int = 0) -> None:
        """
        Sets the epoch value for reproducible shuffling in each bucket's sampler.

        Args:
            epoch (int): Current epoch number.
            batch (int): Global starting batch offset within the bucketing sampler.
        """
        super().set_epoch(epoch, batch)
        for i in range(self.num_buckets):
            self.bucket_samplers[i].set_epoch(epoch, 0)

    def __iter__(self) -> "BucketingSegSampler":
        """
        Reset internal counters and replay any requested global resume offset.

        Returns:
            This sampler instance.
        """
        resume_batch = self.init_batch
        self.init_batch = 0
        super().__iter__()
        self._active_bucket_idxs = []
        for i in range(self.num_buckets):
            self.bucket_samplers[i].set_epoch(self.epoch, 0)
            try:
                self.bucket_samplers[i] = iter(self.bucket_samplers[i])
            except EmptySamplerError as error:
                logging.info("Discarding bucket %d for this epoch: %s", i, error)
                continue
            self._active_bucket_idxs.append(i)
        if not self._active_bucket_idxs:
            raise EmptySamplerError("No buckets can yield a batch on every rank.")
        self._compute_len(self._active_bucket_idxs)

        if resume_batch != 0:
            logging.info(
                "Replaying %d batches to resume BucketingSegSampler.", resume_batch
            )
            replay_start = time.monotonic()
            for _ in range(min(resume_batch, self._len)):
                try:
                    self._sample_from_bucket()
                except StopIteration:
                    break
                self.batch += 1
            logging.info(
                "Finished replaying %d batches to resume BucketingSegSampler in %.2f seconds.",
                self.batch,
                time.monotonic() - replay_start,
            )

        return self

    def all_buckets_depleted(self) -> bool:
        """
        Checks whether all buckets are exhausted.

        Returns:
            bool: True if all buckets are depleted.
        """
        return not self._active_bucket_idxs

    def _sample_from_bucket(self) -> Union[List[str], List[tuple]]:
        """
        Sample one batch from a randomly selected non-depleted bucket.

        Returns:
            Segment IDs or chunk triplets for the batch.
        """
        if self.all_buckets_depleted():
            raise StopIteration

        while self._active_bucket_idxs:
            active_pos = torch.randint(
                low=0,
                high=len(self._active_bucket_idxs),
                size=(1,),
                generator=self.rng,
            ).item()
            bucket_idx = self._active_bucket_idxs[active_pos]

            bucket = self.bucket_samplers[bucket_idx]
            try:
                return next(bucket)
            except StopIteration:
                self._active_bucket_idxs[active_pos] = self._active_bucket_idxs[-1]
                self._active_bucket_idxs.pop()

        raise StopIteration

    def __next__(self) -> Union[List[str], List[tuple]]:
        """
        Samples a batch from a randomly selected non-depleted bucket.

        Returns:
            list[str] | list[tuple]: Segment IDs or chunk triplets for the batch.

        Raises:
            StopIteration: When all buckets are depleted or batch limit is reached.
        """
        if self.batch >= self._len:
            raise StopIteration

        batch = self._sample_from_bucket()

        if self.batch == 0:
            logging.info("batch 0 chunks=%s", str(batch[:10]))

        self.batch += 1
        return batch

    @property
    def avg_batch_size(self) -> float:
        """
        Computes the average batch size across all buckets.

        Returns:
            float: Average batch size.
        """
        avg_batch_size = 0.0
        num_batches = 0
        for sampler in self.bucket_samplers:
            bucket_len = len(sampler)
            avg_batch_size += sampler.avg_batch_size * bucket_len
            num_batches += bucket_len

        return avg_batch_size / num_batches

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters and returns arguments relevant to BucketingSegSampler.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Unfiltered keyword arguments. The selected base sampler filters them
            later.
        """
        return kwargs
