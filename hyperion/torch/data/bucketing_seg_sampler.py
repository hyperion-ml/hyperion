"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Optional, Type

import numpy as np
import torch
import torch.distributed as dist

from ...utils import SegmentSet
from .hyp_sampler import HypSampler
from .seg_sampler import SegSampler


class BucketingSegSampler(HypSampler):
    """
    A sampler that groups segments into buckets based on length and samples batches
    from each bucket using a base sampler (e.g., SegSampler). This improves efficiency
    and minimizes padding when using variable-length inputs.

    Supports distributed training and randomly selects from buckets per batch.

    Args:
        segments (SegmentSet): Full set of segments to bucket and sample from.
        base_sampler (Type[HypSampler]): The sampler class used within each bucket.
        num_buckets (int): Number of buckets to divide the data into by length.
        length_name (str): Name of the column in `segments` that holds duration/length.
        max_batches_per_epoch (Optional[int]): Cap on number of batches per epoch.
        seed (int): Random seed for bucket selection.
        **base_kwargs: Keyword arguments passed to the base sampler.
    """

    def __init__(
        self,
        segments: SegmentSet,
        base_sampler: Type[HypSampler] = SegSampler,
        num_buckets: int = 10,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        seed: int = 1234,
        **base_kwargs,
    ):
        super().__init__(
            max_batches_per_epoch=max_batches_per_epoch, shuffle=False, seed=seed
        )
        self.segments = segments
        self.base_sampler = base_sampler
        self.base_kwargs = base_sampler.filter_args(**base_kwargs)
        self.base_kwargs["seed"] = seed
        self.num_buckets = num_buckets
        self.length_name = length_name
        logging.info(
            "Initializing BucketingSegSampler with %d segments into %d buckets",
            len(self.segments),
            self.num_buckets,
        )
        # Create bucketed samplers and compute epoch length
        self._create_bucket_samplers()
        self._compute_len()
        # Track which buckets are exhausted
        self.depleted_buckets = torch.zeros((num_buckets,), dtype=torch.bool)

    def create_buckets(self):
        """
        Sorts segments by length and divides them into approximately equal total-length buckets.

        Returns:
            List[pd.DataFrame]: A list of segment DataFrames, one per bucket.
        """
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
            buckets.append(bucket)
            current_start = current_end

        return buckets

    def _create_bucket_samplers(self):
        """
        Creates a base sampler (e.g., SegSampler) for each bucket.
        """
        buckets = self.create_buckets()
        bucket_samplers = []
        for i in range(self.num_buckets):
            sampler_i = self.base_sampler(buckets[i], **self.base_kwargs)
            bucket_samplers.append(sampler_i)

        self.bucket_samplers = bucket_samplers

    def __len__(self):
        """Returns the total number of batches in an epoch across all buckets."""
        return self._len

    def _compute_len(self):
        """
        Computes the total number of batches for the current epoch.
        """
        self._len = 0
        for i, sampler in enumerate(self.bucket_samplers):
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

        logging.info("Total batches per epoch: %d", self._len)

    def set_epoch(self, epoch: int, batch: int = 0):
        """
        Sets the epoch value for reproducible shuffling in each bucket's sampler.

        Args:
            epoch (int): Current epoch number.
            batch (int): Starting batch offset within epoch.
        """
        super().set_epoch(epoch, batch)
        for i in range(self.num_buckets):
            self.bucket_samplers[i].set_epoch(epoch, batch)

    def __iter__(self):
        """
        Resets internal counters and bucket states for the start of a new epoch.

        Returns:
            Iterator
        """
        super().__iter__()
        self.depleted_buckets[:] = False
        for i in range(self.num_buckets):
            self.bucket_samplers[i] = iter(self.bucket_samplers[i])

        return self

    def all_buckets_depleted(self):
        """
        Checks whether all buckets are exhausted.

        Returns:
            bool: True if all buckets are depleted.
        """
        return torch.all(self.depleted_buckets).item()

    def __next__(self):
        """
        Samples a batch from a randomly selected non-depleted bucket.

        Returns:
            list[str] | list[tuple]: Segment IDs or chunk triplets for the batch.

        Raises:
            StopIteration: When all buckets are depleted or batch limit is reached.
        """
        if self.batch == self._len or self.all_buckets_depleted():
            raise StopIteration

        while True:
            # Randomly select a bucket
            bucket_idx = torch.randint(
                low=0, high=self.num_buckets, size=(1,), generator=self.rng
            ).item()
            if self.depleted_buckets[bucket_idx]:
                continue

            bucket = self.bucket_samplers[bucket_idx]
            try:
                batch = next(bucket)
                break
            except StopIteration:
                self.depleted_buckets[bucket_idx] = True
                if self.all_buckets_depleted():
                    raise StopIteration()

        if self.batch == 0:
            logging.info("batch 0 chunks=%s", str(batch[:10]))

        self.batch += 1
        return batch

    @property
    def avg_batch_size(self):
        """
        Computes the average batch size across all buckets.

        Returns:
            float: Average batch size.
        """
        avg_batch_size = 0
        for sampler in self.bucket_samplers:
            avg_batch_size += sampler.avg_batch_size

        avg_batch_size /= self.num_buckets
        return avg_batch_size

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters and returns arguments relevant to BucketingSegSampler.

        Returns:
            dict: Filtered arguments.
        """
        return kwargs
