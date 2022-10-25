"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser
import logging

import numpy as np

import torch
from .hyp_sampler import HypSampler
from .seg_sampler import SegSampler
import torch.distributed as dist


class BucketingSegSampler(HypSampler):
    def __init__(
        self,
        seg_set,
        base_sampler=SegSampler,
        num_buckets=10,
        length_column="duration",
        seed=1234,
        **base_kwargs
    ):
        super().__init__(shuffle=False, seed=seed)
        self.seg_set = seg_set
        self.base_sampler = base_sampler
        self.base_kwargs = base_kwargs
        self.base_kwargs["seed"] = seed
        self.num_buckets = num_buckets
        self.length_column = length_column
        self._create_bucket_samplers()
        self._compute_len()
        self.depleted_buckets = torch.zeros((num_buckets,), dtype=torch.bool)

    def create_buckets(self):
        sort_idx = torch.argsort(self.seg_set[self.length_column].values)
        sorted_seg_set = self.seg_set.iloc[sort_idx]
        cum_lengths = torch.cumsum(sorted_seg_set[self.length_column].values)
        bucket_length = cum_lengths[-1] / self.num_buckets
        buckets = []
        for i in range(self.num_buckets):
            bucket_idx = (cum_lengths <= bucket_length) & (cum_lengths > 0)
            bucket_i = sorted_seg_set.loc[bucket_idx]
            buckets.append(bucket_i)
            cum_lengths -= bucket_length

        return buckets

    def _create_bucket_samplers(self):
        buckets = self.create_buckets()
        bucket_samplers = []
        for i in range(self.num_buckets):
            sampler_i = self.base_sampler(buckets[i], self.seed, **self.base_kwargs)
            bucket_samplers.append(sampler_i)

        self.bucket_samplers = bucket_samplers

    def _compute_len(self):
        self._len = 0
        for i in range(self.num_buckets):
            self._len += len(self.bucket_samplers[i])

    def set_epoch(self, epoch):
        for i in range(self.num_buckets):
            self.bucket_samplers[i].set_epoch(epoch)

    def __iter__(self):
        super().__iter__()
        for i in range(self.num_buckets):
            self.bucket_samplers[i].__iter__()

        return self

    def all_buckets_depleted(self):
        return torch.all(self.depleted_buckets).item()

    def __next__(self):

        if self.batch == self._len or self.all_buckets_depleted():
            raise StopIteration

        while True:
            bucket_idx = torch.randint(
                low=0, high=self.num_buckets, size=(1,), generator=self.rng
            ).item()
            if self.depleted_buckets[bucket_idx]:
                continue

            bucket = self.buckets[bucket_idx]
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

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "num_buckets",
            "length_column",
            "shuffle",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)
