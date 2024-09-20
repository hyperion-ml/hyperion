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
from .seg_sampler import SegSampler


class SegChunkSampler(HypSampler):
    def __init__(
        self,
        seg_set: SegmentSet,
        min_chunk_length: float,
        max_chunk_length: Optional[float] = None,
        min_chunk_shift: float = 0,
        max_chunk_shift: float = 0,
        base_sampler: Type[HypSampler] = SegSampler,
        length_name: str = "duration",
        max_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        **base_kwargs,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.seg_set = seg_set
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )
        self.avg_chunk_length = (max_chunk_length + min_chunk_length) / 2
        self.chunk_set = None
        self.length_name = length_name
        self.chunk_sampler = base_sampler
        if "subbase_sampler" in base_kwargs:
            base_kwargs["base_sampler"] = base_kwargs.pop("subbase_sampler")

        self.base_kwargs = base_sampler.filter_args(**base_kwargs)
        self.base_kwargs["seed"] = seed
        self.base_kwargs["shuffle"] = shuffle
        self.base_kwargs["max_batches_per_epoch"] = max_batches_per_epoch

        self.__iter__()
        self.avg_batch_size = self._seg_sampler.avg_batch_size

    def __len__(self):
        return len(self._seg_sampler)

    @property
    def duration_is_random(self):
        return self.min_chunk_length != self.max_chunk_length

    def get_random_duration(self):
        if self.duration_is_random:
            return (
                torch.rand(size=(1,), generator=self.rng).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )
        else:
            return self.min_chunk_length

    def _create_chunks(self):
        chunks = []
        for id, len in zip(self.seg_set["id"], self.seg_set[self.length_name]):
            if len < self.min_chunk_length:
                # discard too short sequences
                continue

            # making it this way, we get the same number of chunks in all epochs
            num_chunks = math.ceil(len / self.avg_chunk_length)
            start = 0
            for i in range(num_chunks - 1):
                remainder = len - start
                if remainder < self.min_chunk_length:
                    remainder = self.min_chunk_length
                    dur = remainder
                    start = len - dur
                else:
                    dur = self.get_random_duration()
                    if dur > remainder:
                        dur = remainder

                chunk = (f"{id}-{i}", id, start, dur)
                chunks.append(chunk)
                start += dur

            # special treatment for last chunk we get from the recording
            remainder = len - start
            chunk_id = f"{id}-{num_chunks - 1}"
            if remainder > self.max_chunk_length:
                # here we discard part of the end
                chunk = (chunk_id, id, start, self.max_chunk_length)
            elif remainder < self.min_chunk_length:
                # here we overlap with second last chunk
                chunk = (
                    chunk_id,
                    id,
                    len - self.min_chunk_length,
                    self.min_chunk_length,
                )
            else:
                # here the last chunk is what it is left
                chunk = (chunk_id, id, start, remainder)

            chunks.append(chunk)

        chunk_set = pd.DataFrame(
            chunks, columns=["id", "seg_id", "chunk_start", self.length_name]
        )
        self.chunk_set = SegmentSet(chunk_set)

    def __iter__(self):
        super().__iter__()
        self._create_chunks()
        self._seg_sampler = self.chunk_sampler(self.chunk_set, **self.base_kwargs)
        self._seg_sampler.set_epoch(self.epoch)
        self._seg_sampler.__iter__()

        return self

    def __next__(self):
        return next(self._seg_sampler)

    @staticmethod
    def filter_args(**kwargs):
        return kwargs
        # valid_args = filter_func_args(SegChunkSampler.__init__, kwargs)
        # base_args = filter_func_args(SegSampler.__init__, kwargs)
        # valid_args.update(base_args)
        # return valid_args

        # valid_args = (
        #     "min_chunk_length",
        #     "max_chunk_length",
        #     "length_name",
        #     "shuffle",
        #     "seed",
        # )

        # return dict((k, kwargs[k]) for k in valid_args if k in kwargs)
