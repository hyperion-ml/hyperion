"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser
import logging

import numpy as np
import pandas as pd

import torch
from .hyp_sampler import HypSampler
from .seg_sampler import SegSampler
import torch.distributed as dist


class SegChunkSampler(HypSampler):
    def __init__(
        self,
        seg_set,
        min_chunk_length,
        max_chunk_length=None,
        base_sampler=SegSampler,
        length_column="duration",
        shuffle=False,
        seed=1234,
        **base_kwargs
    ):

        super().__init__(shuffle=shuffle, seed=seed)
        self.seg_set = seg_set
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )
        self.avg_chunk_length = (max_chunk_length + min_chunk_length) / 2
        self.chunk_set = None
        self.length_column = length_column
        self.chunk_sampler = base_sampler
        self.base_kwargs = base_kwargs
        self.base_kwargs["seed"] = seed
        self.base_kwargs["shuffle"] = shuffle
        if "subbase_sampler" in base_kwargs:
            base_kwargs["base_sampler"] = base_kwargs.pop("subbase_sampler")

        self.__iter__()

    def __len__(self):
        return len(self._seg_sampler)

    # def _compute_num_chunks(self, seg_set):
    #     num_chunks = 0
    #     for len in seg_set['duration']:
    #         if len < self.min_chunk_length:
    #             #discard too short sequences
    #             continue

    #         num_chunks += math.ceil(len/self._avg_chunk_length)

    #     self.num_chunks = num_chunks

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
        for id, len in zip(self.seg_set["id"], self.seg_set[self.length_column]):
            if len < self.min_chunk_length:
                # discard too short sequences
                continue

            # making it this way, we get the same number of chunks in all epochs
            num_chunks = math.ceil(len / self.avg_chunk_length)
            start = 0
            for i in range(num_chunks - 1):
                dur = self.get_random_duration()
                chunk = (id, start, dur)
                chunks.append(chunk)
                start += dur

            # special treatment for last chunk we get from the recording
            remainder = len - start
            if remainder > self.max_chunk_length:
                # here we discard part of the end
                chunk = (id, start, self.max_chunk_length)
            elif remainder < self.min_chunk_length:
                # here we overlap with second last chunk
                chunk = (id, len - self.min_chunk_length, self.min_chunk_length)
            else:
                # here the last chunk is what it is left
                chunk = (id, start, remainder)

            chunks.append(chunk)

        self.chunk_set = pd.DataFrame(
            chunks, columns=["id", "chunk_start", self.length_column]
        )

    def __iter__(self):
        super().__iter__()
        self._create_chunks()
        self._seg_sampler = SegSampler(self.chunk_set, self._base_kwargs)
        self._seg_sampler.set_epoch(self.epoch)
        self._seg_sampler.__iter__()

        return self

    def __next__(self):

        return next(self._seg_sampler)
        # if self.batch == self._len:
        #     raise StopIteration

        # start = (self.batch -1)*self.batch_size
        # chunks = self.chunks[start:start+self.batch_size]

        # if self.batch == 0:
        #     logging.info("batch 0 chunks=%s", str(chunks[:10]))

        # self.batch +=1
        # return chunks

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "min_chunk_length",
            "max_chunk_length",
            "length_column",
            "shuffle",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)
