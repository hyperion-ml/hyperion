"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .hyp_sampler import HypSampler


class SegSampler(HypSampler):
    def __init__(
        self,
        seg_set,
        min_batch_size=1,
        max_batch_size=None,
        max_batch_length=None,
        length_name="duration",
        shuffle=False,
        drop_last=False,
        seed=1234,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.seg_set = seg_set
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_batch_length = max_batch_length
        self.var_batch_size = max_batch_length is not None
        self.length_name = length_name
        if self.var_batch_size:
            avg_batch_size = max_batch_length / np.mean(
                self.seg_set[self.length_name])
        else:
            avg_batch_size = min_batch_size

        self.avg_batch_size = avg_batch_size

        if drop_last:
            self._len = int(
                len(self.seg_set) / (avg_batch_size * self.world_size))
        else:
            self._len = int(
                math.ceil(
                    (len(self.seg_set) // self.world_size) / avg_batch_size))

        self._permutation = None

    def __len__(self):
        return self._len

    def _shuffle_segs(self):
        self._permutation = torch.randperm(len(self.seg_set),
                                           generator=self.rng).numpy()

    def __iter__(self):
        super().__iter__()
        if self.shuffle:
            self._shuffle_segs()

        self.start = self.rank
        return self

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        if self.var_batch_size:
            column_idx = self.seg_set.columns.get_loc(self.length_name)
            idxs = []
            max_length = 0
            batch_size = 0
            while True:
                if self.shuffle:
                    idx = self._permutation[self.start]
                else:
                    idx = self.start

                max_length = max(max_length, self.seg_set.iloc[idx,
                                                               column_idx])
                if max_length * (batch_size + 1) > self.max_batch_length:
                    break

                idxs.append(idx)
                self.start = (self.start + self.world_size) % len(self.seg_set)
                batch_size += 1
                if (self.max_batch_size is not None
                        and batch_size >= self.max_batch_size):
                    break

            assert len(
                idxs
            ) >= 1, f"increase max_batch_length {self.max_batch_length} >= {max_length}"
        else:
            stop = min(self.start + self.world_size * self.min_batch_size,
                       len(self.seg_set))
            if self.shuffle:
                idxs = self._permutation[self.start:stop:self.world_size]
            else:
                idxs = slice(self.start, stop, self.world_size)

            self.start += self.world_size * self.min_batch_size

        if "chunk_start" in self.seg_set:
            chunks = self.seg_set.iloc[idxs]
            seg_ids = [(id, s, d) for id, s, d in zip(
                chunks.seg_id, chunks.chunk_start, chunks[self.length_name])]
        else:
            seg_ids = self.seg_set.iloc[idxs].id.values

        if self.batch == 0:
            logging.info("batch 0 seg_ids=%s", str(seg_ids[:10]))

        self.batch += 1
        return seg_ids

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "min_batch_size",
            "max_batch_size",
            "max_batch_length",
            "length_name",
            "shuffle",
            "drop_last",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--min-batch-size",
            type=int,
            default=1,
            help=("minimum batch size per gpu"),
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=None,
            help=
            ("maximum batch size per gpu, if None, estimated from max_batch_length"
             ),
        )

        parser.add_argument(
            "--max-batch-duration",
            type=float,
            default=None,
            help=
            ("maximum accumlated duration of the batch, if None estimated from the min/max_batch_size and min/max_chunk_lengths"
             ),
        )

        parser.add_argument(
            "--drop-last",
            action=ActionYesNo,
            help="drops the last batch of the epoch",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help=
            "shuffles the segments or chunks at the beginning of the epoch",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        parser.add_argument(
            "--length-name",
            default="duration",
            help=
            "which column in the segment table indicates the duration of the file",
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
