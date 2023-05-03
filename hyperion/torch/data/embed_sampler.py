"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

import torch

from .hyp_sampler import HypSampler


class EmbedSampler(HypSampler):
    def __init__(
        self, embed_set, batch_size=1, shuffle=False, drop_last=False, seed=1234,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.embed_set = embed_set
        self.batch_size = batch_size
        self.avg_batch_size = batch_size

        num_batches = len(self.embed_set) / batch_size / self.world_size
        if drop_last:
            self._len = int(num_batches)
        else:
            self._len = int(math.ceil(num_batches))

        self._permutation = None

    def __len__(self):
        return self._len

    def _shuffle_embeds(self):
        self._permutation = torch.randperm(
            len(self.embed_set), generator=self.rng
        ).numpy()

    def __iter__(self):
        super().__iter__()
        if self.shuffle:
            self._shuffle_segs()

        self.start = self.rank
        return self

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        stop = min(
            self.start + self.world_size * self.min_batch_size, len(self.embed_set)
        )
        if self.shuffle:
            idx = self._permutation[self.start : stop : self.world_size]
        else:
            idx = slice(self.start, stop, self.world_size)

        self.start += self.world_size * self.min_batch_size

        embed_ids = self.embed_set.iloc[idx].id

        if self.batch == 0:
            logging.info("batch 0 chunks=%s", str(embed_ids[:10]))

        self.batch += 1
        return embed_ids

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "batch_size",
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
            "--batch-size", type=int, default=1, help=("minimum batch size per gpu"),
        )

        parser.add_argument(
            "--drop-last", action=ActionYesNo, help="drops the last batch of the epoch",
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="shuffles the segments or chunks at the beginning of the epoch",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
