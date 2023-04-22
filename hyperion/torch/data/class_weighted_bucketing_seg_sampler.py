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
import torch.distributed as dist

from .hyp_sampler import HypSampler
from .class_weighted_seg_sampler import ClassWeightedRandomSegSampler


class ClassWeightedRandomBucketingSegSampler(HypSampler):
    def __init__(self,
                 seg_set,
                 class_info,
                 base_sampler=ClassWeightedRandomSegSampler,
                 num_buckets=10,
                 length_column="duration",
                 num_chunks_per_seg_epoch=1.0,
                 weight_exponent=1.0,
                 weight_mode="custom",
                 class_name="language",
                 max_audio_length=None,
                 seed=1234,
                 **base_kwargs):
        super().__init__(shuffle=False, seed=seed)
        self.class_name = class_name
        self.seg_set = seg_set
        self.class_info = class_info
        self.base_sampler = base_sampler
        self.base_kwargs = base_kwargs
        self.base_kwargs["seed"] = seed
        self.num_buckets = num_buckets
        self.length_column = length_column
        self.num_chunks_per_seg_epoch = num_chunks_per_seg_epoch
        self.weight_exponent = weight_exponent
        self.max_audio_length = max_audio_length
        self.weight_mode = weight_mode
        self._gather_class_info()
        self._set_class_weights()
        self._create_bucket_samplers()
        self._compute_len()
        self.depleted_buckets = torch.zeros((num_buckets, ), dtype=torch.bool)

    def create_buckets(self):
        # class_ids = self._sample_classes()
        sort_idx = np.argsort(self.seg_set[self.length_column].values)
        sorted_seg_set = self.seg_set.iloc[sort_idx]
        # import pdb; pdb.set_trace()
        # remove audio length larger than max_audio_length
        if self.max_audio_length is not None:
            sorted_seg_set = sorted_seg_set.loc[sorted_seg_set[self.length_column] <= self.max_audio_length]
        cum_lengths = np.cumsum(sorted_seg_set[self.length_column].values,
                                axis=0)
        bucket_length = cum_lengths[-1] / self.num_buckets
        buckets = []
        for i in range(self.num_buckets):
            # logging.info("self.seg_set", self.seg_set.get_col_idx(self.length_column))
            # logging.info("sorted_seg_set", sorted_seg_set.get_col_idx(self.length_column))
            bucket_idx = (cum_lengths <= bucket_length) & (cum_lengths > 0)
            bucket_i = sorted_seg_set.loc[bucket_idx]
            # logging.info("bucket_i", bucket_i.get_col_idx(self.length_column))
            buckets.append(bucket_i)
            cum_lengths -= bucket_length

        return buckets

    def _create_bucket_samplers(self):
        buckets = self.create_buckets()
        bucket_samplers = []
        for i in range(self.num_buckets):
            sampler_i = self.base_sampler(buckets[i],
                 self.class_info,
                 class_name=self.class_name, 
                 num_chunks_per_seg_epoch=self.num_chunks_per_seg_epoch,
                 **self.base_kwargs)
            bucket_samplers.append(sampler_i)

        self.bucket_samplers = bucket_samplers

    def __len__(self):
        return self._len

    def _gather_class_info(self):
        # we get some extra info that we need for the classes.

        # we need the maximum/minimum segment duration for each class.
        total_dur = np.zeros(len(self.class_info))
        for i, c in enumerate(self.class_info["id"]):
            seg_idx = self.seg_set[self.class_name] == c
            if seg_idx.sum() > 0:
                durs_i = self.seg_set.loc[seg_idx, self.length_column]
                total_dur[i] = durs_i.sum()
            else:
                total_dur[i] = 0

        self.class_info["total_duration"] = total_dur
        # logging.info("total_duration", self.class_info["total_duration"])

        # we need the mapping from class index to id
        self.map_class_idx_to_ids = self.class_info[["class_idx", "id"]]
        self.map_class_idx_to_ids.set_index("class_idx", inplace=True)

    def _set_class_weights(self):
        # logging.info("setting class weights")
        # logging.info(f'weight mode:{self.weight_mode}')
        # logging.info(f'weight exponent:{self.weight_exponent}')
        # import pdb; pdb.set_trace()
        if self.weight_mode == "uniform":
            self.class_info.set_uniform_weights()
        elif self.weight_mode == "data-prior":
            weights = self.class_info["total_duration"].values
            self.class_info.set_weights(weights)
            logging.info(f'data-prior weight:{self.class_info["weights"]}')

        if self.weight_exponent != 1.0:
            self.class_info.exp_weights(self.weight_exponent)
        logging.info(f'weight_exponent weight:{self.class_info["weights"]}')
        

    def _compute_len(self):
        self._len = 0
        for i in range(self.num_buckets):
            self._len += len(self.bucket_samplers[i])

    def set_epoch(self, epoch):
        for i in range(self.num_buckets):
            self.bucket_samplers[i].set_epoch(epoch)

    def __iter__(self):
        super().__iter__()
        self.depleted_buckets[:] = False
        for i in range(self.num_buckets):
            self.bucket_samplers[i].__iter__()

        return self

    def all_buckets_depleted(self):
        return torch.all(self.depleted_buckets).item()

    def __next__(self):

        if self.batch == self._len or self.all_buckets_depleted():
            raise StopIteration

        while True:
            bucket_idx = torch.randint(low=0,
                                       high=self.num_buckets,
                                       size=(1, ),
                                       generator=self.rng).item()
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
        avg_batch_size = 0
        for sampler in self.bucket_samplers:
            avg_batch_size += sampler.avg_batch_size

        avg_batch_size /= self.num_buckets
        return avg_batch_size

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "num_buckets",
            "length_column",
            "num_chunks_per_seg_epoch",
            "weight_exponent",
            "weight_mode",
            "max_audio_length",
            "class_name",
            "length_column",
            "shuffle",
            "seed",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)


    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")


        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            default=1,
            type=lambda x: x if x == "auto" else float(x),
            help=("number of times we sample a segment in each epoch"),
        )

        parser.add_argument(
            "--weight-exponent",
            default=1.0,
            type=float,
            help=("exponent for class weights"),
        )


        parser.add_argument(
            "--max-audio-length",
            default=None,
            type=float,
            help=("the maximum length of an audio segment in seconds"),
        )

        parser.add_argument(
            "--weight-mode",
            default="custom",
            choices=["custom", "uniform", "data-prior"],
            help=("method to get the class weights"),
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

        parser.add_argument(
            "--length-column",
            default="duration",
            help="which column in the segment table indicates the duration of the segment",
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="which column in the segment table indicates the class of the segment",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
