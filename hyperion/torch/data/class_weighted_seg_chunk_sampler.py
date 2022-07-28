"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import logging

import numpy as np
import pandas as pd

import torch
from .hyp_sampler import HypSampler


class ClassWeightedRandomSegChunkSampler(HypSampler):
    def __init__(
        self,
        seg_set,
        class_info,
        min_chunk_length,
        max_chunk_length=None,
        min_batch_size=1,
        max_batch_size=None,
        max_batch_length=None,
        num_chunks_per_seg_epoch="auto",
        num_segs_per_class=1,
        num_chunks_per_seg=1,
        num_hard_prototypes=0,
        affinity_matrix=None,
        class_column="class_id",
        length_column="duration",
        iters_per_epoch=None,
        batch_size=None,
        seed=1234,
    ):
        super().__init__(seed=seed)
        self.class_column = class_column
        self.length_column = length_column
        self.seg_set = seg_set
        self.class_info = class_info
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = (
            min_chunk_length if max_chunk_length is None else max_chunk_length
        )

        # computing min-batch-size
        if batch_size is not None:
            min_batch_size = batch_size

        min_batch_size = max(num_segs_per_class * num_chunks_per_seg, min_batch_size)

        # computing max-batch-size
        if max_batch_length is None:
            max_batch_size_0 = int(min_batch_size * max_chunk_length / min_chunk_length)
        else:
            max_batch_size_0 = int(max_batch_length / max_chunk_length)

        max_batch_size = (
            max_batch_size_0
            if max_batch_size is None
            else min(max_batch_size_0, max_batch_size)
        )

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.avg_batch_size = (min_batch_size + max_batch_size) / 2
        self.var_batch_size = self.min_batch_size != self.max_batch_size

        self.num_segs_per_class = num_segs_per_class
        self.num_chunks_per_seg = num_chunks_per_seg

        self.num_hard_prototypes = num_hard_prototypes
        self.batch = 0

        # compute the number of batches / epoch
        # legacy config parameter
        num_chunks_per_seg_epoch = (
            iters_per_epoch if iters_per_epoch is not None else num_chunks_per_seg_epoch
        )
        self._set_num_chunks_per_seg_epoch(num_chunks_per_seg_epoch)
        self._compute_len()

        self._gather_class_info()

        self.set_hard_prototypes(affinity_matrix)

        logging.info(
            "batches/epoch=%d min-batch-size=%d, max-batch-size=%d avg-batch-size/gpu=%.2f avg-classes/batch=%.2f  samples/(seg*epoch)=%d",
            self._len,
            self.min_batch_size,
            self.max_batch_size,
            self.avg_batch_size,
            self.avg_batch_size / num_segs_per_class / num_chunks_per_seg,
            self.num_chunks_per_seg_epoch,
        )

    def _set_seed(self):
        self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.rank)

    def _set_num_chunks_per_seg_epoch(self, num_chunks_per_seg_epoch):
        if num_chunks_per_seg_epoch == "auto":
            self._compute_num_chunks_per_seg_epoch_auto()
        else:
            self.num_egs_per_utt_epoch = num_chunks_per_seg_epoch

    def _compute_num_chunks_per_seg_epoch_auto(self):
        seg_set = self.seg_set
        avg_seg_length = np.mean(seg_set[self.length_column])
        avg_chunk_length = (self.max_chunk_length + self.min_chunk_length) / 2
        self.num_chunks_per_seg_epoch = math.ceil(avg_seg_length / avg_chunk_length)
        logging.debug(
            "num egs per segment and epoch: %d", self.num_chunks_per_seg_epoch
        )

    def _compute_len(self):
        self._len = int(
            math.ceil(
                self.num_chunks_per_seg_epoch
                * len(self.seg_set)
                / self.avg_batch_size
                / self.world_size
            )
        )

    def __len__(self):
        return self._len

    def _gather_class_info(self):
        # we get some extra info that we need for the classes.

        # we need the maximum/minimum segment duration for each class.
        max_dur = np.zeros(len(self.class_info))
        min_dur = np.zeros(len(self.class_info))
        for i, c in enumerate(self.class_info["id"]):
            seg_idx = self.seg_set[self.class_column] == c
            durs_i = self.seg_set.loc[seg_idx]
            max_dur[i] = durs_i.max()
            min_dur[i] = durs_i.min()

        self.class_info["max_seg_duration"] = max_dur
        self.class_info["min_seg_duration"] = min_dur

        zero_weight = self.class_info["min_seg_duration"] < self.min_chunk_length
        if np.any(zero_weight):
            self.class_info.loc[zero_weight, "weights"] = 0
            self.class_info.loc["weights"] /= self.class_info["weights"].sum()

        self.var_weights = np.any(
            self.seg_set[self.length_column] < self.max_chunk_length
        )

        self.map_idx_to_ids = self.class_info[["class_idx", "ids"]]
        self.map_idx_to_ids.set_index("class_idx", inplace=True)

    @property
    def hard_prototype_mining(self):
        return self.num_hard_prototypes > 1

    def set_hard_prototypes(self, affinity_matrix):
        if affinity_matrix is None:
            self.hard_prototypes = None
            return

        # affinity_matrix[np.diag(affinity_matrix.shape[0])] = -1.0
        # hard prototypes for a class are itself and k-1 closest to it.
        self.hard_prototypes = torch.topk(
            affinity_matrix, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(self, class_idx):
        return self.hard_prototypes[class_idx].flatten()

    def _sample_chunk_length(self):
        if self.var_batch_size:
            return (
                torch.rand(size=(1,), generator=self.rng).item()
                * (self.max_chunk_length - self.min_chunk_length)
                + self.min_chunk_length
            )

        return self.min_chunk_length

    def _compute_batch_size(self, chunk_length):
        return int(self.min_batch_size * self.max_chunk_length / chunk_length)

    def _compute_num_classes_per_batch(self, batch_size):
        num_classes = batch_size / self.num_segs_per_class / self.num_egs_per_utt
        if self.hard_prototype_mining:
            num_classes /= self.num_hard_prototypes
        return int(math.ceil(num_classes))

    def _get_class_weights(self, chunk_length):
        if not self.var_weights:
            return self.class_info["weights"].values

        # get classes where all segments are shorter than
        # chunk length and put weight to 0
        zero_idx = self.class_info["max_seg_duration"] < chunk_length
        if not np.any(zero_idx):
            return self.class_info["weights"].values

        class_weights = self.class_info["weights"].values.copy()
        class_weights[zero_idx] = 0.0
        # renormalize weights
        class_weights /= class_weights.sum()
        return class_weights

    def _sample_classes(self, num_classes, chunk_length):
        weights = self._get_class_weights(chunk_length)
        row_idx = torch.multinomial(
            weights,
            num_samples=num_classes,
            replacement=True,
        )

        class_ids = self.class_info.iloc[row_idx].id.values
        if self.hard_prototype_mining:
            # map class ids to class indexes
            class_idx = self.class_info.loc[class_ids, "class_idx"]
            class_idx = self.get_hard_prototypes(class_idx)
            # map back to class ids
            class_ids = self.map_idx_to_ids.loc[class_idx]

        return class_ids

    def _sample_segs(self, class_ids, chunk_length):

        seg_ids = []
        for c in class_ids:
            # for each class we sample segments longer than chunk length
            # get segments belonging to c
            seg_mask = (self.seg_set[self.class_column] == c) & (
                self.seg_set[self.length_column] > chunk_length
            )
            seg_ids_c = self.seg_set.loc[seg_mask, "id"]
            # sample num_segs_per_class random segments
            sel_seg_idx_c = torch.randint(
                low=0,
                high=len(seg_ids_c),
                size=(self.num_segs_per_class,),
                generator=self.rng,
            )
            sel_seg_ids_c = seg_ids_c[sel_seg_idx_c]
            seg_ids.extend(sel_seg_ids_c)

        return seg_ids

    def _sample_chunks(self, seg_ids, chunk_length):
        chunks = []
        scale = self.seg_set.loc[seg_ids, self.length_column] - chunk_length
        for i in range(self.num_chunks_per_seg):
            start = scale * torch.rand(size=(len(seg_ids),), generator=self.rng)
            chunks_i = [(id, s, chunk_length) for id, s in zip(seg_ids, start)]
            chunks.expand(chunks_i)

        return chunks

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        chunk_length = self._sample_chunk_length()
        batch_size = self._compute_batch_size()
        num_classes = self._compute_num_classes_per_batch(batch_size)
        class_ids = self._sample_classes(num_classes, chunk_length)
        seg_ids = self._sample_segs(class_ids, chunk_length)
        chunks = self._sample_chunks(seg_ids, chunk_length)
        if self.batch == 0:
            logging.info("batch 0 uttidx=%s", str(chunks[:10]))

        self.batch += 1
        return chunks

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "min_chunk_length",
            "max_chunk_length",
            "min_batch_size",
            "max_batch_size",
            "max_batch_length",
            "num_chunks_per_seg_epoch",
            "num_segs_per_class",
            "num_chunks_per_seg",
            "num_hard_prototypes",
            "class_column",
            "length_column",
            "iters_per_epoch",
            "batch_size",
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
            "--min-chunk-length",
            type=float,
            default=4.0,
            help=("minimum length of the segment chunks"),
        )
        parser.add_argument(
            "--max-chunk-length",
            type=float,
            default=None,
            help=("maximum length of segment chunks"),
        )

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
            help=(
                "maximum batch size per gpu, if None, estimated from max_batch_length"
            ),
        )

        parser.add_argument(
            "--batch-size",
            default=128,
            type=int,
            help=("deprecated, use min-batch-size instead"),
        )

        parser.add_argument(
            "--max-batch-duration",
            type=float,
            default=None,
            help=(
                "maximum accumlated duration of the batch, if None estimated from the min/max_batch_size and min/max_chunk_lengths"
            ),
        )

        parser.add_argument(
            "--iters-per-epoch",
            default=None,
            type=lambda x: x if (x == "auto" or x is None) else float(x),
            help=("deprecated, use --num-egs-per-seg-epoch instead"),
        )

        parser.add_argument(
            "--num-chunks-per-seg-epoch",
            default="auto",
            type=lambda x: x if x == "auto" else float(x),
            help=("number of times we sample a segment in each epoch"),
        )

        parser.add_argument(
            "--num-segs-per-class",
            type=int,
            default=1,
            help=("number of segments per class in batch"),
        )
        parser.add_argument(
            "--num-chunks-per-seg",
            type=int,
            default=1,
            help=("number of chunks per segment in batch"),
        )
        parser.add_argument(
            "--num-hard-prototypes",
            type=int,
            default=0,
            help=("number of hard prototype classes per batch"),
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
            "--class-column",
            default="class_id",
            help="which column in the segment table indicates the class of the segment",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
