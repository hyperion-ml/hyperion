"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import logging
import time

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
        weight_exponent=1.0,
        weight_mode="custom",
        seg_weight_mode="uniform",
        num_hard_prototypes=0,
        affinity_matrix=None,
        class_name="class_id",
        length_name="duration",
        shuffle=False,
        iters_per_epoch=None,
        batch_size=None,
        seed=1234,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.class_name = class_name
        self.length_name = length_name
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

        self.weight_exponent = weight_exponent
        self.weight_mode = weight_mode
        self.seg_weight_mode = seg_weight_mode

        self.num_hard_prototypes = num_hard_prototypes
        self.batch = 0

        # compute the number of batches / epoch
        # legacy config parameter
        num_chunks_per_seg_epoch = (
            iters_per_epoch if iters_per_epoch is not None else num_chunks_per_seg_epoch
        )
        self._set_num_chunks_per_seg_epoch(num_chunks_per_seg_epoch)
        self._compute_len()

        # # fast mapping from classes to segments
        # self.map_class_to_segs = self.seg_set.df[
        #     ["id", self.class_name, self.length_name]
        # ]
        # self.map_class_to_segs.set_index(self.class_name, drop=False, inplace=True)

        self._gather_class_info()
        self._set_class_weights()

        self.set_hard_prototypes(affinity_matrix)

        logging.info(
            (
                "sampler batches/epoch=%d min-batch-size=%d, max-batch-size=%d "
                "avg-batch-size/gpu=%.2f avg-classes/batch=%.2f  samples/(seg*epoch)=%d"
            ),
            self._len,
            self.min_batch_size,
            self.max_batch_size,
            self.avg_batch_size,
            self.avg_batch_size / num_segs_per_class / num_chunks_per_seg,
            self.num_chunks_per_seg_epoch,
        )

    def _set_seed(self):
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.rank)
        else:
            self.rng.manual_seed(self.seed + 100 * self.rank)

    def _set_num_chunks_per_seg_epoch(self, num_chunks_per_seg_epoch):
        if num_chunks_per_seg_epoch == "auto":
            self._compute_num_chunks_per_seg_epoch_auto()
        else:
            self.num_chunks_per_seg_epoch = num_chunks_per_seg_epoch

    def _compute_num_chunks_per_seg_epoch_auto(self):
        seg_set = self.seg_set
        avg_seg_length = np.mean(seg_set[self.length_name])
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
        total_dur = np.zeros(len(self.class_info))
        for i, c in enumerate(self.class_info["id"]):
            seg_idx = self.seg_set[self.class_name] == c
            if seg_idx.sum() > 0:
                durs_i = self.seg_set.loc[seg_idx, self.length_name]
                max_dur[i] = durs_i.max()
                min_dur[i] = durs_i.min()
                total_dur[i] = durs_i.sum()
            else:
                max_dur[i] = min_dur[i] = total_dur[i] = 0

        self.class_info["max_seg_duration"] = max_dur
        self.class_info["min_seg_duration"] = min_dur
        self.class_info["total_duration"] = total_dur

        # we need the mapping from class index to id
        self.map_class_idx_to_ids = self.class_info[["class_idx", "id"]]
        self.map_class_idx_to_ids.set_index("class_idx", inplace=True)

        # we need the list of segments from each class
        # to speed up segment sampling
        # searching then in each batch, it is too slow
        map_class_to_segs = self.seg_set.df[["id", self.class_name]].set_index(
            self.class_name
        )
        self.map_class_to_segs_idx = {}
        for class_id in self.class_info["id"].values:
            if class_id in map_class_to_segs.index:
                seg_ids = map_class_to_segs.loc[class_id, "id"]
                if isinstance(seg_ids, str):
                    seg_ids = [seg_ids]
                else:
                    seg_ids = seg_ids.values

                seg_idx = self.seg_set.get_loc(seg_ids)
            else:
                seg_idx = []
                self.class_info.loc[class_id, "weights"] = 0.0
                self.class_info.renorm_weights()

            self.map_class_to_segs_idx[class_id] = seg_idx

    def _set_class_weights(self):
        if self.weight_mode == "uniform":
            self.class_info.set_uniform_weights()
        elif self.weight_mode == "data-prior":
            weights = self.class_info["total_duration"].values
            self.class_info.set_weights(self, weights)

        if self.weight_exponent != 1.0:
            self.class_info.exp_weights(self.weight_exponent)

        zero_weight = self.class_info["min_seg_duration"] < self.min_chunk_length
        if np.any(zero_weight):
            self.class_info.set_zero_weight(zero_weight)

        self.var_weights = np.any(
            self.seg_set[self.length_name] < self.max_chunk_length
        )

    @property
    def hard_prototype_mining(self):
        return self.num_hard_prototypes > 1

    def set_hard_prototypes(self, affinity_matrix):
        if affinity_matrix is None:
            self.hard_prototypes = None
            return

        # don't sample hard negs from classes with zero weigth or absent
        zero_w = self.class_info["weights"] == 0
        if np.any(zero_w):
            zero_w_idx = self.class_info.loc[zero_w, "class_idx"].values
            affinity_matrix[:, zero_w_idx] = -1000

        for i in range(affinity_matrix.size(1)):
            mask_i = self.class_info["class_idx"] == i
            if np.all(mask_i == 0):
                affinity_matrix[:, i] = -1000

        # affinity_matrix[np.diag(affinity_matrix.shape[0])] = -1.0
        # hard prototypes for a class are itself and k-1 closest to it.
        self.hard_prototypes = torch.topk(
            affinity_matrix, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(self, class_idx):
        return self.hard_prototypes[class_idx].flatten().numpy()

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
        num_classes = batch_size / self.num_segs_per_class / self.num_chunks_per_seg
        if self.hard_prototype_mining:
            num_classes /= self.num_hard_prototypes
        return int(math.ceil(num_classes))

    def _get_class_weights(self, chunk_length):
        if not self.var_weights:
            return torch.as_tensor(self.class_info["weights"].values)

        # get classes where all segments are shorter than
        # chunk length and put weight to 0
        zero_idx = self.class_info["max_seg_duration"] < chunk_length
        if not np.any(zero_idx):
            return torch.as_tensor(self.class_info["weights"].values)

        class_weights = self.class_info["weights"].values.copy()
        class_weights[zero_idx] = 0.0
        # renormalize weights
        class_weights /= class_weights.sum()
        return torch.as_tensor(class_weights)

    def _sample_classes(self, num_classes, chunk_length):
        weights = self._get_class_weights(chunk_length)
        row_idx = torch.multinomial(
            weights, num_samples=num_classes, replacement=True, generator=self.rng,
        ).numpy()

        class_ids = self.class_info.iloc[row_idx].id.values
        if self.hard_prototype_mining:
            # map class ids to class indexes
            class_idx = self.class_info.loc[class_ids, "class_idx"].values
            class_idx = self.get_hard_prototypes(class_idx)
            # map back to class ids
            class_ids = self.map_class_idx_to_ids.loc[class_idx, "id"].values

        return class_ids

    def _sample_segs(self, class_ids, chunk_length):

        dur_col_idx = self.seg_set.get_col_idx(self.length_name)
        id_col_idx = self.seg_set.get_col_idx("id")

        seg_ids = []
        for c in class_ids:
            # for each class we sample segments longer than chunk length
            # get segments belonging to c
            # t1 = time.time()
            seg_idx_c = self.map_class_to_segs_idx[c]
            # t2 = time.time()
            durs = self.seg_set.iloc[seg_idx_c, dur_col_idx].values
            if self.class_info.loc[c, "min_seg_duration"] < chunk_length:
                mask = durs >= chunk_length
                seg_idx_c = seg_idx_c[mask]
                durs = durs[mask]

            # t3 = time.time()
            # sample num_segs_per_class random segments
            if len(seg_idx_c) == 0:
                logging.error("no segments found with class=%s dur=%d", c, chunk_length)
            if self.seg_weight_mode == "uniform":
                sel_idx = torch.randint(
                    low=0,
                    high=len(seg_idx_c),
                    size=(self.num_segs_per_class,),
                    generator=self.rng,
                ).numpy()

            elif self.seg_weight_mode == "data-prior":
                weights = durs / durs.sum()
                sel_idx = torch.multinomial(
                    torch.from_numpy(weights),
                    num_samples=self.num_segs_per_class,
                    replacement=True,
                    generator=self.rng,
                ).numpy()
                # t4 = time.time()
            else:
                raise ValueError("unknown seg-weight-mode=%s", self.seg_weight_mode)

            sel_seg_idx_c = seg_idx_c[sel_idx]
            sel_seg_ids_c = list(self.seg_set.iloc[sel_seg_idx_c, id_col_idx])
            # t5 = time.time()
            seg_ids.extend(sel_seg_ids_c)
            # t6 = time.time()
            # logging.info(
            #     "stime %f %f %f %f %f", t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5
            # )

        return seg_ids

    def _sample_chunks(self, seg_ids, chunk_length):
        chunks = []
        scale = (
            torch.as_tensor(self.seg_set.loc[seg_ids, self.length_name].values)
            - chunk_length
        )
        for i in range(self.num_chunks_per_seg):
            start = scale * torch.rand(size=(len(seg_ids),), generator=self.rng)
            chunks_i = [(id, s.item(), chunk_length) for id, s in zip(seg_ids, start)]
            chunks.extend(chunks_i)

        return chunks

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        # t1 = time.time()
        chunk_length = self._sample_chunk_length()
        # t2 = time.time()
        batch_size = self._compute_batch_size(chunk_length)
        # t3 = time.time()
        num_classes = self._compute_num_classes_per_batch(batch_size)
        # t4 = time.time()
        class_ids = self._sample_classes(num_classes, chunk_length)
        # t5 = time.time()
        seg_ids = self._sample_segs(class_ids, chunk_length)
        # t6 = time.time()
        chunks = self._sample_chunks(seg_ids, chunk_length)
        # t7 = time.time()
        # print(
        #     "next",
        #     t2 - t1,
        #     t3 - t2,
        #     t4 - t3,
        #     t5 - t4,
        #     t6 - t5,
        #     t7 - t6,
        #     batch_size,
        #     num_classes,
        #     self.min_batch_size,
        #     len(chunks),
        #     flush=True,
        # )
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
            "weight_exponent",
            "weight_mode",
            "seg_weight_mode",
            "num_hard_prototypes",
            "class_name",
            "length_name",
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
            "--weight-exponent",
            default=1.0,
            type=float,
            help=("exponent for class weights"),
        )
        parser.add_argument(
            "--weight-mode",
            default="custom",
            choices=["custom", "uniform", "data-prior"],
            help=("method to get the class weights"),
        )

        parser.add_argument(
            "--seg-weight-mode",
            default="uniform",
            choices=["uniform", "data-prior"],
            help=("method to sample segments given a class"),
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
            "--length-name",
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
