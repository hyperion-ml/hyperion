"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import logging

import numpy as np

import torch
from .hyp_sampler import HypSampler


def get_loc(seg_set, keys):
    if isinstance(keys, (list, np.ndarray)):
        return seg_set.index.get_indexer(keys)

    loc = seg_set.index.get_loc(keys)
    if isinstance(loc, int):
        return loc
    elif isinstance(loc, np.ndarray) and loc.dtype == np.bool:
        return np.nonzero(loc)[0]
    else:
        return list(range(loc.start, loc.stop, loc.step))

class ClassWeightedRandomSegSampler(HypSampler):
    def __init__(
        self,
        seg_set,
        class_info,
        min_batch_size=1,
        max_batch_size=None,
        max_batch_length=None,
        length_name="duration",
        shuffle=False,
        drop_last=False,
        # weight_exponent=1.0,
        # weight_mode="custom",
        seg_weight_mode="uniform",
        num_segs_per_class=1,
        class_name="class_id",
        seed=1234,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.class_info = class_info
        # self.weight_exponent=weight_exponent
        # self.weight_mode=weight_mode
        self.seg_weight_mode = seg_weight_mode
        self.num_segs_per_class = num_segs_per_class
        self.class_name=class_name
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

        self._gather_class_info()
        self._permutation = None


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
        # logging.info("total_duration", self.class_info["total_duration"])

        # we need the mapping from class index to id
        self.map_class_idx_to_ids = self.class_info[["class_idx", "id"]]
        self.map_class_idx_to_ids.set_index("class_idx", inplace=True)

        # we need the list of segments from each class
        # to speed up segment sampling
        # searching then in each batch, it is too slow
        map_class_to_segs = self.seg_set[["id", self.class_name]].set_index(
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

                seg_idx = get_loc(self.seg_set,seg_ids)
            else:
                seg_idx = []
                self.class_info.loc[class_id, "weights"] = 0.0
                self.class_info.renorm_weights()

            self.map_class_to_segs_idx[class_id] = seg_idx
        logging.info(f'weight_exponent weight:{self.class_info["weights"]}')


    def _get_class_weights(self):
        # if not self.var_weights:
        # return torch.as_tensor(self.class_info["weights"].values)

        class_weights = self.class_info["weights"].values.copy()
        # renormalize weights
        class_weights /= class_weights.sum()
        return torch.as_tensor(class_weights)

    def _sample_classes(self, num_classes):
        weights = self._get_class_weights()
        # logging.info("weights: %s", weights)

        row_idx = torch.multinomial(
            weights, num_samples=num_classes, replacement=True, generator=self.rng,
        ).numpy()

        class_ids = self.class_info.iloc[row_idx].id.values

        return class_ids


    def _sample_segs(self, class_ids):

        dur_col_idx = self.seg_set.columns.get_loc(self.length_name)
        id_col_idx = self.seg_set.columns.get_loc("id")

        seg_ids = []
        for c in class_ids:
            # for each class we sample segments longer than chunk length
            # get segments belonging to c
            # t1 = time.time()
            seg_idx_c = self.map_class_to_segs_idx[c]
            # seg_idx_c = self.map_class_to_segs_idx[c]
            # t2 = time.time()
            durs = self.seg_set.iloc[seg_idx_c, dur_col_idx].values
            # if self.class_info.loc[c, "min_seg_duration"] < chunk_length:
            #     mask = durs >= chunk_length
            #     seg_idx_c = seg_idx_c[mask]
            #     durs = durs[mask]

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


        class_ids = self._sample_classes(batch_size)
        seg_ids = self._sample_segs(class_ids)


        # if "chunk_start" in self.seg_set:
        #     chunks = self.seg_set.iloc[idxs]
        #     seg_ids = [(id, s, d) for id, s, d in zip(
        #         chunks.seg_id, chunks.chunk_start, chunks[self.length_name])]
        # else:
        #     seg_ids = self.seg_set.iloc[idxs].id.values

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
            # "weight_exponent",
            # "weight_mode",
            "seg_weight_mode",
            "num_segs_per_class",
            "class_name",
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
            "--num-segs-per-class",
            type=int,
            default=1,
            help=("number of segments per class in batch"),
        )
        parser.add_argument(
            "--seg-weight-mode",
            default="uniform",
            choices=["uniform", "data-prior"],
            help=("method to sample segments given a class"),
        )
        parser.add_argument(
            "--class-name",
            default="class_id",
            help="which column in the segment table indicates the class of the segment",
        )


        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
