"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time

import numpy as np
import pandas as pd
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .hyp_sampler import HypSampler


class ClassWeightedEmbedSampler(HypSampler):
    def __init__(
        self,
        embed_set,
        class_info,
        batch_size=1,
        num_embeds_per_class=1,
        weight_exponent=1.0,
        weight_mode="custom",
        num_hard_prototypes=0,
        affinity_matrix=None,
        class_name="class_id",
        shuffle=False,
        seed=1234,
    ):
        super().__init__(shuffle=shuffle, seed=seed)
        self.class_name = class_name
        self.embed_set = embed_set
        self.class_info = class_info
        self.batch_size = batch_size
        self.avg_batch_size = batch_size

        self.num_embeds_per_class = num_embeds_per_class

        self.weight_exponent = weight_exponent
        self.weight_mode = weight_mode

        self.num_hard_prototypes = num_hard_prototypes
        self.batch = 0

        self._compute_len()
        self._compute_num_classes_per_batch()
        self._gather_class_info()
        self._set_class_weights()

        self.set_hard_prototypes(affinity_matrix)

        logging.info(
            ("sampler batches/epoch=%d batch-size=%d, " "classes/batch=%.2f "),
            self._len,
            self.batch_size,
            self.num_classes_per_batch,
        )

    def _set_seed(self):
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.rank)
        else:
            self.rng.manual_seed(self.seed + 100 * self.rank)

    def _compute_len(self):
        self._len = int(
            math.ceil(len(self.embed_set) / self.avg_batch_size / self.world_size)
        )

    def __len__(self):
        return self._len

    def _gather_class_info(self):
        # we get some extra info that we need for the classes.
        # we need the mapping from class index to id
        self.map_class_idx_to_ids = self.class_info[["class_idx", "id"]]
        self.map_class_idx_to_ids.set_index("class_idx", inplace=True)

        # we need the list of embeddings from each class
        # to speed up embedding sampling
        # searching then in each batch, it is too slow
        map_class_to_embeds = self.embed_set.df[["id", self.class_name]].set_index(
            self.class_name
        )
        self.map_class_to_embed_idx = {}
        for class_id in self.class_info["id"].values:
            if class_id in map_class_to_embeds.index:
                embed_ids = map_class_to_embeds.loc[class_id, "id"]
                if isinstance(embed_ids, str):
                    embed_ids = [embed_ids]
                else:
                    embed_ids = embed_ids.values

                embed_idx = self.embed_set.get_loc(embed_ids)
            else:
                embed_idx = []
                self.class_info.loc[class_id, "weights"] = 0.0
                self.class_info.renorm_weights()

            self.map_class_to_embed_idx[class_id] = embed_idx

    def _set_class_weights(self):
        if self.weight_mode == "uniform":
            self.class_info.set_uniform_weights()
        elif self.weight_mode == "data-prior":
            weights = self.class_info["total_duration"].values
            self.class_info.set_weights(self, weights)

        if self.weight_exponent != 1.0:
            self.class_info.exp_weights(self.weight_exponent)

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

        # hard prototypes for a class are itself and k-1 closest to it.
        self.hard_prototypes = torch.topk(
            affinity_matrix, self.num_hard_prototypes, dim=-1
        ).indices

    def get_hard_prototypes(self, class_idx):
        return self.hard_prototypes[class_idx].flatten().numpy()

    def _compute_num_classes_per_batch(self):
        num_classes = self.batch_size / self.num_embeds_per_class
        if self.hard_prototype_mining:
            num_classes /= self.num_hard_prototypes
        self.num_classes_per_batch = int(math.ceil(num_classes))

    def _get_class_weights(self,):
        return torch.as_tensor(self.class_info["weights"].values)

    def _sample_classes(self):
        weights = self._get_class_weights()
        row_idx = torch.multinomial(
            weights,
            num_samples=self.num_classes_per_batch,
            replacement=True,
            generator=self.rng,
        ).numpy()

        class_ids = self.class_info.iloc[row_idx].id.values
        if self.hard_prototype_mining:
            # map class ids to class indexes
            class_idx = self.class_info.loc[class_ids, "class_idx"].values
            class_idx = self.get_hard_prototypes(class_idx)
            # map back to class ids
            class_ids = self.map_class_idx_to_ids.loc[class_idx, "id"].values

        return class_ids

    def _sample_embeds(self, class_ids):

        id_col_idx = self.embed_set.get_col_idx("id")
        embed_ids = []
        for c in class_ids:
            # get embeds belonging to c
            embed_idx_c = self.map_class_to_embed_idx[c]
            # sample num_embeds_per_class randomly
            if len(embed_idx_c) == 0:
                logging.error("no embeddings found with class=%s", c)

            sel_idx = torch.randint(
                low=0,
                high=len(embed_idx_c),
                size=(self.num_embeds_per_class,),
                generator=self.rng,
            ).numpy()

            sel_embed_idx_c = embed_idx_c[sel_idx]
            sel_embed_ids_c = list(self.embed_set.iloc[sel_embed_idx_c, id_col_idx])
            embed_ids.extend(sel_embed_ids_c)

        return embed_ids

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        class_ids = self._sample_classes()
        embed_ids = self._sample_embeds(class_ids)
        if self.batch == 0:
            logging.info("batch 0 uttidx=%s", str(embed_ids[:10]))

        self.batch += 1
        return embed_ids

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "batch_size",
            "num_embeds_per_class",
            "weight_exponent",
            "weight_mode",
            "num_hard_prototypes",
            "class_name",
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
            "--batch-size", type=int, default=1, help=("batch size per gpu"),
        )

        parser.add_argument(
            "--num-embeds-per-class",
            type=int,
            default=1,
            help=("number of embeds per class in batch"),
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
            "--num-hard-prototypes",
            type=int,
            default=0,
            help=("number of hard prototype classes per batch"),
        )

        parser.add_argument(
            "--shuffle",
            action=ActionYesNo,
            help="shuffles the embeddings at the beginning of the epoch",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1234,
            help=("seed for sampler random number generator"),
        )

        parser.add_argument(
            "--class-name",
            default="class_id",
            help="which column in the info table indicates the class",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
