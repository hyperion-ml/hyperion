"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

# import sys
# import os
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from jsonargparse import ActionParser, ArgumentParser, ActionYesNo
from torch.utils.data import Dataset

from ...io import RandomAccessDataReaderFactory as RF
from ...utils.misc import filter_func_args
from ...utils.class_info import ClassInfo
from ...utils.info_table import InfoTable
from ..torch_defs import floatstr_torch


class EmbedDataset(Dataset):
    def __init__(
        self,
        embeds=None,
        embed_info=None,
        class_info=None,
        embed_file=None,
        embed_info_file=None,
        class_names=None,
        class_files=None,
        return_segment_info=None,
        path_prefix=None,
        preload_embeds=False,
        is_val=False,
    ):

        assert embeds is not None or embed_file is not None
        assert embed_info is not None or embed_info is not None
        assert class_info is not None or class_files is not None
        super().__init__()
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.preload_embeds = preload_embeds

        if embed_info is None:
            embed_info = InfoTable.load(embed_info_file)

        self.embed_info = embed_info
        if rank == 0:
            logging.info("dataset contains %d embeddings", len(self.embed_info))

        if embeds is None:
            if rank == 0:
                logging.info("opening dataset %s", rspecifier)
            self.r = RF.create(embed_file, path_prefix=path_prefix, scp_sep=" ")
            if self.preload_embeds:
                self.embeds = self.r.load(embed_info["id"], squeeze=True).astype(
                    floatstr_torch(), copy=False
                )
                del self.r
                self.r = None
        else:
            self.preload_embeds = True
            self.embeds = embeds.astype(floatstr_torch(), copy=False)

        self.is_val = is_val
        if rank == 0:
            logging.info("loading class-info files")
        self._load_class_infos(class_names, class_files, is_val)

        self.return_segment_info = (
            [] if return_segment_info is None else return_segment_info
        )

    def _load_class_infos(self, class_names, class_files, is_val):
        self.class_info = {}
        if class_names is None:
            assert class_files is None
            return

        assert len(class_names) == len(class_files)
        for name, file in zip(class_names, class_files):
            assert (
                name in self.seg_set
            ), f"class_name {name} not present in the segment set"
            if self.rank == 0:
                logging.info("loading class-info file %s" % file)
            table = ClassInfo.load(file)
            self.class_info[name] = table
            if not is_val:
                # check that all classes are present in the training segments
                class_ids = table["id"]
                segment_class_ids = self.seg_set[name].unique()
                for c_id in class_ids:
                    if c_id not in segment_class_ids:
                        logging.warning(
                            "%s class: %s not present in dataset", name, c_id
                        )

    @property
    def num_embeds(self):
        return len(self.embed_info)

    def __len__(self):
        return self.num_embeds

    @property
    def num_classes(self):
        return {k: t.num_classes for k, t in self.class_info.items()}

    def _read_embeds(self, embed_id):
        if self.preload_embeds:
            index = self.embed_info.index.get_loc(embed_id)
            x = self.embeds[index]
        else:
            x = self.r.read([embed_id])[0].astype(floatstr_torch(), copy=False)
        return x

    def _get_embed_info(self, embed_id):
        embed_info = {}
        # converts the class_ids to integers
        for info_name in self.return_embed_info:
            embed_info_i = self.embed_info.loc[embed_id, info_name]
            if info_name in self.class_info:
                # if the type of information is a class-id
                # we use the class information table to
                # convert from id to integer
                class_info = self.class_info[info_name]
                embed_info_i = class_info.loc[embed_info_i, "class_idx"]

            embed_info[info_name] = embed_info_i

        return embed_info

    def __getitem__(self, embed_id):

        x = self._read_embed(embed_id)

        data = {"embed_id": embed_id, "x": x}
        # adds the embed labels
        embed_info = self._get_embed_info(embed_id)
        data.update(embed_info)
        return data
