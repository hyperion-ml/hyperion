"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

# import sys
# import os
import logging
import time

# import copy

import numpy as np
import pandas as pd

import torch

from ..torch_defs import floatstr_torch
from ...io import RandomAccessDataReaderFactory as RF
from ...utils.utt2info import Utt2Info

from torch.utils.data import Dataset


class EmbedDataset(Dataset):
    def __init__(
        self,
        embeds=None,
        class_ids=None,
        class_weights=None,
        rspecifier=None,
        key_file=None,
        class_file=None,
        path_prefix=None,
        preload_embeds=False,
        return_class=True,
        is_val=False,
    ):

        assert embeds is not None or rspecifier is not None
        assert rspecifier is None or key_file is not None
        assert class_ids is not None or key_file is not None

        self.preload_embeds = preload_embeds
        if key_file is not None:
            if isinstance(key_file, Utt2Info):
                self.u2c = key_file
            else:
                logging.info("loading utt2info file %s", key_file)
                self.u2c = Utt2Info.load(key_file, sep=" ")
            self.num_embeds = len(self.u2c)
        else:
            assert embeds is not None
            self.u2c = None
            self.num_embeds = len(embeds)

        if embeds is None:
            logging.info("opening dataset %s", rspecifier)
            self.r = RF.create(rspecifier, path_prefix=path_prefix, scp_sep=" ")
            if self.preload_embeds:
                self.embeds = self.r.load(u2c.key, squeeze=True).astype(
                    floatstr_torch(), copy=False
                )
                del self.r
                self.r = None
        else:
            self.preload_embeds = True
            self.embeds = embeds.astype(floatstr_torch(), copy=False)

        self.is_val = is_val
        self._prepare_class_info(class_file, class_ids, class_weights)
        self.return_class = return_class

        logging.info("dataset contains %d embeds", self.num_embeds)

    def __len__(self):
        return self.num_embeds

    def _prepare_class_info(self, class_file, class_idx=None, class_weights=None):
        if class_file is None:
            if self.u2c is not None:
                classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
            self.num_classes = np.max(class_idx) + 1
        else:
            logging.info("reading class-file %s", class_file)
            class_info = pd.read_csv(class_file, header=None, sep=" ")
            class2idx = {str(k): i for i, k in enumerate(class_info[0])}
            self.num_classes = len(class2idx)
            class_idx = np.array([class2idx[k] for k in self.u2c.info], dtype=int)
            if class_info.shape[1] == 2:
                class_weights = np.array(class_info[1]).astype(
                    floatstr_torch(), copy=False
                )

        class2utt_idx = {}
        class2num_utt = np.zeros((self.num_classes,), dtype=int)

        for k in range(self.num_classes):
            idx = (class_idx == k).nonzero()[0]
            class2utt_idx[k] = idx
            class2num_utt[k] = len(idx)
            if class2num_utt[k] == 0:
                if not self.is_val:
                    logging.warning("class %d doesn't have any samples", k)
                if class_weights is None:
                    class_weights = np.ones((self.num_classes,), dtype=floatstr_torch())
                class_weights[k] = 0

        count_empty = np.sum(class2num_utt == 0)
        if count_empty > 0:
            logging.warning("%d classes have 0 samples", count_empty)

        self.utt_idx2class = class_idx
        self.class2utt_idx = class2utt_idx
        self.class2num_utt = class2num_utt
        if class_weights is not None:
            class_weights /= np.sum(class_weights)
            class_weights = torch.Tensor(class_weights)
        self.class_weights = class_weights

    def __getitem__(self, index):
        if self.preload_embeds:
            x = self.embeds[index]
        else:
            key = self.u2c.key[index]
            x = self.r.read([key])[0].astype(floatstr_torch(), copy=False)

        if not self.return_class:
            return x

        class_idx = self.utt_idx2class[index]
        return x, class_idx
