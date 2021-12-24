"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
# import os
import math
import logging

import numpy as np

import torch
from torch.utils.data import Sampler


class ClassWeightedEmbedSampler(Sampler):
    def __init__(self, dataset, batch_size=1, iters_per_epoch=1, num_egs_per_class=1):

        super().__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_egs_per_class = num_egs_per_class
        self.batch = 0

        self.iters_per_epoch = iters_per_epoch
        self._len = int(math.ceil(self.iters_per_epoch * len(dataset) / batch_size))

        logging.info("num batches per epoch: %d" % self._len)

        self._num_classes_per_batch = int(math.ceil(batch_size / num_egs_per_class))
        logging.info("num classes per batch: %d" % self._num_classes_per_batch)

    def __len__(self):
        return self._len

    def __iter__(self):
        self.batch = 0
        return self

    def _remove_duplicate_idx(self, utt_idx):
        utt_idx_uniq = torch.unique(utt_idx)
        c = 0
        # we make 3 tries to remove duplicate utt idx
        delta = len(utt_idx) - len(utt_idx_uniq)
        while delta > 0 and c < 3:
            extra_idx = torch.randint(low=0, high=len(self.dataset), size=(delta,))
            utt_idx = torch.cat((utt_idx_uniq, extra_idx))
            utt_idx_uniq = torch.unique(utt_idx)
            delta = len(utt_idx) - len(utt_idx_uniq)
            c += 1

        return utt_idx

    def _get_utt_idx(self):
        dataset = self.dataset
        num_classes_per_batch = self._num_classes_per_batch
        if dataset.class_weights is None:
            class_idx = torch.randint(
                low=0, high=dataset.num_classes, size=(num_classes_per_batch,)
            )
        else:
            class_idx = torch.multinomial(
                dataset.class_weights,
                num_samples=num_classes_per_batch,
                replacement=True,
            )

        if self.num_egs_per_class > 1:
            class_idx = class_idx.repeat(self.num_egs_per_class)

        utt_idx = torch.as_tensor(
            [
                dataset.class2utt_idx[c][
                    torch.randint(low=0, high=int(dataset.class2num_utt[c]), size=(1,))
                ]
                for c in class_idx.tolist()
            ]
        )

        utt_idx = self._remove_duplicate_idx(utt_idx)
        return utt_idx

    def __next__(self):

        if self.batch == self._len:
            raise StopIteration

        utt_idx = self._get_utt_idx()
        if self.batch == 0:
            logging.info("batch 0 uttidx=%s", str(utt_idx[:10]))

        self.batch += 1
        return utt_idx.tolist()
