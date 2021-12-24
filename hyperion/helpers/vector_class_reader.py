"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import argparse
import time
import copy

import numpy as np

from ..io import RandomAccessDataReaderFactory as DRF
from ..utils.utt2info import Utt2Info
from ..utils.tensors import to3D_by_class
from ..transforms import TransformList


class VectorClassReader(object):
    """Class to load data to train LDA, PLDA, PDDA."""

    def __init__(
        self,
        v_file,
        key_file,
        preproc=None,
        vlist_sep=" ",
        class2int_file=None,
        min_spc=1,
        max_spc=None,
        spc_pruning_mode="random",
        csplit_min_spc=1,
        csplit_max_spc=None,
        csplit_mode="random",
        csplit_overlap=0,
        vcr_seed=1024,
        csplit_once=True,
    ):

        self.r = DRF.create(v_file)
        self.u2c = Utt2Info.load(key_file, sep=vlist_sep)
        self.preproc = preproc

        self.map_class2int = None
        if class2int_file is not None:
            with open(class2int_file, "r") as f:
                self.map_class2int = {
                    v[0]: int(v[1]) for v in [line.rstrip().split() for line in f]
                }

        self.rng = np.random.RandomState(vcr_seed)
        self.csplit_max_spc = csplit_max_spc
        self.csplit_min_spc = csplit_min_spc
        self.csplit_mode = csplit_mode
        self.csplit_overlap = csplit_overlap
        self.csplit_once = csplit_once
        self._samples_per_class = None
        self.u2c = self._filter_by_spc(
            self.u2c, min_spc, max_spc, spc_pruning_mode, self.rng
        )
        if csplit_once:
            self.u2c = self._split_classes(
                self.u2c,
                self.csplit_min_spc,
                self.csplit_max_spc,
                self.csplit_mode,
                self.csplit_overlap,
                self.rng,
            )

    def read(self, return_3d=False, max_length=0):
        if self.csplit_once:
            u2c = self.u2c
        else:
            u2c = self._split_classes(
                self.u2c,
                self.csplit_min_spc,
                self.csplit_max_spc,
                self.csplit_mode,
                self.csplit_overlap,
                self.rng,
            )

        x = self.r.read(u2c.key, squeeze=True)
        if self.preproc is not None:
            x = self.preproc.predict(x)

        if self.map_class2int is None:
            _, class_ids = np.unique(u2c.info, return_inverse=True)
        else:
            class_ids = np.array([self.map_class2int[k] for k in u2c.info], dtype=int)
        if return_3d:
            x, sample_weight = to3D_by_class(x, class_ids, max_length)
            return x, sample_weight
        return x, class_ids

    @property
    def class_names(self):
        if self.map_class2int is None:
            return np.unique(self.u2c.info)
        else:
            map_int2class = {k: v for v, k in self.map_class2int.items()}
            classes = [map_int2class[i] for i in range(len(map_int2class))]
            return np.asarray(classes)

    @property
    def samples_per_class(self):
        if self._samples_per_class is None:
            if self.csplit_once:
                u2c = self.u2c
            else:
                u2c = self._split_classes(
                    self.u2c,
                    self.csplit_min_spc,
                    self.csplit_max_spc,
                    self.csplit_mode,
                    self.csplit_overlap,
                    self.rng,
                )
            _, self._samples_per_class = np.unique(u2c.info, return_counts=True)

        return self._samples_per_class

    @property
    def max_samples_per_class(self):
        num_spc = self.samples_per_class
        return np.max(num_spc)

    @staticmethod
    def _filter_by_spc(u2c, min_spc=1, max_spc=None, spc_pruning_mode="last", rng=None):
        if min_spc <= 1 and max_spc == None:
            return u2c

        if min_spc > 1:
            classes, num_spc = np.unique(u2c.info, return_counts=True)
            filter_key = classes[num_spc >= min_spc]
            u2c = u2c.filter_info(filter_key)

        if max_spc is not None:
            classes, class_ids, num_spc = np.unique(
                u2c.info, return_inverse=True, return_counts=True
            )

            if np.all(num_spc <= max_spc):
                return u2c
            f = np.ones_like(class_ids, dtype=bool)
            for i in range(np.max(class_ids) + 1):
                if num_spc[i] > max_spc:
                    indx = np.where(class_ids == i)[0]
                    num_reject = len(indx) - max_spc
                    if spc_pruning_mode == "random":
                        # indx = rng.permutation(indx)
                        # indx = indx[-num_reject:]
                        indx = rng.choice(indx, size=num_reject, replace=False)
                    if spc_pruning_mode == "last":
                        indx = indx[-num_reject:]
                    if spc_pruning_mode == "first":
                        indx = indx[:num_reject]
                    f[indx] = False

            if np.any(f == False):
                u2c = Utt2Info.create(u2c.key[f], u2c.info[f])

        return u2c

    @staticmethod
    def _split_classes(u2c, min_spc, max_spc, mode="sequential", overlap=0, rng=None):
        if max_spc is None:
            return u2c
        if mode == "random_1part":
            return VectorClassReader._filter_by_spc(
                u2c, min_spc, max_spc, "random", rng
            )

        _, class_ids, num_spc = np.unique(
            u2c.info, return_inverse=True, return_counts=True
        )
        if np.all(num_spc <= max_spc):
            return VectorClassReader._filter_by_spc(u2c, min_spc)

        num_classes = np.max(class_ids) + 1

        shift = max_spc - overlap
        new_indx = np.zeros(
            max_spc * int(np.max(num_spc) * num_classes / shift + 1), dtype=int
        )
        new_class_ids = np.zeros_like(new_indx)

        j = 0
        new_i = 0
        for i in range(num_classes):
            indx_i = np.where(class_ids == i)[0]
            if num_spc[i] > max_spc:
                num_subclass = int(np.ceil((num_spc[i] - max_spc) / shift + 1))
                if mode == "sequential":
                    l = 0
                    for k in range(num_subclass - 1):
                        new_indx[j : j + max_spc] = indx_i[l : l + max_spc]
                        new_class_ids[j : j + max_spc] = new_i
                        l += shift
                        j += max_spc
                        new_i += 1
                    n = num_spc[i] - (num_subclass - 1) * shift
                    new_indx[j : j + n] = indx_i[l : l + n]
                    new_class_ids[j : j + n] = new_i
                    j += n
                    new_i += 1
                if mode == "random":
                    for k in range(num_subclass):
                        # indx[j:j+max_spc] = rng.permutation(indx_i)[:max_spc]
                        new_indx[j : j + max_spc] = rng.choice(
                            indx_i, size=max_spc, replace=False
                        )
                        new_class_ids[j : j + max_spc] = new_i
                        j += max_spc
                        new_i += 1
            else:
                new_indx[j : j + num_spc[i]] = indx_i
                new_class_ids[j : j + num_spc[i]] = new_i
                new_i += 1
                j += num_spc[i]

        new_indx = new_indx[:j]
        new_class_ids = new_class_ids[:j].astype("U")
        key = u2c.key[new_indx]
        u2c = Utt2Info.create(key, new_class_ids)

        return VectorClassReader._filter_by_spc(u2c, min_spc)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "vlist_sep",
            "class2int_file",
            "min_spc",
            "max_spc",
            "spc_pruning_mode",
            "csplit_min_spc",
            "csplit_max_spc",
            "csplit_mode",
            "csplit_overlap",
            "csplit_once",
            "vcr_seed",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"

        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "vlist-sep", default=" ", help=("utt2class file field separator")
        )

        parser.add_argument(
            p1 + "class2int-file",
            default=None,
            help=("file that maps class string to integer"),
        )

        parser.add_argument(
            p1 + "min-spc", type=int, default=1, help=("minimum samples per class")
        )
        parser.add_argument(
            p1 + "max-spc", type=int, default=None, help=("maximum samples per class")
        )
        parser.add_argument(
            p1 + "spc-pruning-mode",
            default="random",
            choices=["random", "first", "last"],
            help=("vector pruning method when spc > max-spc"),
        )
        parser.add_argument(
            p1 + "csplit-min-spc",
            type=int,
            default=None,
            help=("minimum samples per class when doing class spliting"),
        )
        parser.add_argument(
            p1 + "csplit-max-spc",
            type=int,
            default=None,
            help=("split one class into subclasses with " "spc <= csplit-max-spc"),
        )

        parser.add_argument(
            p1 + "csplit-mode",
            default="random",
            type=str.lower,
            choices=["sequential", "random", "random_1subclass"],
            help=("class splitting mode"),
        )
        parser.add_argument(
            p1 + "csplit-overlap",
            type=float,
            default=0,
            help=("overlap between subclasses"),
        )
        parser.add_argument(
            p1 + "no-csplit-once",
            default=True,
            action="store_false",
            help=("class spliting done in each iteration"),
        )
        parser.add_argument(
            p1 + "vcr-seed", type=int, default=1024, help=("seed for rng")
        )

    add_argparse_args = add_class_args
