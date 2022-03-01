"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
import argparse
import time
import copy

import numpy as np

from ..io import HypDataReader
from ..utils import TrialNdx, SCPList
from ..np.transforms import TransformList


class ClassifTrialDataReader(object):
    """
    Loads data to eval classification problems (deprecated)
    """

    def __init__(
        self,
        v_file,
        class2int_file,
        test_file,
        preproc,
        v_field="",
        seg_idx=1,
        num_seg_parts=1,
    ):

        self.r = HypDataReader(v_file)
        self.preproc = preproc
        self.field = v_field

        with open(class2int_file, "r") as f:
            model_set = [line.rstrip().split()[0] for line in f]

        with open(test_file, "r") as f:
            seg_set = [line.rstrip().split()[0] for line in f]

        ndx = TrialNdx(model_set, seg_set)
        logging.debug(num_seg_parts)
        if num_seg_parts > 1:
            ndx = TrialNdx.split(1, 1, seg_idx, num_seg_parts)

        self.ndx = ndx

    def read(self):
        x_t = self.r.read(self.ndx.seg_set, self.field, return_tensor=True)
        if self.preproc is not None:
            x_t = self.preproc.predict(x_t)

        return x_t, self.ndx

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("v_field", "seg_idx", "num_seg_parts")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "."
            p2 = prefix + "."

        parser.add_argument(
            p1 + "v-field", default="", help=("dataset field in the data file")
        )

        parser.add_argument(
            p1 + "seg-part-idx",
            dest=(p2 + "seg_idx"),
            default=1,
            type=int,
            help=("test part index"),
        )
        parser.add_argument(
            p1 + "num-seg-parts",
            default=1,
            type=int,
            help=(
                "number of parts in which we divide the test list "
                "to run evaluation in parallel"
            ),
        )

    add_argparse_args = add_class_args
