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
from ..utils import Utt2Info, TrialNdx, TrialKey
from ..transforms import TransformList


class MultiTestTrialDataReaderV2(object):
    """
    Loads Ndx, enroll file and x-vectors to evaluate PLDA.
    """

    def __init__(
        self,
        enroll_v_file,
        test_v_file,
        ndx_file,
        enroll_file,
        test_file,
        preproc=None,
        tlist_sep=" ",
        model_idx=1,
        num_model_parts=1,
        seg_idx=1,
        num_seg_parts=1,
        eval_set="enroll-test",
    ):

        self.r_e = DRF.create(enroll_v_file)
        self.r_t = DRF.create(test_v_file)
        self.preproc = preproc

        enroll = Utt2Info.load(enroll_file, sep=tlist_sep)
        test = None
        if test_file is not None:
            test = Utt2Info.load(test_file, sep=tlist_sep)
        ndx = None
        if ndx_file is not None:
            try:
                ndx = TrialNdx.load(ndx_file)
            except:
                ndx = TrialKey.load(ndx_file).to_ndx()

        ndx, enroll = TrialNdx.parse_eval_set(ndx, enroll, test, eval_set)
        if num_model_parts > 1 or num_seg_parts > 1:
            ndx = TrialNdx.split(model_idx, num_model_parts, seg_idx, num_seg_parts)
            enroll = enroll.filter_info(ndx.model_set)

        self.enroll = enroll
        self.ndx = ndx

    def read(self):
        x_e = self.r_e.read(self.enroll.key, squeeze=True)
        x_t = self.r_t.read(self.ndx.seg_set, squeeze=False)

        orig_seg = []
        for i, x_ti in enumerate(x_t):
            orig_seg.append(np.asarray([i] * x_ti.shape[0], dtype=np.int))

        x_t = np.concatenate(tuple(x_t), axis=0)
        orig_seg = np.concatenate(tuple(orig_seg), axis=0)

        if self.preproc is not None:
            x_e = self.preproc.predict(x_e)
            x_t = self.preproc.predict(x_t)

        return x_e, x_t, self.enroll.info, self.ndx, orig_seg

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ""
        else:
            p = prefix + "_"
        valid_args = (
            "tlist_sep",
            "model_idx",
            "num_model_parts",
            "seg_idx",
            "num_seg_parts",
            "eval_set",
        )
        return dict((k, kwargs[p + k]) for k in valid_args if p + k in kwargs)

    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "-"
            p2 = prefix + "_"
        parser.add_argument(
            p1 + "tlist-sep",
            dest=(p2 + "tlist_sep"),
            default=" ",
            help=("trial lists field separator"),
        )

        parser.add_argument(
            p1 + "model-part-idx",
            dest=(p2 + "model_idx"),
            default=1,
            type=int,
            help=("model part index"),
        )
        parser.add_argument(
            p1 + "num-model-parts",
            default=1,
            type=int,
            help=(
                "number of parts in which we divide the model"
                "list to run evaluation in parallel"
            ),
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

        parser.add_argument(
            p1 + "eval-set",
            type=str.lower,
            default="enroll-test",
            choices=["enroll-test", "enroll-coh", "coh-test", "coh-coh"],
            help=("evaluation subset"),
        )
