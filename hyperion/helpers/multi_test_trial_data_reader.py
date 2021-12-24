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
from ..utils import TrialNdx, TrialKey, Utt2Info
from ..transforms import TransformList


class MultiTestTrialDataReader(object):
    """
    Loads Ndx, enroll file and x-vectors to evaluate PLDA.
    """

    def __init__(
        self,
        v_file,
        ndx_file,
        enroll_file,
        test_file,
        test_subseg2orig_file,
        preproc,
        tlist_sep=" ",
        model_idx=1,
        num_model_parts=1,
        seg_idx=1,
        num_seg_parts=1,
        eval_set="enroll-test",
    ):

        self.r = DRF.create(v_file)
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

        subseg2orig = Utt2Info.load(test_subseg2orig_file, sep=tlist_sep)

        ndx, enroll = TrialNdx.parse_eval_set(ndx, enroll, test, eval_set)
        if num_model_parts > 1 or num_seg_parts > 1:
            ndx = TrialNdx.split(model_idx, num_model_parts, seg_idx, num_seg_parts)
            enroll = enroll.filter_info(ndx.model_set)
            subseg2orig = subseg2orig.filter_info(ndx.seg_set)

        self.enroll = enroll
        self.ndx = ndx
        self.subseg2orig = subseg2orig

    def read(self):
        x_e = self.r.read(self.enroll.key, squeeze=True)
        x_t = self.r.read(self.subseg2orig.key, squeeze=True)

        if self.preproc is not None:
            x_e = self.preproc.predict(x_e)
            x_t = self.preproc.predict(x_t)

        return x_e, x_t, self.enroll.info, self.ndx, self.subseg2orig.info

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "tlist_sep",
            "model_idx",
            "num_model_parts",
            "seg_idx",
            "num_seg_parts",
            "eval_set",
        )
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
            p1 + "tlist-sep", default=" ", help=("trial lists field separator")
        )
        # parser.add_argument(p1+'v-field', dest=(p2+'v_field'), default='',
        #                     help=('dataset field in the data file'))

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

    add_argparse_args = add_class_args
