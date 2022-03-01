"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser
import sys
import os
import argparse
import time
import copy

import numpy as np

from ..io import RandomAccessDataReaderFactory as DRF
from ..utils.scp_list import SCPList
from ..np.transforms import TransformList


class VectorReader(object):
    """Class to load data to train PCA, centering, whitening."""

    def __init__(self, v_file, key_file, preproc=None, vlist_sep=" "):

        self.r = DRF.create(v_file)
        self.scp = SCPList.load(key_file, sep=vlist_sep)
        self.preproc = preproc

    def read(self):
        try:
            x = self.r.read(self.scp.key, squeeze=True)
            if self.preproc is not None:
                x = self.preproc.predict(x)
        except:
            x = self.r.read(self.scp.key, squeeze=False)
            if self.preproc is not None:
                for i in range(len(x)):
                    if x[i].ndim == 1:
                        x[i] = x[i][None, :]
                    x[i] = self.preproc.predict(x[i])

        return x

    @staticmethod
    def filter_args(**kwargs):
        valid_args = "vlist_sep"
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--vlist-sep", default=" ", help=("utterance file field separator")
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='vector reader params')

    add_argparse_args = add_class_args
