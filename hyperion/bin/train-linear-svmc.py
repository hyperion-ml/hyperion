#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains linear SVM classifier
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.transforms import TransformList
from hyperion.classifiers import LinearSVMC as SVM


def train_svm(iv_file, train_list, preproc_file, output_path, **kwargs):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vcr_args = VCR.filter_args(**kwargs)
    vcr_train = VCR(iv_file, train_list, preproc, **vcr_args)
    x, class_ids = vcr_train.read()

    t1 = time.time()

    model_args = SVM.filter_train_args(**kwargs)
    model = SVM(**model_args)
    model.fit(x, class_ids)
    logging.info("Elapsed time: %.2f s." % (time.time() - t1))

    model.save(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train linear SVM classifier",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VCR.add_argparse_args(parser)
    SVM.add_argparse_train_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_svm(**vars(args))
