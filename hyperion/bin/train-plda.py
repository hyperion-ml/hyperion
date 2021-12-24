#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains PLDA
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList


def train_plda(
    iv_file,
    train_list,
    val_list,
    preproc_file,
    epochs,
    ml_md,
    md_epochs,
    output_path,
    **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vcr_args = VCR.filter_args(**kwargs)
    vcr_train = VCR(iv_file, train_list, preproc, **vcr_args)
    x, class_ids = vcr_train.read()

    x_val = None
    class_ids_val = None
    if val_list is not None:
        vcr_val = VCR(iv_file, val_list, preproc, **vcr_args)
        x_val, class_ids_val = vcr_val.read()

    t1 = time.time()

    plda_args = F.filter_train_args(**kwargs)
    model = F.create_plda(**plda_args)
    elbos = model.fit(
        x,
        class_ids,
        x_val=x_val,
        class_ids_val=class_ids_val,
        epochs=epochs,
        ml_md=ml_md,
        md_epochs=md_epochs,
    )

    logging.info("Elapsed time: %.2f s." % (time.time() - t1))

    model.save(output_path)

    elbo = np.vstack(elbos)
    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    elbo_path = os.path.splitext(output_path)[0] + ".csv"
    np.savetxt(elbo_path, elbo, delimiter=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train PLDA",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--val-list", dest="val_list", default=None)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_plda(**vars(args))
