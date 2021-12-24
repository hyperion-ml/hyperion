#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorReader as VR
from hyperion.transforms import TransformList, CentWhiten, PCA

from numpy.linalg import matrix_rank


def train_be(iv_file, train_list, output_path, **kwargs):

    # Read data
    vr_args = VR.filter_args(**kwargs)
    vr_train = VR(iv_file, train_list, None, **vr_args)
    x = vr_train.read()
    del vr_train

    t1 = time.time()
    rank = matrix_rank(x)
    pca = None
    if rank < x.shape[1]:
        # do PCA if rank of x is smaller than its dimension
        pca = PCA(pca_dim=rank, name="pca")
        pca.fit(x)
        x = pca.predict(x)
        logging.info("PCA rank=%d" % (rank))

    # Train centering and whitening
    t1 = time.time()
    cw = CentWhiten(name="cw")
    cw.fit(x)

    logging.info("PCA-CW Elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    if pca is None:
        preproc = TransformList([cw])
    else:
        preproc = TransformList([pca, cw])

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/cw.h5")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train Back-end")

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)

    VR.add_argparse_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_be(**namespace_to_dict(args))
