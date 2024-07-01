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
from hyperion.np.transforms import TransformList, CentWhiten, PCA, LNorm

# from numpy.linalg import matrix_rank


def train_be_lda(v_file, train_list, output_path, pca, **kwargs):
    from hyperion.helpers import VectorClassReader as VCR
    from hyperion.np.transforms import LDA, LNorm
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Read data
    vr_args = VCR.filter_args(**kwargs)
    vr_train = VCR(v_file, train_list, None, **vr_args)
    x, ids = vr_train.read()
    del vr_train

    t1 = time.time()
    lnorm = LNorm()
    x = lnorm(x)
    _, ids = np.unique(ids, return_inverse=True)
    pca = LDA(lda_dim=90)
    pca.fit(x, ids)
    logging.info("PCA elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    preproc = TransformList([lnorm, pca])

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/preproc.h5")


def train_be(v_file, train_list, output_path, pca, **kwargs):

    # Read data
    vr_args = VR.filter_args(**kwargs)
    vr_train = VR(v_file, train_list, None, **vr_args)
    x = vr_train.read()
    del vr_train

    t1 = time.time()
    pca = PCA(**pca)
    pca.fit(x)
    logging.info("PCA dimenson=%d", pca.pca_dim)
    logging.info("PCA elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    preproc = TransformList([pca])
    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/preproc.h5")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train Back-end")

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)

    VR.add_argparse_args(parser)
    PCA.add_class_args(parser, prefix="pca")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_be(**namespace_to_dict(args))