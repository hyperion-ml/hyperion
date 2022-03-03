#!/usr/bin/env python
"""                                                                                                     Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)                                     
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import sys
import os
import argparse
import time

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.transforms import TransformList, LDA, LNorm, PCA
from hyperion.helpers import PLDAFactory as F

from numpy.linalg import matrix_rank


def train_be(
    iv_file,
    train_list,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    output_path,
    **kwargs
):

    # Read data
    vcr_args = VCR.filter_args(**kwargs)
    vcr_train = VCR(iv_file, train_list, None, **vcr_args)
    x, class_ids = vcr_train.read()
    del vcr_train

    t1 = time.time()
    rank = matrix_rank(x)
    pca = None
    if rank < x.shape[1]:
        # do PCA if rank of x is smaller than its dimension
        pca = PCA(pca_dim=rank, name="pca")
        pca.fit(x)
        x = pca.predict(x)
        if y_dim > rank:
            y_dim = rank
        logging.info("PCA rank=%d" % (rank))

    # Train centering and whitening
    t1 = time.time()
    lnorm = LNorm(name="lnorm")
    lnorm.fit(x)

    x_ln = lnorm.predict(x)
    logging.info("LNorm Elapsed time: %.2f s." % (time.time() - t1))

    # Train PLDA
    t1 = time.time()

    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, name="plda")
    elbo = plda.fit(x_ln, class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs)

    logging.info("PLDA Elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    if pca is None:
        preproc = TransformList([lnorm])
    else:
        preproc = TransformList([pca, lnorm])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    preproc.save(output_path + "/lnorm.h5")
    plda.save(output_path + "/plda.h5")

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo.csv", elbo, delimiter=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train Back-end",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)

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

    train_be(**vars(args))
