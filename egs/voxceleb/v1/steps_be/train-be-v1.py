#!/usr/bin/env python
"""                                                                                                     Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)                                     
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
from hyperion.helpers import VectorClassReader as VCR
from hyperion.transforms import TransformList, LDA, LNorm, PCA
from hyperion.helpers import PLDAFactory as F


def remove_outlayers(x, class_ids):
    bound = 100
    idx = np.all(np.abs(x) < bound, axis=1)
    x = x[idx]
    class_ids = class_ids[idx]
    return x, class_idx


def train_be(
    iv_file,
    train_list,
    lda_dim,
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
    # x, class_ids = remove_outlayers(x, class_ids)
    rank = PCA.get_pca_dim_for_var_ratio(x, var_r=1)
    pca = None
    if rank < x.shape[1]:
        # do PCA if rank of x is smaller than its dimension
        pca = PCA(pca_dim=rank, name="pca")
        pca.fit(x)
        x = pca.predict(x)
        if lda_dim > rank:
            lda_dim = rank
        if y_dim > rank:
            y_dim = rank
        logging.info("PCA rank=%d" % (rank))

    # Train LDA
    lda = LDA(lda_dim=lda_dim, name="lda")
    lda.fit(x, class_ids)

    x_lda = lda.predict(x)
    logging.info("PCA-LDA Elapsed time: %.2f s." % (time.time() - t1))

    # Train centering and whitening
    t1 = time.time()
    lnorm = LNorm(name="lnorm")
    lnorm.fit(x_lda)

    x_ln = lnorm.predict(x_lda)
    logging.info("LNorm Elapsed time: %.2f s." % (time.time() - t1))

    # Train PLDA
    t1 = time.time()

    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, name="plda")
    elbo = plda.fit(x_ln, class_ids, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs)

    logging.info("PLDA Elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    if pca is None:
        preproc = TransformList([lda, lnorm])
    else:
        preproc = TransformList([pca, lda, lnorm])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    preproc.save(output_path + "/lda_lnorm.h5")
    plda.save(output_path + "/plda.h5")

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo.csv", elbo, delimiter=",")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train Back-end")

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--lda-dim", dest="lda_dim", type=int, default=None)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_be(**namespace_to_dict(args))
