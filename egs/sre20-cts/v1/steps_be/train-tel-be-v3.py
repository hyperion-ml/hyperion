#!/usr/bin/env python
"""
Trains Backend for SRE20 tel condition
"""
import logging
import sys
import os
import argparse
import time

import numpy as np

from hyperion.helpers import VectorClassReader as VCR
from hyperion.helpers import VectorReader as VR
from hyperion.transforms import TransformList, PCA, LDA, LNorm
from hyperion.helpers import PLDAFactory as F
from hyperion.utils import Utt2Info

from numpy.linalg import matrix_rank


def train_be(
    iv_file,
    train_list,
    adapt_iv_file,
    adapt_list,
    lda_dim,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    w_mu,
    w_B,
    w_W,
    output_path,
    **kwargs
):

    # Read data
    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, train_list, None, **vcr_args)
    x, class_ids = vcr.read()

    # Train LDA
    t1 = time.time()
    rank = matrix_rank(x)
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

    lda = LDA(lda_dim=lda_dim, name="lda")
    lda.fit(x, class_ids)

    x_lda = lda.predict(x)
    logging.info("LDA Elapsed time: %.2f s." % (time.time() - t1))

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

    preproc = TransformList(lda)
    preproc.append(lnorm)

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/lda_lnorm.h5")
    plda.save(output_path + "/plda.h5")

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo.csv", elbo, delimiter=",")

    # Load labeled adapt data
    vcr = VCR(adapt_iv_file, adapt_list, None)
    x_adapt, class_ids_adapt = vcr.read()

    if pca:
        x_adapt = pca.predict(x_adapt)
    x_adapt_lda = lda.predict(x_adapt)
    lnorm.update_T = False
    lnorm.fit(x_adapt_lda)

    if pca is None:
        preproc = TransformList([lda, lnorm])
    else:
        preproc = TransformList([pca, lda, lnorm])

    preproc.save(output_path + "/lda_lnorm_adapt.h5")

    x_adapt_ln = lnorm.predict(x_adapt_lda)

    plda_adapt1 = plda.copy()
    if np.max(class_ids_adapt) + 1 < plda.y_dim:
        plda.update_V = False

    elbo = plda.fit(x_adapt_ln, class_ids_adapt, epochs=20)
    plda_adapt1.weighted_avg_model(plda, w_mu, w_B, w_W)
    plda_adapt1.save(output_path + "/plda_adapt.h5")

    num = np.arange(20)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo_adapt.csv", elbo, delimiter=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train Back-end for SRE20 telephone condition",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--train-list", dest="train_list", required=True)
    parser.add_argument("--adapt-iv-file", dest="adapt_iv_file", required=True)
    parser.add_argument("--adapt-list", dest="adapt_list", required=True)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lda-dim", type=int, default=150)
    parser.add_argument("--w-mu", dest="w_mu", type=float, default=1)
    parser.add_argument("--w-b", dest="w_B", type=float, default=0.5)
    parser.add_argument("--w-w", dest="w_W", type=float, default=0.5)

    args = parser.parse_args()

    train_be(**vars(args))
