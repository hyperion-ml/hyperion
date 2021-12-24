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
from hyperion.transforms import TransformList, PCA, LDA, LNorm, CORAL
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
    w_coral_mu,
    w_coral_T,
    output_path,
    **kwargs
):

    # Read train data
    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, train_list, None, **vcr_args)
    x, class_ids = vcr.read()

    # Load labeled adapt data
    vcr = VCR(adapt_iv_file, adapt_list, None)
    x_adapt, class_ids_adapt = vcr.read()

    rank = matrix_rank(x)
    pca = None
    if rank < x.shape[1]:
        # do PCA if rank of x is smaller than its dimension
        pca = PCA(pca_dim=rank, name="pca")
        pca.fit(x)
        x = pca.predict(x)
        x_adapt = pca.predict(x_adapt)
        if lda_dim > rank:
            lda_dim = rank
        if y_dim > rank:
            y_dim = rank
        logging.info("PCA rank=%d" % (rank))

    # Train CORAL
    t1 = time.time()
    x_in = x_adapt
    coral = CORAL(alpha_mu=w_coral_mu, alpha_T=w_coral_T)
    coral.fit(x_in, x_out=x)
    # del x_in

    # Apply CORAL to out-of-domain data
    x_coral = coral.predict(x)

    # Train LDA
    x_lab_tot = np.concatenate((x_coral, x_adapt), axis=0)
    class_ids_lab_tot = np.concatenate((class_ids, class_ids_adapt))
    lda = LDA(lda_dim=lda_dim, name="lda")
    lda.fit(x_lab_tot, class_ids_lab_tot)
    del x_lab_tot

    # Apply LDA to all datasets
    x_lda = lda.predict(x_coral)
    x_adapt_lda = lda.predict(x_adapt)

    logging.info("LDA Elapsed time: %.2f s." % (time.time() - t1))

    # Train centering and whitening
    t1 = time.time()
    x_lda_all = np.concatenate((x_lda, x_adapt_lda), axis=0)
    lnorm = LNorm(name="lnorm")
    lnorm.fit(x_lda_all)
    del x_lda_all

    # Apply lnorm to out-of-domain data
    x_ln = lnorm.predict(x_lda)

    # Train centering for in-domain
    lnorm_in = lnorm.copy()
    lnorm_in.update_T = False
    x_lda_in = x_adapt_lda
    lnorm_in.fit(x_lda_in)

    # Apply lnorm to in-domain
    x_adapt_ln = lnorm_in.predict(x_adapt_lda)
    logging.info("LNorm Elapsed time: %.2f s." % (time.time() - t1))

    # Train PLDA
    t1 = time.time()
    x_lab_ln = np.concatenate((x_ln, x_adapt_ln), axis=0)
    class_ids_lab_tot = np.concatenate((class_ids, class_ids_adapt))
    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, name="plda")
    elbo = plda.fit(
        x_lab_ln, class_ids_lab_tot, epochs=epochs, ml_md=ml_md, md_epochs=md_epochs
    )

    logging.info("PLDA Elapsed time: %.2f s." % (time.time() - t1))

    # Save models
    if pca is None:
        preproc = TransformList([lda, lnorm])
    else:
        preproc = TransformList([pca, lda, lnorm])

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + "/lda_lnorm.h5")
    plda.save(output_path + "/plda.h5")

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo.csv", elbo, delimiter=",")

    if pca is None:
        preproc = TransformList([lda, lnorm_in])
    else:
        preproc = TransformList([pca, lda, lnorm_in])

    preproc.save(output_path + "/lda_lnorm_adapt.h5")

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

    parser.add_argument("--iv-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--adapt-iv-file", required=True)
    parser.add_argument("--adapt-list", required=True)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--lda-dim", dest="lda_dim", type=int, default=150)
    parser.add_argument("--w-coral-mu", dest="w_coral_mu", type=float, default=0.5)
    parser.add_argument("--w-coral-t", dest="w_coral_T", type=float, default=0.75)
    parser.add_argument("--w-mu", dest="w_mu", type=float, default=1)
    parser.add_argument("--w-b", dest="w_B", type=float, default=1)
    parser.add_argument("--w-w", dest="w_W", type=float, default=1)
    args = parser.parse_args()

    train_be(**vars(args))
