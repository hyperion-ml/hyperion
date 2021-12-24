#!/usr/bin/env python
"""
Trains a back-end Backend per each trial side
"""
import logging
import sys
import os
import argparse
import time

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.pdfs import PLDA
from hyperion.transforms import TransformList, PCA, LDA, LNorm
from hyperion.helpers import PLDAFactory as F
from hyperion.utils.utt2info import Utt2Info
from hyperion.utils.math import cosine_scoring

from numpy.linalg import matrix_rank


def train_be(
    x_et,
    x_trn,
    class_ids_trn,
    u2c_trn,
    lda_dim,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    k_nn,
    output_path,
):

    t1 = time.time()
    x_et_avg = np.mean(x_et, axis=0, keepdims=True)
    D_trn = PLDA.compute_stats_hard(x_trn, class_ids=class_ids_trn)
    x_trn_avg = D_trn[1] / np.expand_dims(D_trn[0], axis=-1)
    scores = cosine_scoring(x_et_avg, x_trn_avg)
    cohort_class_idx = np.argsort(-scores[0])[:k_nn]
    cohort_seg_mask = np.zeros((x_trn.shape[0],), dtype=np.bool)
    for i in range(k_nn):
        idx_i = class_ids_trn == cohort_class_idx[i]
        cohort_seg_mask[idx_i] = True

    x_trn = x_trn[cohort_seg_mask]
    class_ids_trn = class_ids_trn[cohort_seg_mask]
    u2c_trn = Utt2Info(u2c_trn.utt_info[cohort_seg_mask])

    x = np.concatenate((x_trn, x_et), axis=0)
    class_ids_et = (np.max(class_ids_trn) + 1) * np.ones((x_et.shape[0],), dtype=np.int)
    class_ids = np.concatenate((class_ids_trn, class_ids_et), axis=0)
    _, class_ids = np.unique(class_ids, return_inverse=True)  # make classids 0-(N-1)

    t1 = time.time()
    rank = matrix_rank(x)
    pca = None
    logging.info("x rank=%d" % (rank))
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    preproc.save(output_path + "/lda_lnorm.h5")
    plda.save(output_path + "/plda.h5")

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + "/elbo.csv", elbo, delimiter=",")

    u2c_trn.save(output_path + "/knn")


def train_bes(
    v_file_train,
    train_list,
    v_file_enroll_test,
    enroll_test_list,
    lda_dim,
    plda_type,
    y_dim,
    z_dim,
    epochs,
    ml_md,
    md_epochs,
    k_nn,
    output_path,
    part_idx,
    num_parts,
    **kwargs
):

    # Read train data
    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(v_file_train, train_list, None, **vcr_args)
    x_trn, class_ids_trn = vcr.read()
    u2c_trn = vcr.u2c
    del vcr

    reader = DRF.create(v_file_enroll_test)
    u2c = Utt2Info.load(enroll_test_list)
    u2c = u2c.split(part_idx, num_parts, group_by_field=1)
    class_names_et, class_ids_et = np.unique(u2c.info, return_inverse=True)
    num_classes_et = np.max(class_ids_et) + 1
    for c in range(num_classes_et):
        logging.info("Training PLDA for %s" % (class_names_et[c]))
        sel_idx = class_ids_et == c
        key_c = u2c.key[sel_idx]
        x_et = reader.read(key_c, squeeze=True)
        output_path_c = output_path + "/" + class_names_et[c]
        train_be(
            x_et,
            x_trn,
            class_ids_trn,
            u2c_trn,
            lda_dim,
            plda_type,
            y_dim,
            z_dim,
            epochs,
            ml_md,
            md_epochs,
            k_nn,
            output_path_c,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Train a Back-end for each trial side using kNN",
    )

    parser.add_argument("--v-file-train", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--v-file-enroll-test", required=True)
    parser.add_argument("--enroll-test-list", required=True)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lda-dim", type=int, default=150)
    parser.add_argument("--k-nn", type=int, default=500)
    parser.add_argument("--part-idx", type=int, default=1)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    train_bes(**vars(args))
