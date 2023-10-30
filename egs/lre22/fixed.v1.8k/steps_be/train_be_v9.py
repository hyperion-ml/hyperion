#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
    ActionYesNo,
)
import time
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.utils.math import softmax
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)

tar_langs = (
    "afr-afr",
    "ara-aeb",
    "ara-arq",
    "ara-ayl",
    "eng-ens",
    "eng-iaf",
    "fra-ntf",
    "nbl-nbl",
    "orm-orm",
    "tir-tir",
    "tso-tso",
    "ven-ven",
    "xho-xho",
    "zul-zul",
)


def read_ood_data(train_list, v_file, class_name):
    v_reader = DRF.create(v_file)

    segs = SegmentSet.load(train_list)
    idx = np.zeros((len(segs),), dtype=bool)
    for lang in tar_langs:
        idx_i = segs[class_name] == lang
        idx = np.logical_or(idx, idx_i)

    segs_tar = segs.loc[idx].copy()
    if len(segs_tar) > 0:
        x_tar = v_reader.read(segs_tar["id"], squeeze=True)
    else:
        x_tar = None

    logging.info(
        "read %s got ntar: %d", train_list, len(segs_tar),
    )
    return segs_tar, x_tar


def compute_metrics(y_true, y_pred, labels):

    acc = compute_accuracy(y_true, y_pred)
    logging.info("training acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels, fmt=".2f")


def train_be(
    v_file,
    train_list,
    sre_v_file,
    sre_list,
    cv_v_file,
    cv_list,
    afr_v_file,
    afr_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    gbe,
    output_dir,
    verbose,
):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    v_reader = DRF.create(v_file)
    x_trn = v_reader.read(train_segs["id"], squeeze=True)
    del v_reader
    logging.info("loaded %d train samples", x_trn.shape[0])

    segs_sre, x_sre = read_ood_data(sre_list, sre_v_file, class_name,)
    segs_cv, x_cv = read_ood_data(cv_list, cv_v_file, class_name)
    segs_afr, x_afr = read_ood_data(afr_list, afr_v_file, class_name,)

    class_ids_trn = train_segs[class_name].values
    x_ood = np.concatenate((x_sre, x_cv, x_afr), axis=0)
    class_ids_ood = (
        list(segs_sre[class_name].values)
        + list(segs_cv[class_name].values)
        + list(segs_afr[class_name].values)
    )

    labels, y_true_trn = np.unique(class_ids_trn, return_inverse=True)
    _, y_true_ood = np.unique(
        np.concatenate((labels, class_ids_ood)), return_inverse=True
    )
    y_true_ood = y_true_ood[len(labels) :]

    logging.info("%d ood samples", x_ood.shape[0])
    logging.info("%d training samples", x_trn.shape[0])

    x_ood += np.mean(x_trn, axis=0, keepdims=True) - np.mean(
        x_ood, axis=0, keepdims=True
    )
    logging.info("PCA args=%s", str(pca))
    pca_var_r = pca["pca_var_r"]
    pca_dim = pca["pca_dim"]
    if pca_var_r is not None and pca_var_r < 1.0 or pca_dim is not None:
        logging.info("training PCA")
        pca = PCA(**pca)
        pca.fit(x_trn)
        logging.info("PCA dimension: %d", pca.pca_dim)
        logging.info("apply PCA")
        x_trn = pca(x_trn)
        x_ood = pca(x_ood)
    else:
        pca = None

    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("training whitening")
            lnorm.fit(x_trn)

        logging.info("apply lnorm")
        x_trn = lnorm(x_trn)
        x_ood = lnorm(x_ood)
    else:
        lnorm = None

    prior_0 = GBE(
        mu=np.zeros((len(labels), x_trn.shape[1])),
        W=np.eye(x_trn.shape[1]),
        beta=16,
        nu=x_trn.shape[1],
    )
    print(prior_0.__dict__)
    prior = GBE(prior=prior_0)
    prior.fit(x_ood, y_true_ood)
    prior.nu = 0.1 * prior.nu
    prior.beta = 0.01 * prior.beta
    print(prior.__dict__)
    model = GBE(labels=labels, prior=prior)
    model.fit(x_trn, y_true_trn)
    print(model.__dict__, flush=True)
    logging.info("trained GBE")
    scores = model(x_trn)
    y_pred = np.argmax(scores, axis=-1)

    compute_metrics(y_true_trn, y_pred, labels)

    logging.info("Saving transforms and GBE")
    transforms = []
    if pca is not None:
        transforms.append(pca)
    if lnorm is not None:
        transforms.append(lnorm)

    if transforms:
        transforms = TransformList(transforms)
        transforms.save(output_dir / "transforms.h5")

    model.save(output_dir / "model_gbe.h5")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train linear GBE Classifier",)

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--sre-v-file", required=True)
    parser.add_argument("--sre-list", required=True)
    parser.add_argument("--cv-v-file", required=True)
    parser.add_argument("--cv-list", required=True)
    parser.add_argument("--afr-v-file", required=True)
    parser.add_argument("--afr-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    GBE.add_class_args(parser, prefix="gbe")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
