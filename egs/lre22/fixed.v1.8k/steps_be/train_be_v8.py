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
from hyperion.np.classifiers import GaussianSVMC as GSVM
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

non_langs = (
    "fra-can",
    "fra-fra",
    "fra-ntf",
    "afr-afr",
    "ara-acm",
    "ara-arz",
    "ara-jor",
    "ara-ksa",
    "ara-kuw",
    "ara-leb",
    "ara-mau",
    "ara-mor",
    "ara-oma",
    "ara-pal",
    "ara-qat",
    "ara-sud",
    "ara-syr",
    "ara-uae",
    "ara-yem",
    "ara-apc",
    "eng-gbr",
    "eng-usg",
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

    idx = np.zeros((len(segs),), dtype=bool)
    for lang in non_langs:
        idx_i = segs[class_name] == lang
        idx = np.logical_or(idx, idx_i)

    segs_non = segs.loc[idx].copy()
    segs_non[class_name] = "zzzzzz"
    if len(segs_non) > 0:
        x_non = v_reader.read(segs_non["id"], squeeze=True)
    else:
        x_non = None

    logging.info(
        "read %s got ntar: %d nnon: %d", train_list, len(segs_tar), len(segs_non)
    )
    return segs_tar, x_tar, segs_non, x_non


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
    lre17_v_file,
    lre17_list,
    cv_v_file,
    cv_list,
    afr_v_file,
    afr_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    svm,
    output_dir,
    ood_weight,
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

    segs_sre_tar, x_sre_tar, segs_sre_non, x_sre_non = read_ood_data(
        sre_list, sre_v_file, class_name,
    )
    _, _, segs_lre17_non, x_lre17_non = read_ood_data(
        lre17_list, lre17_v_file, class_name,
    )
    segs_cv_tar, x_cv_tar, segs_cv_non, x_cv_non = read_ood_data(
        cv_list, cv_v_file, class_name
    )
    segs_afr_tar, x_afr_tar, segs_afr_non, x_afr_non = read_ood_data(
        afr_list, afr_v_file, class_name,
    )

    # class_ids = train_segs[class_name].values
    # labels, y_true = np.unique(class_ids, return_inverse=True)
    # gbe = GBE()
    # gbe.fit(x_trn, y_true)
    # scores_non = np.max(gbe(x_non), axis=1)
    # sel_non = np.argsort(scores_non)[-num_nons:]
    # segs_non = segs_non.iloc[sel_non]
    # x_non = x_non[sel_non]
    # logging.info("selected %d non-tar segments", x_non.shape[0])

    # class_ids = (
    #     list(train_segs[class_name].values)
    #     + list(segs_sre_tar[class_name].values)
    #     + list(segs_cv_tar[class_name].values)
    #     + list(segs_afr_tar[class_name].values)
    #     + list(segs_sre_non[class_name].values)
    #     + list(segs_lre17_non[class_name].values)
    #     + list(segs_cv_non[class_name].values)
    #     + list(segs_afr_non[class_name].values)
    # )
    # x_trn = np.concatenate(
    #     (
    #         x_trn,
    #         x_sre_tar,
    #         x_cv_tar,
    #         x_afr_tar,
    #         x_sre_non,
    #         x_lre17_non,
    #         x_cv_non,
    #         x_afr_non,
    #     ),
    #     axis=0,
    # )
    class_ids = (
        list(train_segs[class_name].values)
        + list(segs_sre_tar[class_name].values)
        + list(segs_cv_tar[class_name].values)
        + list(segs_afr_tar[class_name].values)
        + list(segs_sre_non[class_name].values)
        + list(segs_lre17_non[class_name].values)
        + list(segs_cv_non[class_name].values)
        + list(segs_afr_non[class_name].values)
    )
    x = np.concatenate(
        (
            x_trn,
            x_sre_tar,
            x_cv_tar,
            x_afr_tar,
            x_sre_non,
            x_lre17_non,
            x_cv_non,
            x_afr_non,
        ),
        axis=0,
    )
    sample_weight = np.concatenate(
        (
            np.ones((len(train_segs),)),
            ood_weight * np.ones((len(segs_sre_tar),)),
            ood_weight * np.ones((len(segs_cv_tar),)),
            ood_weight * np.ones((len(segs_afr_tar),)),
            ood_weight * np.ones((len(segs_sre_non),)),
            np.ones((len(segs_lre17_non),)),
            ood_weight * np.ones((len(segs_cv_non),)),
            ood_weight * np.ones((len(segs_afr_non),)),
        )
    )

    labels, y_true = np.unique(class_ids, return_inverse=True)
    logging.info("%d training samples", x_trn.shape[0])

    logging.info("PCA args=%s", str(pca))
    pca_var_r = pca["pca_var_r"]
    pca_dim = pca["pca_dim"]
    if pca_var_r is not None and pca_var_r < 1.0 or pca_dim is not None:
        logging.info("training PCA")
        pca = PCA(**pca)
        pca.fit(x_trn)
        logging.info("PCA dimension: %d", pca.pca_dim)
        logging.info("apply PCA")
        x = pca(x)
    else:
        pca = None

    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("training whitening")
            lnorm.fit(x)

        logging.info("apply lnorm")
        x = lnorm(x)
    else:
        lnorm = None

    logging.info("SVM args=%s", str(svm))
    model = GSVM(labels=labels, **svm)
    model.fit(x, y_true, sample_weight=sample_weight)
    logging.info("trained SVM")
    scores = model(x)
    y_pred = np.argmax(scores, axis=-1)

    compute_metrics(y_true, y_pred, labels)

    logging.info("Saving transforms and SVM")
    transforms = []
    if pca is not None:
        transforms.append(pca)
    if lnorm is not None:
        transforms.append(lnorm)

    if transforms:
        transforms = TransformList(transforms)
        transforms.save(output_dir / "transforms.h5")

    model_labels = list(np.copy(model.labels))
    if "zzzzzz" in model_labels:
        model_labels.remove("zzzzzz")
    model.labels = model_labels
    print("model.labels before save", np.shape(model.labels))
    model.save(output_dir / "model_svm.h5")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train linear SVM Classifier",)

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--sre-v-file", required=True)
    parser.add_argument("--sre-list", required=True)
    parser.add_argument("--lre17-v-file", required=True)
    parser.add_argument("--lre17-list", required=True)
    parser.add_argument("--cv-v-file", required=True)
    parser.add_argument("--cv-list", required=True)
    parser.add_argument("--afr-v-file", required=True)
    parser.add_argument("--afr-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    GSVM.add_class_args(parser, prefix="svm")
    parser.add_argument("--class-name", default="class_id")
    # parser.add_argument("--num-nons", default=10000, type=int)
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--ood-weight", default=0.1, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
