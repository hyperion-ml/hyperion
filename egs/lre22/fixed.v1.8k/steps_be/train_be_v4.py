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
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearGBE as GBE
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)


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
    lre17_v_file,
    lre17_list,
    class_name,
    do_lnorm,
    whiten,
    ary_thr,
    # num_nons,
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

    segs_lre17 = SegmentSet.load(lre17_list)
    ary_idx = segs_lre17[class_name] == "ara-ary"
    # lre17_segs.loc[ara_ary_idx, class_name] = "ara-ayl"  # "ara-arq"  # "ara-aeb"
    segs_ary = segs_lre17.loc[ary_idx]

    logging.info("label maghrebi arabic samples")
    v_reader = DRF.create(lre17_v_file)
    x_ary = v_reader.read(segs_ary["id"], squeeze=True)
    logging.info("loaded %d lre17 ara-ary samples", x_ary.shape[0])

    ara_idx = train_segs[class_name].isin(["ara-ayl", "ara-arq", "ara-aeb"])
    x_ara = x_trn[ara_idx]
    class_ids_ara = train_segs.loc[ara_idx, class_name].values

    gbe_ara = GBE()
    labels_ara, y_true_ara = np.unique(class_ids_ara, return_inverse=True)
    gbe_ara.fit(x_ara, y_true_ara)
    scores_ary = gbe_ara(x_ary)
    y_pred_ary = np.argmax(scores_ary, axis=-1)
    p_ary = np.max(softmax(scores_ary, axis=-1), axis=-1)
    sel_ary = p_ary > ary_thr
    segs_ary = segs_ary.loc[sel_ary]
    y_pred_ary = y_pred_ary[sel_ary]
    x_ary = x_ary[sel_ary]
    segs_ary[class_name] = [labels_ara[c] for c in y_pred_ary]
    logging.info("selected %d ara-ary segments", x_ary.shape[0])
    segs_ary["p"] = p_ary[sel_ary]
    SegmentSet(segs_ary).save(output_dir / "segs_ary.csv")

    # logging.info("selecting non-target segments")
    # segs_non = segs_lre17.loc[~ary_idx].copy()
    # segs_non[class_name] = "zzzzzz"
    # x_non = v_reader.read(segs_non["id"], squeeze=True)
    # logging.info("loaded %d lre17 non-tar samples", x_non.shape[0])

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
    #     + list(segs_ary[class_name].values)
    #     + list(segs_non[class_name].values)
    # )
    # x_trn = np.concatenate((x_trn, x_ary, x_non), axis=0)
    class_ids = list(train_segs[class_name].values) + list(segs_ary[class_name].values)
    x_trn = np.concatenate((x_trn, x_ary), axis=0)
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
        x_trn = pca(x_trn)
    else:
        pca = None

    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("training whitening")
            lnorm.fit(x_trn)

        logging.info("apply lnorm")
        x_trn = lnorm(x_trn)
    else:
        lnorm = None

    logging.info("GBE args=%s", str(gbe))
    model = GBE(labels=labels, **gbe)
    model.fit(x_trn, y_true)
    logging.info("trained GBE")
    scores = model(x_trn)
    y_pred = np.argmax(scores, axis=-1)

    compute_metrics(y_true, y_pred, labels)

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

    parser = ArgumentParser(
        description="Train linear GBE Classifier",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--lre17-v-file", required=True)
    parser.add_argument("--lre17-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    GBE.add_class_args(parser, prefix="gbe")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--ary-thr", default=10, type=float)
    # parser.add_argument("--num-nons", default=10000, type=int)
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
