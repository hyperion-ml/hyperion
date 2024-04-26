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
from hyperion.np.classifiers import LinearSVMC as SVM
from hyperion.np.classifiers import GaussianSVMC as GSVM
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
    voxlingua_v_file,
    voxlingua_list,
    class_name,
    do_lnorm,
    whiten,
    ary_thr,
    num_nons,
    pca,
    svm,
    output_dir,
    verbose,
    do_vl,
    do_lre17,
):
    print(locals(), flush=True)
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    v_reader = DRF.create(v_file)
    x_trn = v_reader.read(train_segs["id"], squeeze=True)
    del v_reader
    logging.info("loaded %d train samples", x_trn.shape[0])

    x_ary = []
    x_non = []
    y_ary = []
    y_non = []

    if do_lre17:
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
        logp_ary = np.max(softmax(scores_ary, axis=-1), axis=-1)
        print(logp_ary, y_pred_ary)
        # dscores_ary = np.diff(np.sort(scores_ary, axis=-1), axis=-1)[:, -1]
        # sel_ary = dscores_ary > ary_thr
        sel_ary = logp_ary > ary_thr
        segs_ary = segs_ary.loc[sel_ary]
        y_pred_ary = y_pred_ary[sel_ary]
        x_ary = x_ary[sel_ary]
        segs_ary[class_name] = [labels_ara[c] for c in y_pred_ary]
        logging.info("selected %d ara-ary segments", x_ary.shape[0])
        segs_ary["logp"] = logp_ary[sel_ary]
        SegmentSet(segs_ary).save(output_dir / "segs_ary.csv")

        logging.info("selecting non-target segments")
        lre17_close_idx = segs_lre17[class_name].isin(
            ["ara-acm", "ara-apc", "eng-usg", "por-brz"]
        )
        segs_non = segs_lre17.loc[lre17_close_idx].copy()
        segs_non[class_name] = "zzzzzz"
        x_non = v_reader.read(segs_non["id"], squeeze=True)
        logging.info("loaded %d lre17 non-tar samples", x_non.shape[0])

        y_ary = list(segs_ary[class_name].values)
        y_non = list(segs_non[class_name].values)

    # class_ids = train_segs[class_name].values
    # labels, y_true = np.unique(class_ids, return_inverse=True)
    # gbe = GBE()
    # gbe.fit(x_trn, y_true)
    # scores_non = np.max(gbe(x_non), axis=1)
    # sel_non = np.argsort(scores_non)[-num_nons:]
    # segs_non = segs_non.iloc[sel_non]
    # x_non = x_non[sel_non]
    # logging.info("selected %d non-tar segments", x_non.shape[0])

    if do_vl:
        v_reader_vl = DRF.create(voxlingua_v_file)
        segs_voxlingua = SegmentSet.load(voxlingua_list)
        vl_close_idx = segs_voxlingua[class_name].isin(
            [
                "en-en",
                "am-am",
                "sn-sn",
                "fra-mix",
                "haw-haw",
                "zho-cmn",
                "ia-ia",
                "ceb-ceb",
                "sa-sa",
                "su-su",
                "te-te",
                "yo-yo",
                "sw-sw",
                "pt-pt",
                "war-war",
                "km-km",
                "tr-tr",
                "gn-gn",
            ]
        )
        segs_vl_close = segs_voxlingua.loc[vl_close_idx].copy()
        segs_vl_close[class_name] = "zzzzzz"
        x_non_vl = v_reader_vl.read(segs_vl_close["id"], squeeze=True)

        vl_afk_idx = segs_voxlingua[class_name] == "afr-afr"
        if not np.any(vl_afk_idx):
            vl_afk_idx = segs_voxlingua[class_name] == "af-af"
        segs_vl_afk = segs_voxlingua.loc[vl_afk_idx].copy()
        segs_vl_afk[class_name] = "afr-afr"
        x_trn_vl = v_reader_vl.read(segs_vl_afk["id"], squeeze=True)

        y_trn_vl = list(segs_vl_afk[class_name].values)
        y_non_vl = list(segs_vl_close[class_name].values)

        del v_reader_vl
    else:
        x_trn_vl = np.zeros((0, x_trn.shape[1]))
        x_non_vl = np.zeros((0, x_trn.shape[1]))
        y_trn_vl = []
        y_non_vl = []

    class_ids = (
        list(train_segs[class_name].values) + y_trn_vl + y_ary + y_non + y_non_vl
    )
    x_trn = np.concatenate((x_trn, x_trn_vl, x_ary, x_non, x_non_vl), axis=0)
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

    logging.info("Gaussian SVM args=%s", str(svm))
    model = GSVM(labels=labels, **svm)
    model.fit(x_trn, y_true)
    logging.info("trained SVM")
    scores = model(x_trn)
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

    #    model.svm.coef_ = model.svm.coef_[:-1]
    #    model.svm.intercept_ = model.svm.intercept_[:-1]
    model_labels = list(np.copy(model.labels))
    if "zzzzzz" in model_labels:
        model_labels.remove("zzzzzz")
    model.labels = model_labels
    print("model.labels before save", np.shape(model.labels))
    model.save(output_dir / "model_svm.h5")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train gaussian SVM Classifier",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--lre17-v-file", required=True)
    parser.add_argument("--lre17-list", required=True)
    parser.add_argument("--voxlingua-v-file", required=True)
    parser.add_argument("--voxlingua-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    GSVM.add_class_args(parser, prefix="svm")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--ary-thr", default=10, type=float)
    parser.add_argument("--num-nons", default=10000, type=int)
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--do-vl", default=True, action=ActionYesNo)
    parser.add_argument("--do-lre17", default=True, action=ActionYesNo)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    #    parser.add_argument("--classifier", default="lsvm", choices=["lsvm", "gsvm", "rf"], required=False)

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
