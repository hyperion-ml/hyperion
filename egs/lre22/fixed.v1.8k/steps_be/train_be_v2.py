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
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearSVMC as SVM
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
    print_confusion_matrix(C * 100, labels)


def train_be(
    v_file,
    train_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    svm,
    output_dir,
    verbose,
):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader
    class_ids = train_segs[class_name]
    labels, y_true = np.unique(class_ids, return_inverse=True)
    logging.info("loaded %d samples", x_trn.shape[0])

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

    logging.info("SVM args=%s", str(svm))
    model = SVM(labels=labels, **svm)
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

    model.save(output_dir / "model_svm.h5")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Train linear SVM Classifier",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    SVM.add_class_args(parser, prefix="svm")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_be(**namespace_to_dict(args))
