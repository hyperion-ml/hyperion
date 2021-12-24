#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import sys
import os
import argparse
import time

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.metrics.acc import compute_accuracy
from hyperion.metrics.confusion_matrix import (
    compute_confusion_matrix,
    print_confusion_matrix,
)
from hyperion.transforms import PCA, LNorm
from hyperion.pdfs import SPLDA
from numpy.linalg import matrix_rank

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# markers = ['x', 'o', '+', '*', 's', 'h', 'D', '^', 'v', 'p', '8']

# color_marker = [(c,m) for m in markers for c in colors]


def read_class_file(class_file):

    class_info = pd.read_csv(class_file, header=None, sep=" ")
    classes = class_info[0]
    class2ids = {str(k): i for i, k in enumerate(class_info[0])}
    return classes, class2ids


def eval_classif_perf(v_file, key_file, class_file, output_path=None, **kwargs):

    rng = np.random.RandomState(seed=11235)
    utts = Utt2Info.load(key_file)
    classes, class2ids = read_class_file(class_file)

    classes_test, y_true = np.unique(utts.info, return_inverse=True)
    reader = DRF.create(v_file)
    x = reader.read(utts.key, squeeze=True)
    del reader

    class_names = np.asarray(utts.info)
    class_names[class_names == "imp"] = "pgd-linf"
    classes, class_ids = np.unique(class_names, return_inverse=True)
    # divide train and test
    mask = rng.rand(len(x)) < 0.3
    x_train = x[mask]
    class_ids_train = class_ids[mask]
    x_test = x[mask == False]
    y_true = class_ids[mask == False]

    rank = matrix_rank(x_train)
    # do PCA if rank of x is smaller than its dimension
    pca = PCA(pca_dim=rank, name="pca")
    pca.fit(x_train)
    x_train = pca.predict(x_train)
    x_test = pca.predict(x_test)

    lnorm = LNorm(name="lnorm")
    lnorm.fit(x_train)
    x_train = lnorm.predict(x_train)
    x_test = lnorm.predict(x_test)

    plda = SPLDA(y_dim=min(max(class_ids) + 1, x_train.shape[1]))
    plda.fit(x_train, class_ids=class_ids_train)

    y = plda.llr_Nvs1(x_train, x_test, ids1=class_ids_train, method="book").T

    y_pred = np.argmax(y, axis=1)
    acc = compute_accuracy(y_true, y_pred)
    logging.info("Classification accuracy %.2f %%" % (acc * 100))

    labels = np.arange(len(classes), dtype=np.int)
    C = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=False)
    logging.info("Unnormalized Confusion Matrix:")
    print_confusion_matrix(C, labels_true=classes)

    Cn = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=True)
    logging.info("Normalized Confusion Matrix:")
    print_confusion_matrix(Cn * 100, labels_true=classes, fmt=".1f")

    # remove benign class, which has id=0
    mask = y_true > 0
    y = y[mask, 1:]
    y_true = y_true[mask] - 1
    logging.info("without benign class")
    y_pred = np.argmax(y, axis=1)
    acc = compute_accuracy(y_true, y_pred)
    logging.info("Classification accuracy %.2f %%" % (acc * 100))

    labels = np.arange(len(classes) - 1, dtype=np.int)
    C = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=False)
    logging.info("Unnormalized Confusion Matrix:")
    print_confusion_matrix(C, labels_true=classes[1:])

    Cn = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=True)
    logging.info("Normalized Confusion Matrix:")
    print_confusion_matrix(Cn * 100, labels_true=classes[1:], fmt=".1f")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Evaluates attack classification accuracy",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--class-file", required=True)

    # parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_classif_perf(**vars(args))
