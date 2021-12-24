#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
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
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.metrics.acc import compute_accuracy
from hyperion.metrics.confusion_matrix import (
    compute_confusion_matrix,
    print_confusion_matrix,
)


def read_class_file(class_file):

    class_info = pd.read_csv(class_file, header=None, sep=" ")
    classes = class_info[0]
    class2ids = {str(k): i for i, k in enumerate(class_info[0])}
    return classes, class2ids


def eval_classif_perf(score_file, key_file, class_file, output_path=None, **kwargs):

    utts = Utt2Info.load(key_file)
    classes, class2ids = read_class_file(class_file)
    mask = [True if c in class2ids else False for c in utts.info]
    info = utts.info[mask]
    key = utts.key[mask]
    y_true = [class2ids[c] for c in info]

    reader = DRF.create(score_file)
    y = reader.read(key, squeeze=True)
    del reader

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


if __name__ == "__main__":

    parser = ArgumentParser(description="Evaluates attack classification accuracy")

    parser.add_argument("--score-file", required=True)
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

    eval_classif_perf(**namespace_to_dict(args))
