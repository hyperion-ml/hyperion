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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from hyperion.hyp_defs import config_logger
from hyperion.utils import Utt2Info
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.transforms import LDA

colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def proj_attack_lda(
    train_v_file, train_list, test_v_file, test_list, lda_cols, title, output_path
):

    train_utts = Utt2Info.load(train_list)
    test_utts = Utt2Info.load(test_list)

    train_reader = DRF.create(train_v_file)
    x_trn = train_reader.read(train_utts.key, squeeze=True)
    del train_reader

    test_reader = DRF.create(test_v_file)
    x_test = test_reader.read(test_utts.key, squeeze=True)
    del test_reader

    lda_cols = np.asarray(lda_cols) - 1
    lda_classes = train_utts.info[:, lda_cols]
    if lda_classes.shape[1] > 1:
        new_classes = []
        for i in range(lda_classes.shape[0]):
            new_classes.append(str(lda_classes[i]))
        lda_classes = np.asarray(new_classes)
    lda_classes, class_ids = np.unique(lda_classes, return_inverse=True)

    lda = LDA(lda_dim=2)

    lda.fit(x_trn, class_ids)
    x_lda_trn = lda.predict(x_trn)
    x_lda_test = lda.predict(x_test)
    p_test = np.random.rand(x_test.shape[0]) < 0.05
    x_lda_test = x_lda_test[p_test]
    for col in range(3):
        fig_file = "%s/train_lda_%d.png" % (output_path, col)
        classes = train_utts.info[:, col]
        classes, class_ids = np.unique(classes, return_inverse=True)
        for c in range(np.max(class_ids) + 1):
            idx = class_ids == c
            plt.scatter(
                x_lda_trn[idx, 0],
                x_lda_trn[idx, 1],
                c=color_marker[c][0],
                marker=color_marker[c][1],
                label=classes[c],
            )

        plt.legend()
        plt.grid(True)
        plt.show()
        plt.title("Train-set LDA(%s)" % title)
        plt.savefig(fig_file)
        plt.clf()

        fig_file = "%s/test_lda_%d.png" % (output_path, col)
        classes = test_utts.info[p_test, col]
        classes, class_ids = np.unique(classes, return_inverse=True)
        for c in range(np.max(class_ids) + 1):
            idx = class_ids == c
            plt.scatter(
                x_lda_test[idx, 0],
                x_lda_test[idx, 1],
                c=color_marker[c][0],
                marker=color_marker[c][1],
                label=classes[c],
            )

        plt.legend()
        plt.grid(True)
        plt.show()
        plt.title("Test-set LDA(%s)" % title)
        plt.savefig(fig_file)
        plt.clf()


if __name__ == "__main__":

    parser = ArgumentParser(description="Proj x-vector with LDA to classify attacks")

    parser.add_argument("--train-v-file", required=True)
    parser.add_argument("--train-list", required=True)

    parser.add_argument("--test-v-file", required=True)
    parser.add_argument("--test-list", required=True)

    parser.add_argument("--lda-cols", type=int, nargs="+", required=True)
    parser.add_argument("--title", default="")

    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    proj_attack_lda(**namespace_to_dict(args))
