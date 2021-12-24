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
from hyperion.transforms import PCA, SklTSNE, LNorm

colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def proj_attack_tsne(
    train_v_file, train_list, pca_var_r, prob_plot, lnorm, title, output_path, **kwargs
):

    train_utts = Utt2Info.load(train_list)

    train_reader = DRF.create(train_v_file)
    x_trn = train_reader.read(train_utts.key, squeeze=True)
    del train_reader

    if lnorm:
        x_trn = LNorm().predict(x_trn)

    if pca_var_r < 1:
        pca = PCA(pca_var_r=pca_var_r)
        pca.fit(x_trn)
        x_pca = pca.predict(x_trn)
        logging.info("pca-dim={}".format(x_pca.shape[1]))
    else:
        x_pca = x_trn

    tsne_args = SklTSNE.filter_args(**kwargs["tsne"])
    tsne = SklTSNE(**tsne_args)
    x_tsne = tsne.fit(x_pca)
    p = np.random.rand(x_tsne.shape[0]) < prob_plot
    x_tsne = x_tsne[p]

    for col in range(1):
        fig_file = "%s/train_tsne_%d.png" % (output_path, col)
        classes = train_utts.info[p]
        classes, class_ids = np.unique(classes, return_inverse=True)
        for c in range(np.max(class_ids) + 1):
            idx = class_ids == c
            plt.scatter(
                x_tsne[idx, 0],
                x_tsne[idx, 1],
                c=color_marker[c][0],
                marker=color_marker[c][1],
                label=classes[c],
            )

        plt.legend()
        plt.grid(True)
        plt.title(title)
        plt.savefig(fig_file)
        plt.clf()


if __name__ == "__main__":

    parser = ArgumentParser(description="Proj x-vector with TSNE to visualize attacks")

    parser.add_argument("--train-v-file", required=True)
    parser.add_argument("--train-list", required=True)

    parser.add_argument("--pca-var-r", default=0.95, type=float)
    parser.add_argument("--prob-plot", default=0.1, type=float)
    parser.add_argument("--lnorm", default=False, action="store_true")
    parser.add_argument("--title", default="")
    SklTSNE.add_class_args(parser, prefix="tsne")

    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    proj_attack_tsne(**namespace_to_dict(args))
