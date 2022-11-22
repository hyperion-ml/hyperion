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
    ActionYesNo,
)
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.utils.math import cosine_scoring
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import PCA, SklTSNE, LNorm
from hyperion.np.clustering import AHC


matplotlib.use("Agg")
colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def plot_embedding_tsne(
    train_v_file,
    train_list,
    pca_var_r,
    prob_plot,
    lnorm,
    title,
    max_classes,
    plot_class_name,
    do_ahc,
    cluster_tsne,
    num_clusters,
    ahc_thr,
    output_dir,
    **kwargs,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(train_v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader
    logging.info("loaded %d samples", x_trn.shape[0])
    if lnorm:
        x_trn = LNorm().predict(x_trn)

    if pca_var_r < 1:
        pca = PCA(pca_var_r=pca_var_r)
        pca.fit(x_trn)
        x_pca = pca.predict(x_trn)
        logging.info("pca-dim=%d", x_pca.shape[1])
    else:
        x_pca = x_trn

    class_ids = train_segs[plot_class_name]
    classes, class_idx = np.unique(class_ids, return_inverse=True)
    if max_classes is not None:
        index = class_idx < max_classes
        x_pca = x_pca[index]
        class_idx = class_idx[index]

    tsne_args = SklTSNE.filter_args(**kwargs["tsne"])
    tsne = SklTSNE(**tsne_args)
    if do_ahc:
        ahc = AHC()
        global_subclass_idx = np.zeros_like(class_idx)

    for c in range(np.max(class_idx) + 1):
        fig_file = f"{output_dir}/train_tsne_{plot_class_name}_{classes[c]}.png"
        idx = class_idx == c
        logging.info("plot class %s with %d samples", classes[c], np.sum(idx))
        x_c = x_pca[idx]
        x_tsne = tsne.fit(x_c)
        if do_ahc:
            if cluster_tsne:
                # in the low dim space, we cannot use cosine scoring
                x2 = np.sum(x_tsne ** 2, axis=1)[:, None]
                d2 = x2 - 2 * np.dot(x_tsne, x_tsne.T) + x2.T
                d2 = np.clip(d2, a_min=0, a_max=None)
                scores = -np.sqrt(d2)
            else:
                scores = cosine_scoring(x_c, x_c)
            ahc.fit(scores)
            if num_clusters is None:
                subclass_idx_c = ahc.get_flat_clusters(ahc_thr)
            else:
                subclass_idx_c = ahc.get_flat_clusters(num_clusters, "num_clusters")
            global_subclass_idx[idx] = subclass_idx_c

        p = np.random.rand(x_tsne.shape[0]) <= prob_plot
        x_tsne = x_tsne[p]
        logging.info("plots %d samples", x_tsne.shape[0])
        if do_ahc:
            subclass_idx_c = subclass_idx_c[p]
            for sc in range(min(np.max(subclass_idx_c) + 1, len(color_marker))):
                idx_sc = subclass_idx_c == sc
                plt.scatter(
                    x_tsne[idx_sc, 0],
                    x_tsne[idx_sc, 1],
                    c=color_marker[sc][0],
                    marker=color_marker[sc][1],
                )
        else:
            plt.scatter(
                x_tsne[:, 0],
                x_tsne[:, 1],
                c=color_marker[0][0],
                marker=color_marker[0][1],
            )

        # plt.legend()
        plt.grid(True)
        plt.title(f"{title} {classes[c]}")
        plt.savefig(fig_file)
        plt.clf()

    if do_ahc:
        # subclass_ids = [f"{a}-{b}" for a, b in zip(class_ids, global_subclass_idx)]
        # _, subclass_idx = np.unique(subclass_ids, return_inverse=True)
        # train_segs["subclass_id"] = subclass_ids
        train_segs["subclass_idx"] = global_subclass_idx
        train_segs.save(output_dir / "segments.csv")


if __name__ == "__main__":

    parser = ArgumentParser(
        description=(
            "Projects embeddings using TSNE, "
            "plots a TSNE per class to discover subclusters inside of the classes"
        )
    )

    parser.add_argument("--train-v-file", required=True)
    parser.add_argument("--train-list", required=True)

    parser.add_argument("--pca-var-r", default=0.95, type=float)
    parser.add_argument("--prob-plot", default=0.1, type=float)
    parser.add_argument("--lnorm", default=False, action=ActionYesNo)
    parser.add_argument(
        "--plot-class-name",
        default="class_id",
        help="names of the class column we plot",
    )
    parser.add_argument("--title", default="")
    SklTSNE.add_class_args(parser, prefix="tsne")

    parser.add_argument(
        "--max-classes", default=None, type=int, help="max number of clases to plot"
    )
    parser.add_argument(
        "--do-ahc", default=False, action=ActionYesNo, help="Do AHC on each class"
    )
    parser.add_argument(
        "--cluster-tsne",
        default=False,
        action=ActionYesNo,
        help="if true, clustering is done after TSNE, otherwise after PCA",
    )

    parser.add_argument(
        "--num-clusters",
        default=None,
        type=int,
        help="if not None, number of clusters for AHC, discards ahc-threshold",
    )
    parser.add_argument("--ahc-thr", default=0.7, type=float, help="AHC threshold")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    plot_embedding_tsne(**namespace_to_dict(args))
