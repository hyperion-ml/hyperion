#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.clustering import AHC
from hyperion.np.transforms import PCA, LNorm, SklTSNE
from hyperion.utils import PathLike, SegmentSet
from hyperion.utils.math_funcs import cosine_scoring

matplotlib.use("Agg")
colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["x", "o", "+", "*", "s", "h", "D", "^", "v", "p", "8"]

color_marker = [(c, m) for m in markers for c in colors]


def plot_embedding_tsne(
    train_v_file: PathLike,
    train_list: PathLike,
    pca_var_r: float,
    prob_plot: float,
    lnorm: bool,
    title: str,
    max_classes: Optional[int],
    plot_class_name: str,
    do_ahc: bool,
    cluster_tsne: bool,
    num_clusters: Optional[int],
    ahc_thr: float,
    output_dir: PathLike,
    **kwargs: Any,
) -> None:
    """Project embeddings with t-SNE and save one plot per class.

    Args:
        train_v_file: Input embeddings rspecifier/file.
        train_list: Segment list file with sample ids and class labels.
        pca_var_r: Target explained-variance ratio for PCA (skip if ``>=1``).
        prob_plot: Probability of keeping each point for plotting.
        lnorm: If ``True``, apply length normalization to embeddings.
        title: Prefix title used on saved figures.
        max_classes: Optional maximum number of classes to process.
        plot_class_name: Segment column containing class labels.
        do_ahc: If ``True``, run AHC clustering inside each class.
        cluster_tsne: If ``True``, cluster in t-SNE space; otherwise in PCA space.
        num_clusters: Optional fixed number of AHC clusters per class.
        ahc_thr: AHC threshold when ``num_clusters`` is not set.
        output_dir: Output directory for generated figures and optional segments file.
        **kwargs: Extra parsed arguments, including ``tsne`` sub-configuration.
    """
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
                x2 = np.sum(x_tsne**2, axis=1)[:, None]
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


def main() -> None:
    """Parse CLI arguments and plot per-class t-SNE embeddings."""
    parser = ArgumentParser(
        description=(
            "Projects embeddings using TSNE, "
            "plots a TSNE per class to discover subclusters inside of the classes"
        )
    )

    parser.add_argument(
        "--train-v-file",
        required=True,
        help="Input embeddings rspecifier/file.",
    )
    parser.add_argument(
        "--train-list",
        required=True,
        help="Segments file with ids and class labels.",
    )

    parser.add_argument(
        "--pca-var-r",
        default=0.95,
        type=float,
        help="PCA explained-variance ratio; set >=1 to disable PCA.",
    )
    parser.add_argument(
        "--prob-plot",
        default=0.1,
        type=float,
        help="Probability of plotting each sample point.",
    )
    parser.add_argument(
        "--lnorm",
        default=False,
        action=ActionYesNo,
        help="Apply length normalization before PCA/t-SNE.",
    )
    parser.add_argument(
        "--plot-class-name",
        default="class_id",
        help="Name of the class-label column used for per-class plotting.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title prefix added to each class plot.",
    )
    SklTSNE.add_class_args(parser, prefix="tsne")

    parser.add_argument(
        "--max-classes", default=None, type=int, help="Maximum number of classes to plot."
    )
    parser.add_argument(
        "--do-ahc",
        default=False,
        action=ActionYesNo,
        help="Run AHC clustering within each class.",
    )
    parser.add_argument(
        "--cluster-tsne",
        default=False,
        action=ActionYesNo,
        help="If true, cluster in t-SNE space; otherwise in PCA space.",
    )

    parser.add_argument(
        "--num-clusters",
        default=None,
        type=int,
        help="If set, fixed number of AHC clusters (overrides --ahc-thr).",
    )
    parser.add_argument("--ahc-thr", default=0.7, type=float, help="AHC threshold")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated t-SNE plots.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    plot_embedding_tsne(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
