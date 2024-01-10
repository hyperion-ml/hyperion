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

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from scipy import sparse

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.clustering import AHC, KMeans, KMeansInitMethod, SpectralClustering
from hyperion.np.pdfs import DiagGMM
from hyperion.np.transforms import PCA, LNorm
from hyperion.utils import SegmentSet
from hyperion.utils.math_funcs import cosine_scoring

subcommand_list = [
    "cos_ahc",
    "spectral_clustering",
]


def add_common_args(parser):
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--filter-by-gmm-post",
        default=0,
        type=float,
        help="remove segments with gmm posterior lower than threshold",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def load_data(segments_file, feats_file):
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    return segments, x


def do_pca(x, pca_args):
    pca_var_r = pca_args["pca_var_r"]
    logging.info("computing pca pca_var_r=%f", pca_var_r)
    if pca_var_r < 1:
        pca = PCA(**pca_args)
        pca.fit(x)
        x = pca(x)
        logging.info("pca-dim=%d", x.shape[1])

    return x


def do_kmeans(x, samples_per_cluster, epochs, rtol, init_method, num_workers):
    if samples_per_cluster > 1:
        km_clusters = x.shape[0] // samples_per_cluster
        logging.info("kmeans with num_clusters=%d", km_clusters)
        kmeans = KMeans(
            num_clusters=km_clusters,
            rtol=rtol,
            epochs=epochs,
            init_method=init_method,
            num_workers=num_workers,
        )
        kmeans.fit(x)
        idx_km, _ = kmeans(x)
        x_km = kmeans.mu
        del kmeans
    else:
        idx_km = None
        x_km = x

    return x_km, idx_km


def get_gmm_post(x, y):
    logging.info("computing cluster posteriors with gmm")
    num_comp = np.max(y) + 1
    gmm = DiagGMM(num_comp=num_comp, x_dim=x.shape[1], min_N=1)
    u_dim = gmm.compute_suff_stats(x[:1]).shape[1]
    N = np.zeros((num_comp,), dtype=float) + 1e-5
    u_x = np.zeros((num_comp, u_dim), dtype=float)

    for c in range(num_comp):
        mask = y == c
        N_c = np.sum(mask)
        if N_c == 0:
            continue

        N[c] = N_c
        u_x_c = gmm.compute_suff_stats(x[mask])
        u_x[c] = np.sum(u_x_c, axis=0)

    gmm.Mstep(N, u_x)
    p = gmm.compute_pz(x, mode="std")
    p_max = p[np.arange(x.shape[0]), y]
    zz = p_max < 0.5
    print(np.mean(p[zz]), np.max(p[zz]), p_max[zz])
    p_2nd = np.sort(p, axis=1, kind="heapsort")[:, -2]
    return p_max, p_2nd


def cos_ahc(
    segments_file,
    feats_file,
    output_file,
    lnorm,
    pca,
    linkage_method,
    stop_criterion,
    num_clusters,
    threshold,
    ahc_precision,
    pre_kmeans,
    num_workers,
    filter_by_gmm_post,
):
    segments, x = load_data(segments_file, feats_file)
    if lnorm:
        x = LNorm()(x)

    x = do_pca(x, pca)
    x_km, idx_km = do_kmeans(x, num_workers=num_workers, **pre_kmeans)

    logging.info("compute affinity matrix")
    if ahc_precision == "single":
        x_lowprec = x_km.astype(np.float32)
    elif ahc_precision == "half":
        x_lowprec = x_km.astype(np.float16)
    else:
        x_lowprec = x_km

    scores = cosine_scoring(x_lowprec, x_lowprec)

    logging.info("running AHC")
    ahc = AHC(method=linkage_method)
    ahc.fit(scores)
    if stop_criterion == "threshold":
        y = ahc.get_flat_clusters_from_thr(threshold)
    else:
        y = ahc.get_flat_clusters_from_num_clusters(num_clusters)

    del ahc
    if idx_km is not None:
        y = y[idx_km]
        del x_km

    p_max, p_2nd = get_gmm_post(x, y)
    segments["cluster"] = y
    segments["post_cluster"] = p_max
    segments["post_cluster_2nd"] = p_2nd
    if filter_by_gmm_post > 0:
        idx = segments["post_cluster"] > filter_by_gmm_post
        segments = SegmentSet(segments.loc[idx])

    segments.save(output_file)


def make_cos_ahc_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    add_common_args(parser)
    parser.add_argument("--lnorm", default=False, action=ActionYesNo)
    PCA.add_class_args(parser, prefix="pca")
    parser.add_argument(
        "--linkage-method",
        default="average",
        choices=["single", "complete", "average", "weighted", "ward"],
        help="linkage method",
    )
    parser.add_argument(
        "--stop-criterion",
        default="threshold",
        choices=["threshold", "num_clusters"],
        help="stopping criterion",
    )
    parser.add_argument(
        "--num-clusters", default=None, type=int, help="number of AHC clusters"
    )
    parser.add_argument("--threshold", default=0, type=float, help="stopping threshold")
    parser.add_argument(
        "--ahc-precision", default="single", choices=["half", "single", "double"]
    )
    parser.add_argument(
        "--pre_kmeans.samples-per-cluster",
        default=1,
        type=int,
        help="first k-means is done to recuce the computing cost of AHC",
    )
    parser.add_argument(
        "--pre_kmeans.init_method",
        default=KMeansInitMethod.max_dist,
        choices=KMeansInitMethod.choices(),
    )
    parser.add_argument("--pre_kmeans.epochs", default=100, type=int)
    parser.add_argument("--pre_kmeans.rtol", default=0.001, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
    return parser


def compute_sc_affinity(x, aff_func, gauss_sigma, aff_thr, precision):
    if precision == "single":
        x = x.astype(np.float32)
    elif precision == "half":
        x = x.astype(np.float16)

    scores = cosine_scoring(x, x)
    if aff_func == "gauss_cos":
        assert gauss_sigma > 0
        d2 = 1 - scores
        scores = np.exp(-d2 / gauss_sigma)

    assert aff_thr < 1
    scores[scores < aff_thr] = 0
    num_nodes = scores.shape[0]
    scores.flat[:: num_nodes + 1] = 0
    aff_size = num_nodes**2
    num_edges = np.sum(scores > 0)
    r = aff_size / num_edges
    logging.info("num_nodes^2=%d, num_edges=%d r=%f", aff_size, num_edges, r)
    if r > 4:
        scores = sparse.csr_matrix(scores)
    return scores


def spectral_clustering(
    segments_file,
    feats_file,
    output_file,
    lnorm,
    pca,
    pre_kmeans,
    affinity,
    spectral_clustering,
    filter_by_gmm_post,
):
    segments, x = load_data(segments_file, feats_file)
    if lnorm:
        x = LNorm()(x)

    x = do_pca(x, pca)
    x_km, idx_km = do_kmeans(x, **pre_kmeans)
    A = compute_sc_affinity(x_km, **affinity)
    sc = SpectralClustering(**spectral_clustering)
    y, num_clusters, eigengap_stats = sc.fit(A)
    if idx_km is not None:
        y = y[idx_km]
        del x_km

    segments["cluster"] = y
    if num_clusters > 1:
        p_max, p_2nd = get_gmm_post(x, y)
        segments["post_cluster"] = p_max
        segments["post_cluster_2nd"] = p_2nd

    if filter_by_gmm_post > 0:
        idx = segments["post_cluster"] > filter_by_gmm_post
        segments = SegmentSet(segments.loc[idx])

    segments.save(output_file)
    output_file = Path(output_file)
    fig_file = output_file.with_stem(output_file.stem + "_eigengap").with_suffix(".png")
    sc.plot_eigengap_stats(eigengap_stats, num_clusters, fig_file)

    df_eig = pd.DataFrame(
        {k: eigengap_stats[k] for k in ["eig_vals", "eigengap", "d_eig_vals"]}
    )
    df_eig["num_clusters"] = np.arange(1, len(df_eig) + 1)
    eig_file = fig_file.with_suffix(".csv")
    df_eig.to_csv(eig_file, index=False)


def make_spectral_clustering_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    add_common_args(parser)
    parser.add_argument("--lnorm", default=False, action=ActionYesNo)
    PCA.add_class_args(parser, prefix="pca")
    parser.add_argument(
        "--pre_kmeans.samples-per-cluster",
        default=1,
        type=int,
        help="first k-means is done to recuce the computing cost of AHC",
    )
    parser.add_argument(
        "--pre_kmeans.init_method",
        default=KMeansInitMethod.max_dist,
        choices=KMeansInitMethod.choices(),
    )
    parser.add_argument("--pre_kmeans.epochs", default=100, type=int)
    parser.add_argument("--pre_kmeans.rtol", default=0.001, type=float)
    parser.add_argument("--pre_kmeans.num_workers", default=1, type=int)
    parser.add_argument(
        "--affinity.aff_func", default="cos", choices=["cos", "gauss_cos"]
    )
    parser.add_argument(
        "--affinity.gauss-sigma",
        default=1,
        type=float,
        help="std. dev. of gauss function",
    )
    parser.add_argument(
        "--affinity.aff-thr",
        default=0,
        type=float,
        help="values under this are set to 0",
    )
    parser.add_argument(
        "--affinity.precision", default="single", choices=["half", "single", "double"]
    )
    SpectralClustering.add_class_args(parser, prefix="spectral_clustering")

    return parser


def main():
    parser = ArgumentParser(
        description="Cluster embeddings into classes, usually speakers"
    )
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
