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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from scipy import sparse

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.clustering import AHC, KMeans, KMeansInitMethod, SpectralClustering
from hyperion.np.pdfs import SPLDA, DiagGMM, PLDAFactory
from hyperion.np.transforms import PCA, LNorm
from hyperion.utils import SegmentSet
from hyperion.utils.math_funcs import cosine_scoring
from hyperion.utils.misc import PathLike

subcommand_list: List[str] = ["cos_ahc", "spectral_clustering", "cos_ahc_plda_ahc"]


def add_common_args(parser: ArgumentParser) -> None:
    """Add CLI arguments shared by all clustering subcommands."""
    parser.add_argument(
        "--feats-file",
        required=True,
        help="input embedding rspecifier/path",
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        help="input SegmentSet file containing embedding ids",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="output SegmentSet file with assigned cluster ids",
    )
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
        help="verbosity level (0=error, 1=warning, 2=info, 3=debug)",
    )


def load_data(segments_file: PathLike, feats_file: PathLike) -> Tuple[SegmentSet, np.ndarray]:
    """Load segments metadata and corresponding embeddings."""
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    return segments, x


def do_pca(x: np.ndarray, pca_args: Dict[str, Any]) -> np.ndarray:
    """Apply PCA dimensionality reduction if requested by variance ratio."""
    pca_var_r = pca_args["pca_var_r"]
    logging.info("computing pca pca_var_r=%f", pca_var_r)
    if pca_var_r < 1:
        pca = PCA(**pca_args)
        pca.fit(x)
        x = pca(x)
        logging.info("pca-dim=%d", x.shape[1])

    return x


def do_kmeans(
    x: np.ndarray,
    samples_per_cluster: int,
    epochs: int,
    rtol: float,
    init_method: str,
    num_workers: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Optionally run pre-clustering K-means to reduce AHC/SC cost."""
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


def change_precision(x: np.ndarray, precision: Optional[str] = None) -> np.ndarray:
    """Cast score/embedding arrays to requested floating-point precision."""
    if precision == "single":
        return x.astype(np.float32)
    elif precision == "half":
        return x.astype(np.float16)
    else:
        return x


def do_cosine_scoring(x: np.ndarray, precision: Optional[str] = None) -> np.ndarray:
    """Compute pairwise cosine scores."""
    logging.info("compute cosine affinity matrix")
    x = change_precision(x)
    return cosine_scoring(x, x)


def train_plda(
    x: np.ndarray,
    y: np.ndarray,
    plda: Dict[str, Any],
    min_samples_per_cluster: int,
    max_samples_per_cluster: Optional[int] = None,
) -> Tuple[LNorm, Any]:
    """Train length-normalization transform and PLDA model from pseudo-labels."""
    logging.info("Train Centering/Whitening + PLDA")
    _, cluster_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    max_samples_per_cluster = (
        np.max(counts) if max_samples_per_cluster is None else max_samples_per_cluster
    )
    transforms = LNorm()
    transforms.fit(x)
    if plda["y_dim"] > x.shape[1]:
        plda["y_dim"] = x.shape[1]
    plda_model = PLDAFactory.create(**plda)

    counts = counts[cluster_idx]
    keep = np.logical_and(
        counts >= min_samples_per_cluster, counts <= max_samples_per_cluster
    )
    x = x[keep]
    cluster_idx = cluster_idx[keep]
    _, cluster_idx = np.unique(cluster_idx, return_inverse=True)
    plda_model.fit(x, class_ids=cluster_idx)

    return transforms, plda_model


def do_ahc(
    scores: np.ndarray,
    linkage_method: str,
    stop_criterion: str,
    threshold: float,
    num_clusters: Optional[int],
) -> np.ndarray:
    """Run agglomerative hierarchical clustering and return flat cluster labels."""
    logging.info(
        f"running AHC stop_criterion: {stop_criterion} thr: {threshold} num_clusters: {num_clusters}",
    )
    ahc = AHC(method=linkage_method)
    ahc.fit(scores)
    if stop_criterion == "threshold":
        y = ahc.get_flat_clusters_from_thr(threshold)
    else:
        y = ahc.get_flat_clusters_from_num_clusters(num_clusters)

    return y


def get_gmm_post(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate posterior confidence for assigned and second-best clusters."""
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
    p_2nd = np.sort(p, axis=1, kind="heapsort")[:, -2]
    return p_max, p_2nd


def plot_score_hist(scores: np.ndarray, fig_file: PathLike) -> None:
    """Plot histogram of pairwise scores and save to disk."""
    mask = np.triu(np.ones_like(scores, dtype=bool))
    fig = plt.figure()
    scores = scores[mask]
    logging.info(
        f"score-mean=%f score-std=%f score-max=%f score-min=%f",
        scores.mean(),
        scores.std(),
        scores.max(),
        scores.min(),
    )
    if np.any(scores < -1.1) or np.any(scores > 1.1):
        # if scores come from plda we limit the max and min val
        thr = 2 * np.std(scores)
        scores = scores.copy()
        scores[scores > thr] = thr
        scores[scores < -thr] = -thr

    plt.hist(scores, bins=100, density=True)
    fig.savefig(fig_file)


def plot_cluster_size_hist(y: np.ndarray, fig_file: PathLike) -> None:
    """Plot cluster-size histogram and save to disk."""
    _, counts = np.unique(y, return_counts=True)
    fig = plt.figure()
    bins = np.arange(1, np.max(counts) + 1)
    plt.hist(counts, bins=bins, density=False)
    fig.savefig(fig_file)


def cos_ahc(
    segments_file: PathLike,
    feats_file: PathLike,
    output_file: PathLike,
    lnorm: bool,
    pca: Dict[str, Any],
    linkage_method: str,
    stop_criterion: str,
    num_clusters: Optional[int],
    threshold: float,
    ahc_precision: str,
    pre_kmeans: Dict[str, Any],
    num_workers: int,
    filter_by_gmm_post: float,
) -> None:
    """Cluster embeddings using cosine scoring followed by AHC."""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    segments, x = load_data(segments_file, feats_file)
    if lnorm:
        x = LNorm()(x)

    x = do_pca(x, pca)
    x_km, idx_km = do_kmeans(x, num_workers=num_workers, **pre_kmeans)
    scores = do_cosine_scoring(x_km, ahc_precision)
    fig_file = Path(output_file).parent / "score_hist.png"
    plot_score_hist(scores, fig_file)
    y = do_ahc(scores, linkage_method, stop_criterion, threshold, num_clusters)
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
    fig_file = Path(output_file).parent / "cluster_size_hist.png"
    plot_cluster_size_hist(segments["cluster"], fig_file)


def make_cos_ahc_parser() -> ArgumentParser:
    """Build CLI parser for cosine+AHC clustering."""
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    add_common_args(parser)
    parser.add_argument(
        "--lnorm", default=False, action=ActionYesNo, help="apply length normalization"
    )
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
    parser.add_argument(
        "--threshold",
        default=0,
        type=float,
        help="AHC stopping threshold when --stop-criterion=threshold",
    )
    parser.add_argument(
        "--ahc-precision",
        default="single",
        choices=["half", "single", "double"],
        help="precision used to compute the AHC score matrix",
    )
    parser.add_argument(
        "--pre_kmeans.samples-per-cluster",
        default=1,
        type=int,
        help="run pre-kmeans with roughly this many samples per cluster",
    )
    parser.add_argument(
        "--pre_kmeans.init_method",
        default=KMeansInitMethod.max_dist,
        choices=KMeansInitMethod.choices(),
        help="initialization method for pre-kmeans",
    )
    parser.add_argument(
        "--pre_kmeans.epochs", default=100, type=int, help="maximum pre-kmeans epochs"
    )
    parser.add_argument(
        "--pre_kmeans.rtol",
        default=0.001,
        type=float,
        help="relative convergence tolerance for pre-kmeans",
    )
    parser.add_argument(
        "--num-workers", default=1, type=int, help="number of workers for pre-kmeans"
    )
    return parser


def cos_ahc_plda_ahc(
    segments_file: PathLike,
    feats_file: PathLike,
    output_file: PathLike,
    lnorm: bool,
    pca: Dict[str, Any],
    linkage_method: str,
    stop_criterion: str,
    num_clusters_stage_1: Optional[int],
    threshold_stage_1: float,
    num_clusters_stage_2: Optional[int],
    threshold_stage_2: float,
    min_samples_per_cluster: int,
    max_samples_per_cluster: Optional[int],
    plda: Dict[str, Any],
    ahc_precision: str,
    pre_kmeans: Dict[str, Any],
    num_workers: int,
    filter_by_gmm_post: float,
) -> None:
    """Cluster embeddings with cosine-AHC, then PLDA scoring and a second AHC pass."""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    segments, x = load_data(segments_file, feats_file)
    if lnorm:
        x = LNorm()(x)

    x = do_pca(x, pca)

    # stage 1
    x_km, idx_km = do_kmeans(x, num_workers=num_workers, **pre_kmeans)
    scores = do_cosine_scoring(x_km, ahc_precision)
    fig_file = Path(output_file).parent / "cosine_score_hist.png"
    plot_score_hist(scores, fig_file)
    y = do_ahc(
        scores, linkage_method, stop_criterion, threshold_stage_1, num_clusters_stage_1
    )
    if idx_km is not None:
        y = y[idx_km]
        del x_km

    fig_file = Path(output_file).parent / "cosine_cluster_size_hist.png"
    plot_cluster_size_hist(y, fig_file)
    # stage 2
    transform, plda_model = train_plda(
        x, y, plda, min_samples_per_cluster, max_samples_per_cluster
    )
    x = transform(x)
    z = plda_model.compute_py_g_x(x)
    _, idx_km = do_kmeans(z, num_workers=num_workers, **pre_kmeans)

    if idx_km is None:
        scores = plda_model.llr_1vs1(x, x)
    else:
        scores = plda_model.llr_NvsM(x, x, ids1=idx_km, ids2=idx_km)

    scores = change_precision(scores, ahc_precision)
    fig_file = Path(output_file).parent / "plda_score_hist.png"
    plot_score_hist(scores, fig_file)
    y = do_ahc(
        scores, linkage_method, stop_criterion, threshold_stage_2, num_clusters_stage_2
    )
    if idx_km is not None:
        y = y[idx_km]

    p_max, p_2nd = get_gmm_post(x, y)
    segments["cluster"] = y
    segments["post_cluster"] = p_max
    segments["post_cluster_2nd"] = p_2nd
    if filter_by_gmm_post > 0:
        idx = segments["post_cluster"] > filter_by_gmm_post
        segments = SegmentSet(segments.loc[idx])

    segments.save(output_file)
    fig_file = Path(output_file).parent / "plda_cluster_size_hist.png"
    plot_cluster_size_hist(segments["cluster"], fig_file)


def make_cos_ahc_plda_ahc_parser() -> ArgumentParser:
    """Build CLI parser for two-stage cosine/PLDA AHC clustering."""
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    add_common_args(parser)
    parser.add_argument(
        "--lnorm", default=False, action=ActionYesNo, help="apply length normalization"
    )
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
        "--num-clusters-stage-1",
        default=None,
        type=int,
        help="number of AHC clusters for first stage",
    )
    parser.add_argument(
        "--threshold-stage-1",
        default=0,
        type=float,
        help="stopping threshold for first stage",
    )
    parser.add_argument(
        "--num-clusters-stage-2",
        default=None,
        type=int,
        help="number of AHC clusters for second stage",
    )
    parser.add_argument(
        "--threshold-stage-2",
        default=0,
        type=float,
        help="stopping threshold for second stage",
    )
    parser.add_argument(
        "--ahc-precision",
        default="single",
        choices=["half", "single", "double"],
        help="precision used to compute AHC score matrices",
    )
    parser.add_argument(
        "--min-samples-per-cluster",
        default=8,
        type=int,
        help="minimum samples/cluster for a cluster to be used to train PLDA",
    )
    parser.add_argument(
        "--max-samples-per-cluster",
        default=50,
        type=int,
        help="maximum samples/cluster for a cluster to be used to train PLDA",
    )
    PLDAFactory.add_class_args(parser, prefix="plda")
    parser.add_argument(
        "--pre_kmeans.samples-per-cluster",
        default=1,
        type=int,
        help="run pre-kmeans with roughly this many samples per cluster",
    )
    parser.add_argument(
        "--pre_kmeans.init_method",
        default=KMeansInitMethod.max_dist,
        choices=KMeansInitMethod.choices(),
        help="initialization method for pre-kmeans",
    )
    parser.add_argument(
        "--pre_kmeans.epochs", default=100, type=int, help="maximum pre-kmeans epochs"
    )
    parser.add_argument(
        "--pre_kmeans.rtol",
        default=0.001,
        type=float,
        help="relative convergence tolerance for pre-kmeans",
    )
    parser.add_argument(
        "--num-workers", default=1, type=int, help="number of workers for pre-kmeans"
    )
    return parser


def compute_sc_affinity(
    x: np.ndarray,
    aff_func: str,
    gauss_sigma: float,
    aff_thr: float,
    precision: str,
) -> Union[np.ndarray, sparse.csr_matrix]:
    """Compute and sparsify the affinity matrix used by spectral clustering."""
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
    segments_file: PathLike,
    feats_file: PathLike,
    output_file: PathLike,
    lnorm: bool,
    pca: Dict[str, Any],
    pre_kmeans: Dict[str, Any],
    affinity: Dict[str, Any],
    spectral_clustering: Dict[str, Any],
    filter_by_gmm_post: float,
) -> None:
    """Cluster embeddings using spectral clustering over a cosine-based affinity."""
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
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
    fig_file = Path(output_file).parent / "cluster_size_hist.png"
    plot_cluster_size_hist(segments["cluster"], fig_file)

    fig_file = output_file.with_stem(output_file.stem + "_eigengap").with_suffix(".png")
    sc.plot_eigengap_stats(eigengap_stats, num_clusters, fig_file)

    df_eig = pd.DataFrame(
        {k: eigengap_stats[k] for k in ["eig_vals", "eigengap", "d_eig_vals"]}
    )
    df_eig["num_clusters"] = np.arange(1, len(df_eig) + 1)
    eig_file = fig_file.with_suffix(".csv")
    df_eig.to_csv(eig_file, index=False)


def make_spectral_clustering_parser() -> ArgumentParser:
    """Build CLI parser for spectral clustering."""
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")
    add_common_args(parser)
    parser.add_argument(
        "--lnorm", default=False, action=ActionYesNo, help="apply length normalization"
    )
    PCA.add_class_args(parser, prefix="pca")
    parser.add_argument(
        "--pre_kmeans.samples-per-cluster",
        default=1,
        type=int,
        help="run pre-kmeans with roughly this many samples per cluster",
    )
    parser.add_argument(
        "--pre_kmeans.init_method",
        default=KMeansInitMethod.max_dist,
        choices=KMeansInitMethod.choices(),
        help="initialization method for pre-kmeans",
    )
    parser.add_argument(
        "--pre_kmeans.epochs", default=100, type=int, help="maximum pre-kmeans epochs"
    )
    parser.add_argument(
        "--pre_kmeans.rtol",
        default=0.001,
        type=float,
        help="relative convergence tolerance for pre-kmeans",
    )
    parser.add_argument(
        "--pre_kmeans.num_workers",
        default=1,
        type=int,
        help="number of workers for pre-kmeans",
    )
    parser.add_argument(
        "--affinity.aff_func",
        default="cos",
        choices=["cos", "gauss_cos"],
        help="affinity function used to build graph edges",
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
        help="affinity values below this threshold are set to 0",
    )
    parser.add_argument(
        "--affinity.precision",
        default="single",
        choices=["half", "single", "double"],
        help="precision used to compute the affinity matrix",
    )
    SpectralClustering.add_class_args(parser, prefix="spectral_clustering")

    return parser


def main() -> None:
    """Parse CLI arguments and run the selected clustering pipeline."""
    parser = ArgumentParser(
        description="Cluster embeddings into classes, usually speakers"
    )
    parser.add_argument("--cfg", action=ActionConfigFile, help="configuration file")

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
