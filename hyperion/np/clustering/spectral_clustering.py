"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import copy
from enum import Enum
from typing import Any, Dict, Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.preprocessing import normalize

from ...hyp_defs import float_cpu
from ...utils import PathLike
from ..np_model import NPModel
from .kmeans import KMeans, KMeansInitMethod


class LaplacianType(str, Enum):
    unnormalized = "unnormalized"
    norm_sym = "norm_sym"
    norm_rw = "norm_rw"

    @staticmethod
    def choices():
        return [
            LaplacianType.unnormalized,
            LaplacianType.norm_sym,
            LaplacianType.norm_rw,
        ]


class SpectralClusteringNumClassCriterion(str, Enum):
    max_eigengap = "max_eigengap"
    max_d_eig_vals = "max_d_eig_vals"
    thr_eigengap = "thr_eigengap"
    thr_d_eig_vals = "thr_d_eig_vals"

    @staticmethod
    def choices():
        return [
            SpectralClusteringNumClassCriterion.max_eigengap,
            SpectralClusteringNumClassCriterion.max_d_eig_vals,
            SpectralClusteringNumClassCriterion.thr_eigengap,
            SpectralClusteringNumClassCriterion.thr_d_eig_vals,
        ]


class SpectralClustering(NPModel):
    """Spectral Clustering class"""

    def __init__(
        self,
        laplacian: str = "norm_sym",
        num_clusters: Optional[int] = None,
        max_num_clusters: Optional[int] = None,
        criterion: SpectralClusteringNumClassCriterion = SpectralClusteringNumClassCriterion.max_eigengap,
        thr_eigengap: float = 1e-3,
        kmeans_epochs: int = 100,
        kmeans_init_method: KMeansInitMethod = KMeansInitMethod.max_dist,
        num_workers: int = 1,
    ):
        self.laplacian = laplacian
        self.num_clusters = num_clusters
        self.max_num_clusters = max_num_clusters
        self.criterion = criterion
        self.kmeans_epochs = kmeans_epochs
        self.thr_eigengap = thr_eigengap
        self.kmeans_init_method = kmeans_init_method
        self.num_workers = num_workers

    def spectral_embedding(self, x: np.ndarray):
        num_nodes = x.shape[0]
        if not sparse.issparse(x):
            x.flat[:: num_nodes + 1] = 0
            r = num_nodes**2 / np.sum(x > 0)
            if r > 4:
                x = sparse.csr_matrix(x)

        D = None
        if self.laplacian in LaplacianType.unnormalized:
            L = csgraph_laplacian(x, normed=False)
        elif self.laplacian == LaplacianType.norm_sym:
            L = csgraph_laplacian(x, normed=True)
        elif self.laplacian == LaplacianType.norm_rw:
            L, dd = csgraph_laplacian(x, normed=False, return_diag=True)
            if sparse.issparse(L):
                D = sparse.diags(dd)
            else:
                D = np.diag(dd)

        max_num_clusters = num_nodes - 1
        if self.max_num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.max_num_clusters)
        if self.num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.num_clusters)

        eig_vals, eig_vecs = eigsh(L, k=max_num_clusters, M=D, which="SM")
        eig_vals = eig_vals[1:]
        eig_vecs = eig_vecs[:, 1:]
        return eig_vals, eig_vecs

    def spectral_embedding_0(self, x: np.ndarray):
        num_nodes = x.shape[0]
        x.flat[:: num_nodes + 1] = 0
        d = np.sum(x, axis=1)
        D = None
        if self.laplacian in LaplacianType.unnormalized:
            L = np.diag(d) - x
        elif self.laplacian == LaplacianType.norm_sym:
            idsqrt = 1 / np.sqrt(d)
            L = np.eye(num_nodes) - idsqrt[:, None] * x * idsqrt
        elif self.laplacian == LaplacianType.norm_rw:
            D = np.diag(d)
            L = D - x

        max_num_clusters = num_nodes
        if self.max_num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.max_num_clusters)
        if self.num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.num_clusters)

        eig_vals, eig_vecs = eigh(
            L, b=D, overwrite_a=True, subset_by_index=[1, max_num_clusters - 1]
        )

        return eig_vals, eig_vecs

    def compute_eigengap(self, eig_vals: np.ndarray):
        eig_vals = np.concatenate(([0.0], eig_vals))
        eigengap = np.diff(np.concatenate(([0.0], eig_vals)))
        filter = np.array([1 / 60, -3 / 20, 3 / 4, 0.0, -3 / 4, 3 / 20, -1 / 60])
        eig_vals_ext = np.concatenate((eig_vals, [eig_vals[-1]] * 3))
        d_eig_vals = np.convolve(eig_vals, filter)[3:-6]
        k_max = np.argmax(eigengap)
        gap_max = eigengap[k_max]
        # k_relmax = []
        # gap_relmax = []
        # gap_norm_relmax = []
        # for k in range(len(eigengap)):
        #     if k == 0 and eigengap[k] > eigengap[k + 1]:
        #         k_relmax.append(k)
        #         gap_relmax.append(eigengap[k])
        #         gap_norm_relmax.append(eigengap[k] / eigengap[k + 1])
        #     elif k == len(eigengap) - 1 and eigengap[k] > eigengap[k - 1]:
        #         k_relmax.append(k)
        #         gap_relmax.append(eigengap[k])
        #         gap_norm_relmax.append(eigengap[k] / eigengap[k - 1])
        #     elif eigengap[k] > eigengap[k - 1] and eigengap[k] > eigengap[k + 1]:
        #         k_relmax.append(k)
        #         gap_relmax.append(eigengap[k])
        #         gap_norm_relmax.append(
        #             2 * eigengap[k] / (eigengap[k - 1] + eigengap[k + 1])
        #         )

        # idx = np.argmax(gap_norm_relmax)
        # gap_norm_relmax_max = gap_norm_relmax[idx]
        # k_relmax_max = k_relmax[idx]
        eigengap_stats = {
            "eig_vals": eig_vals,
            "eigengap": eigengap,
            "gap_max": gap_max,
            "k_max": k_max,
            # "gap_relmax": gap_relmax,
            # "k_relmax": k_relmax,
            # "gap_norm_relmax": gap_norm_relmax,
            # "gap_norm_relmax_max": gap_norm_relmax_max,
            # "k_relmax_max": k_relmax_max,
            "d_eig_vals": d_eig_vals,
        }
        return eigengap_stats

    def predict_num_clusters(self, eigengap_stats: np.ndarray):
        if self.num_clusters is not None:
            num_clusters = self.num_clusters

        elif self.criterion == SpectralClusteringNumClassCriterion.max_eigengap:
            num_clusters = eigengap_stats["k_max"] + 1
        elif self.criterion == SpectralClusteringNumClassCriterion.max_d_eig_vals:
            num_clusters = np.argmax(eigengap_stats["d_eig_vals"]) + 1
        elif self.criterion == SpectralClusteringNumClassCriterion.thr_eigengap:
            nz = (eigengap_stats["eigengap"] < self.thr_eigengap).nonzero()[0]
            num_clusters = nz[nz > eigengap_stats["k_max"]][0] + 1
        elif self.criterion == SpectralClusteringNumClassCriterion.thr_d_eig_vals:
            nz = (eigengap_stats["d_eig_vals"] < self.thr_eigengap).nonzero()[0]
            num_clusters = nz[nz > eigengap_stats["k_max"]][0] + 1
        else:
            raise ValueError(f"invalid num clusters criterion {self.criterion}")
        return num_clusters

    def normalize_eigvecs(self, eig_vecs: np.ndarray):
        if self.laplacian == LaplacianType.norm_sym:
            return normalize(eig_vecs, axis=1)
        else:
            return eig_vecs

    def do_kmeans(self, x: np.ndarray, num_clusters: Optional[int] = None):
        if num_clusters is None:
            num_clusters = x.shape[1] + 1
        kmeans = KMeans(
            num_clusters=num_clusters,
            epochs=self.kmeans_epochs,
            init_method=self.kmeans_init_method,
            num_workers=self.num_workers,
        )
        kmeans.fit(x)
        y, _ = kmeans(x)
        return y

    def fit(self, x: np.ndarray):
        logging.info("compute spectral embeddings")

        eig_vals, eig_vecs = self.spectral_embedding(x)
        if self.num_clusters is None:
            logging.info("compute eigengap stats")
            eigengap_stats = self.compute_eigengap(eig_vals)
        else:
            eigengap_stats = None

        logging.info("predicting number of clusters")
        num_clusters = self.predict_num_clusters(eigengap_stats)
        logging.info("predicted num_clusters=%d", num_clusters)
        if num_clusters == 1:
            return np.zeros((x.shape[0]), dtype=int), num_clusters, eigengap_stats
        # minus one because we already removed the first eig vector
        logging.info("normalizing embeddings")
        eig_vecs = eig_vecs[:, : num_clusters - 1]
        eig_vecs = self.normalize_eigvecs(eig_vecs)
        logging.info("running k-means")
        y = self.do_kmeans(eig_vecs, num_clusters)
        return y, num_clusters, eigengap_stats

    def plot_eigengap_stats(
        self,
        eigengap_stats: Dict[str, Any],
        num_clusters: int,
        fig_file: Optional[PathLike] = None,
    ):
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows=1, ncols=3, sharex=True, figsize=(12, 6)
        )
        eig_vals = eigengap_stats["eig_vals"]
        ax0.plot(np.arange(1, len(eig_vals) + 1), eig_vals, "b")
        ax0.vlines(
            num_clusters, ymin=np.min(eig_vals), ymax=np.max(eig_vals), colors="r"
        )
        ax0.grid()
        ax0.set_title("eigen_vals")
        eigengap = eigengap_stats["eigengap"]
        ax1.plot(np.arange(1, len(eigengap) + 1), eigengap, "b")
        ax1.vlines(
            num_clusters, ymin=np.min(eigengap), ymax=np.max(eigengap), colors="r"
        )
        ax1.grid()
        ax1.set_title("eigengap")
        d_eig_vals = eigengap_stats["d_eig_vals"]
        ax2.plot(np.arange(1, len(d_eig_vals) + 1), d_eig_vals, "b")
        ax2.vlines(
            num_clusters, ymin=np.min(d_eig_vals), ymax=np.max(d_eig_vals), colors="r"
        )
        ax2.grid()
        ax2.set_title("d_eigen_val")
        if fig_file is not None:
            fig.savefig(fig_file)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """It adds the arguments corresponding to the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--laplacian",
            default=LaplacianType.norm_sym,
            choices=LaplacianType.choices(),
        )
        parser.add_argument("--num-clusters", default=None, type=int)
        parser.add_argument("--max-num-clusters", default=None, type=int)
        parser.add_argument(
            "--criterion",
            default=SpectralClusteringNumClassCriterion.max_eigengap,
            choices=SpectralClusteringNumClassCriterion.choices(),
        )
        parser.add_argument("--thr-eigengap", default=1e-3, type=float)
        parser.add_argument("--kmeans-epochs", default=100, type=int)
        parser.add_argument(
            "--kmeans-init-method",
            default=KMeansInitMethod.max_dist,
            choices=KMeansInitMethod.choices(),
        )
        parser.add_argument("--num-workers", default=1, type=int)

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )
