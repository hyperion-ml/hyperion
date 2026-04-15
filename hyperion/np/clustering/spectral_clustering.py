"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import ActionParser, ArgumentParser
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize

from ...utils import PathLike
from ..hyper_np_model import HyperNPModel
from .kmeans import KMeans, KMeansInitMethod


class LaplacianType(str, Enum):
    unnormalized = "unnormalized"
    norm_sym = "norm_sym"
    norm_rw = "norm_rw"

    @staticmethod
    def choices() -> list["LaplacianType"]:
        """Gets supported laplacian choices.

        Returns:
          List of supported laplacian enum values.
        """
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
    def choices() -> list["SpectralClusteringNumClassCriterion"]:
        """Gets supported automatic cluster-count criteria.

        Returns:
          List of supported criterion enum values.
        """
        return [
            SpectralClusteringNumClassCriterion.max_eigengap,
            SpectralClusteringNumClassCriterion.max_d_eig_vals,
            SpectralClusteringNumClassCriterion.thr_eigengap,
            SpectralClusteringNumClassCriterion.thr_d_eig_vals,
        ]


class SpectralClustering(HyperNPModel):
    """Spectral Clustering class.

    Attributes:
      laplacian: Type of graph Laplacian used to compute the spectral embedding.
      num_clusters: Fixed number of output clusters. If ``None``, the number of
        clusters is estimated from eigenvalue statistics.
      max_num_clusters: Maximum number of clusters/eigenvectors considered while
        estimating the number of clusters.
      criterion: Criterion used to infer the number of clusters.
      thr_eigengap: Threshold used by threshold-based criteria.
      kmeans_epochs: Maximum number of epochs used by k-means in embedding space.
      kmeans_init_method: Initialization method for k-means seeds.
      num_workers: Number of worker threads used by k-means.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.clustering.spectral_clustering import SpectralClustering
      >>> x = np.array([
      ...     [0.0, 0.9, 0.1, 0.0],
      ...     [0.9, 0.0, 0.2, 0.1],
      ...     [0.1, 0.2, 0.0, 0.8],
      ...     [0.0, 0.1, 0.8, 0.0],
      ... ], dtype=np.float32)
      >>> sc = SpectralClustering(num_clusters=2, laplacian="norm_sym")
      >>> y, num_clusters, eigengap_stats = sc.fit(x)
    """

    def __init__(
        self,
        laplacian: Union[LaplacianType, str] = LaplacianType.norm_sym,
        num_clusters: Optional[int] = None,
        max_num_clusters: Optional[int] = None,
        criterion: Union[
            SpectralClusteringNumClassCriterion, str
        ] = SpectralClusteringNumClassCriterion.max_eigengap,
        thr_eigengap: float = 1e-3,
        kmeans_epochs: int = 100,
        kmeans_init_method: Union[KMeansInitMethod, str] = KMeansInitMethod.max_dist,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes a ``SpectralClustering`` model.

        Args:
          laplacian: Graph Laplacian type.
          num_clusters: Fixed number of clusters, or ``None`` to estimate.
          max_num_clusters: Maximum number of clusters considered during
            automatic selection.
          criterion: Criterion used to estimate number of clusters.
          thr_eigengap: Threshold for threshold-based criteria.
          kmeans_epochs: Number of k-means epochs in embedding space.
          kmeans_init_method: K-means initialization method.
          num_workers: Number of threads for k-means.
          **kwargs: Extra arguments forwarded to ``HyperNPModel``.
        """
        super().__init__(**kwargs)
        if num_clusters is not None and (
            not isinstance(num_clusters, int) or num_clusters < 1
        ):
            raise ValueError(
                f"num_clusters must be a positive integer or None, got {num_clusters!r}"
            )
        if max_num_clusters is not None and (
            not isinstance(max_num_clusters, int) or max_num_clusters < 1
        ):
            raise ValueError(
                f"max_num_clusters must be a positive integer or None, got {max_num_clusters!r}"
            )
        if (
            num_clusters is not None
            and max_num_clusters is not None
            and num_clusters > max_num_clusters
        ):
            raise ValueError(
                f"num_clusters ({num_clusters}) cannot be greater than "
                f"max_num_clusters ({max_num_clusters})"
            )
        if kmeans_epochs < 1:
            raise ValueError(f"kmeans_epochs must be >= 1, got {kmeans_epochs!r}")
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers!r}")
        if thr_eigengap < 0:
            raise ValueError(f"thr_eigengap must be >= 0, got {thr_eigengap!r}")

        self.laplacian = LaplacianType(laplacian)
        self.num_clusters = num_clusters
        self.max_num_clusters = max_num_clusters
        self.criterion = SpectralClusteringNumClassCriterion(criterion)
        self.kmeans_epochs = kmeans_epochs
        self.thr_eigengap = thr_eigengap
        self.kmeans_init_method = KMeansInitMethod(kmeans_init_method)
        self.num_workers = num_workers

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "laplacian": self.laplacian.value,
            "num_clusters": self.num_clusters,
            "max_num_clusters": self.max_num_clusters,
            "criterion": self.criterion.value,
            "thr_eigengap": self.thr_eigengap,
            "kmeans_epochs": self.kmeans_epochs,
            "kmeans_init_method": self.kmeans_init_method.value,
            "num_workers": self.num_workers,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def spectral_embedding(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes graph spectral embedding.

        Args:
          x: Affinity/similarity matrix with shape ``(num_nodes, num_nodes)``.

        Returns:
          eig_vals: Eigenvalues associated with the selected embedding vectors.
          eig_vecs: Eigenvectors with shape ``(num_nodes, num_eigenvectors)``.
        """
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError(f"x must be a square matrix, got shape={x.shape}")
        num_nodes = x.shape[0]
        if num_nodes < 2:
            raise ValueError("x must contain at least 2 nodes for spectral embedding")
        if not sparse.issparse(x):
            x.flat[:: num_nodes + 1] = 0
            nnz_pos = np.sum(x > 0)
            r = np.inf if nnz_pos == 0 else num_nodes**2 / nnz_pos
            if r > 4:
                x = sparse.csr_matrix(x)
        else:
            x.setdiag(0)

        D = None
        if self.laplacian == LaplacianType.unnormalized:
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
        if max_num_clusters < 1:
            raise ValueError(
                f"invalid effective max_num_clusters={max_num_clusters}; "
                "check num_clusters/max_num_clusters configuration"
            )

        eig_vals, eig_vecs = eigsh(L, k=max_num_clusters, M=D, which="SM")
        eig_vals = eig_vals[1:]
        eig_vecs = eig_vecs[:, 1:]
        return eig_vals, eig_vecs

    def spectral_embedding_0(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes dense spectral embedding using ``scipy.linalg.eigh``.

        Args:
          x: Dense affinity/similarity matrix with shape ``(num_nodes, num_nodes)``.

        Returns:
          eig_vals: Eigenvalues associated with the selected embedding vectors.
          eig_vecs: Eigenvectors with shape ``(num_nodes, num_eigenvectors)``.
        """
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError(f"x must be a square matrix, got shape={x.shape}")
        num_nodes = x.shape[0]
        if num_nodes < 2:
            raise ValueError("x must contain at least 2 nodes for spectral embedding")
        x.flat[:: num_nodes + 1] = 0
        d = np.sum(x, axis=1)
        D = None
        if self.laplacian == LaplacianType.unnormalized:
            L = np.diag(d) - x
        elif self.laplacian == LaplacianType.norm_sym:
            idsqrt = np.zeros_like(d, dtype=float)
            mask = d > 0
            idsqrt[mask] = 1.0 / np.sqrt(d[mask])
            L = np.eye(num_nodes) - idsqrt[:, None] * x * idsqrt
        elif self.laplacian == LaplacianType.norm_rw:
            D = np.diag(d)
            L = D - x

        max_num_clusters = num_nodes
        if self.max_num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.max_num_clusters)
        if self.num_clusters is not None:
            max_num_clusters = min(max_num_clusters, self.num_clusters)
        if max_num_clusters < 2:
            raise ValueError(
                f"invalid effective max_num_clusters={max_num_clusters}; "
                "needs to be >= 2 for dense spectral embedding path"
            )

        eig_vals, eig_vecs = eigh(
            L, b=D, overwrite_a=True, subset_by_index=[1, max_num_clusters - 1]
        )

        return eig_vals, eig_vecs

    def compute_eigengap(self, eig_vals: np.ndarray) -> Dict[str, Any]:
        """Computes eigengap statistics used for cluster-count prediction.

        Args:
          eig_vals: Sorted eigenvalues (excluding the trivial first one).

        Returns:
          Dictionary with eigenvalue/eigengap derived statistics.
        """
        eig_vals = np.concatenate(([0.0], eig_vals))
        eigengap = np.diff(np.concatenate(([0.0], eig_vals)))
        filter = np.array([1 / 60, -3 / 20, 3 / 4, 0.0, -3 / 4, 3 / 20, -1 / 60])
        eig_vals_ext = np.concatenate((eig_vals, [eig_vals[-1]] * 3))
        d_eig_vals = np.convolve(eig_vals_ext, filter)[3:-6]
        k_max = np.argmax(eigengap)
        gap_max = eigengap[k_max]
        eigengap_stats = {
            "eig_vals": eig_vals,
            "eigengap": eigengap,
            "gap_max": gap_max,
            "k_max": k_max,
            "d_eig_vals": d_eig_vals,
        }
        return eigengap_stats

    def predict_num_clusters(self, eigengap_stats: Optional[Dict[str, Any]]) -> int:
        """Predicts number of clusters from eigengap statistics.

        Args:
          eigengap_stats: Output of :meth:`compute_eigengap`, or ``None`` when
            ``self.num_clusters`` is fixed.

        Returns:
          Predicted (or fixed) number of clusters.
        """
        if self.num_clusters is not None:
            num_clusters = self.num_clusters
        elif eigengap_stats is None:
            raise ValueError(
                "eigengap_stats cannot be None when num_clusters is not fixed"
            )
        elif self.criterion == SpectralClusteringNumClassCriterion.max_eigengap:
            num_clusters = eigengap_stats["k_max"] + 1
        elif self.criterion == SpectralClusteringNumClassCriterion.max_d_eig_vals:
            num_clusters = np.argmax(eigengap_stats["d_eig_vals"]) + 1
        elif self.criterion == SpectralClusteringNumClassCriterion.thr_eigengap:
            nz = (eigengap_stats["eigengap"] < self.thr_eigengap).nonzero()[0]
            nz_after_kmax = nz[nz > eigengap_stats["k_max"]]
            if len(nz_after_kmax) > 0:
                num_clusters = nz_after_kmax[0] + 1
            else:
                num_clusters = len(eigengap_stats["eigengap"])
        elif self.criterion == SpectralClusteringNumClassCriterion.thr_d_eig_vals:
            nz = (eigengap_stats["d_eig_vals"] < self.thr_eigengap).nonzero()[0]
            nz_after_kmax = nz[nz > eigengap_stats["k_max"]]
            if len(nz_after_kmax) > 0:
                num_clusters = nz_after_kmax[0] + 1
            else:
                num_clusters = len(eigengap_stats["d_eig_vals"])
        else:
            raise ValueError(f"invalid num clusters criterion {self.criterion}")
        return num_clusters

    def normalize_eigvecs(self, eig_vecs: np.ndarray) -> np.ndarray:
        """Applies row-normalization to eigenvectors when required.

        Args:
          eig_vecs: Spectral embedding vectors.

        Returns:
          Normalized (or unchanged) embedding vectors.
        """
        if self.laplacian == LaplacianType.norm_sym:
            return normalize(eig_vecs, axis=1)
        else:
            return eig_vecs

    def do_kmeans(
        self, x: np.ndarray, num_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Runs k-means on spectral embeddings.

        Args:
          x: Spectral embeddings with shape ``(num_samples, emb_dim)``.
          num_clusters: Number of clusters. If ``None``, uses ``x.shape[1] + 1``.

        Returns:
          Cluster assignments with shape ``(num_samples,)``.
        """
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

    def fit(self, x: np.ndarray) -> Tuple[np.ndarray, int, Optional[Dict[str, Any]]]:
        """Performs spectral clustering.

        Args:
          x: Affinity/similarity matrix with shape ``(num_nodes, num_nodes)``.

        Returns:
          y: Cluster assignments with shape ``(num_nodes,)``.
          num_clusters: Predicted (or fixed) number of clusters.
          eigengap_stats: Eigengap statistics dictionary, or ``None`` when
            ``num_clusters`` is fixed.
        """
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError(f"x must be a square matrix, got shape={x.shape}")
        if x.shape[0] < 1:
            raise ValueError("x must contain at least one node")
        if x.shape[0] == 1:
            if self.num_clusters is not None and self.num_clusters != 1:
                raise ValueError(
                    f"num_clusters={self.num_clusters} is infeasible for num_nodes=1; "
                    "max feasible is 1"
                )
            return np.zeros((1,), dtype=int), 1, None
        if self.num_clusters == 1:
            return np.zeros((x.shape[0],), dtype=int), 1, None
        if self.num_clusters is not None:
            max_feasible = x.shape[0] - 1
            if self.max_num_clusters is not None:
                max_feasible = min(max_feasible, self.max_num_clusters)
            if self.num_clusters > max_feasible:
                raise ValueError(
                    f"num_clusters={self.num_clusters} is infeasible for "
                    f"num_nodes={x.shape[0]} and "
                    f"max_num_clusters={self.max_num_clusters}; "
                    f"max feasible is {max_feasible}"
                )

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
    ) -> None:
        """Plots eigengap statistics.

        Args:
          eigengap_stats: Dictionary returned by :meth:`compute_eigengap`.
          num_clusters: Selected number of clusters.
          fig_file: Optional output path to save figure.

        Returns:
          ``None``.
        """
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
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds class arguments to a jsonargparse parser.

        Args:
          parser: jsonargparse parser instance.
          prefix: argument prefix.

        Returns:
          ``None``.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--laplacian",
            default=LaplacianType.norm_sym,
            choices=LaplacianType.choices(),
            help="graph Laplacian type used to compute spectral embeddings",
        )
        parser.add_argument(
            "--num-clusters",
            default=None,
            type=int,
            help="fixed number of clusters; if omitted, estimate from eigengap",
        )
        parser.add_argument(
            "--max-num-clusters",
            default=None,
            type=int,
            help="maximum number of clusters/eigenvectors considered for automatic selection",
        )
        parser.add_argument(
            "--criterion",
            default=SpectralClusteringNumClassCriterion.max_eigengap,
            choices=SpectralClusteringNumClassCriterion.choices(),
            help="criterion used to estimate number of clusters when --num-clusters is not set",
        )
        parser.add_argument(
            "--thr-eigengap",
            default=1e-3,
            type=float,
            help="threshold used by threshold-based cluster-count criteria",
        )
        parser.add_argument(
            "--kmeans-epochs",
            default=100,
            type=int,
            help="maximum number of k-means epochs in spectral embedding space",
        )
        parser.add_argument(
            "--kmeans-init-method",
            default=KMeansInitMethod.max_dist,
            choices=KMeansInitMethod.choices(),
            help="k-means centroid initialization strategy",
        )
        parser.add_argument(
            "--num-workers",
            default=1,
            type=int,
            help="number of worker threads used by k-means",
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )
