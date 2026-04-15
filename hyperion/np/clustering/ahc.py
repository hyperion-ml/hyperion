"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from copy import copy
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import completeness_score, homogeneity_score

from ...hyp_defs import float_cpu
from ..hyper_np_model import HyperNPModel


class AHC(HyperNPModel):
    """Agglomerative Hierarchical Clustering class.

    Attributes:
      method: linkage method to calculate the distance between a new agglomerated
              cluster and the rest of clusters.
              This can be ["average", "single", "complete", "weighted", "centroid", "median", "ward"].
              See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
      metric: indicates the type of metric used to calculate the input scores.
              It can be: "llr" (log-likelihood ratios), "prob" (probabilities), "distance": (distance metric).

    Example:
      >>> import numpy as np
      >>> from hyperion.np.clustering.ahc import AHC
      >>> x = np.array([
      ...     [0.0, 0.9, 0.2, 0.1],
      ...     [0.9, 0.0, 0.3, 0.2],
      ...     [0.2, 0.3, 0.0, 0.8],
      ...     [0.1, 0.2, 0.8, 0.0],
      ... ], dtype=np.float32)
      >>> ahc = AHC(method="average", metric="llr")
      >>> ahc.fit(x)
      >>> clusters_thr = ahc.get_flat_clusters(t=0.5, criterion="threshold")
      >>> clusters_k2 = ahc.get_flat_clusters(t=2, criterion="num_clusters")
    """

    def __init__(
        self, method: str = "average", metric: str = "llr", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.metric = metric
        self.Z: Optional[np.ndarray] = None
        self.flat_clusters: Optional[np.ndarray] = None

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"method": self.method, "metric": self.metric}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def fit(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Performs the clustering.
           It stores the AHC tree in the Z property of the object.

        Args:
          x: input score matrix (num_samples, num_samples).
             It will use the upper triangular matrix only.
          mask: boolean mask where False in position i,j means that
                nodes i and j should not be merged.
        """
        # Invalidate cached flat clusterings from previous fits.
        self.flat_clusters = None

        if mask is not None:
            if self.metric == "llr" or self.metric == "prob":
                x = copy(x)
                x[mask == False] = -1e10
            else:
                raise ValueError(
                    "mask is only supported for metric='llr' or metric='prob'"
                )

        idx = np.triu(np.ones_like(x, dtype=bool), k=1)
        scores = x[idx]

        if self.metric == "llr":
            max_score = np.max(scores)
            scores = -scores + max_score
            self.Z = linkage(scores, method=self.method)
            self.Z[:, 2] = -self.Z[:, 2] + max_score
        elif self.metric == "prob":
            scores = 1 - scores
            self.Z = linkage(scores, method=self.method)
            self.Z[:, 2] = 1 - self.Z[:, 2]
        else:
            self.Z = linkage(scores, method=self.method, metric=self.metric)

    def get_flat_clusters(
        self, t: Union[int, float], criterion: str = "threshold"
    ) -> np.ndarray:
        """Computes the flat clusters from the AHC tree.

        Args:
          t: threshold or number of clusters
          criterion: if "threshold" with llr/prob larger than threshold or
                    distance lower than threshold.
                     if "num_clusters" returns the clusters corresponding
                     to selecting a given number of clusters.

        Returns:
          Clusters assigments for x as numpy integer vector (num_samples,).
        """
        if criterion == "threshold":
            return self.get_flat_clusters_from_thr(t)
        if criterion == "num_clusters":
            return self.get_flat_clusters_from_num_clusters(t)
        raise ValueError(
            f"Invalid criterion={criterion!r}. "
            "Expected one of: {'threshold', 'num_clusters'}"
        )

    def get_flat_clusters_from_num_clusters(self, num_clusters: int) -> np.ndarray:
        """Computes the flat clusters from the AHC tree using
        num_clusters criterion"
        """
        if self.Z is None:
            raise ValueError("AHC model is not fitted. Call fit() before clustering.")
        if not isinstance(num_clusters, (int, np.integer)):
            raise ValueError(
                f"num_clusters must be an integer in [1, N], got {num_clusters!r}"
            )

        N = self.Z.shape[0] + 1
        if num_clusters < 1 or num_clusters > N:
            raise ValueError(f"num_clusters must be in [1, {N}], got {num_clusters}")
        num_clusters = min(N, num_clusters)
        p_idx = N - num_clusters
        if self.flat_clusters is not None:
            return self.flat_clusters[p_idx]

        flat_clusters = np.arange(N, dtype=int)
        for i in range(p_idx):
            segm_idx = np.logical_or(
                flat_clusters == self.Z[i, 0], flat_clusters == self.Z[i, 1]
            )
            flat_clusters[segm_idx] = N + i

        _, flat_clusters = np.unique(flat_clusters, return_inverse=True)
        return flat_clusters

    def get_flat_clusters_from_thr(self, thr: float) -> np.ndarray:
        """Computes the flat clusters from the AHC tree using
        threshold criterion"
        """
        if self.Z is None:
            raise ValueError("AHC model is not fitted. Call fit() before clustering.")
        if self.metric == "llr" or self.metric == "prob":
            idx = self.Z[:, 2] >= thr
        else:
            idx = self.Z[:, 2] <= thr
        num_clusters = self.Z.shape[0] + 1 - np.sum(idx)
        return self.get_flat_clusters_from_num_clusters(num_clusters)

    def compute_flat_clusters(self) -> None:
        """Computes the flat clusters for all possible number of clusters

        Returns:
            numpy matrix (num_samples, num_samples) where row i contains the
            clusters assignments for the case of choosing num_samples - i clusters.
        """
        if self.Z is None:
            raise ValueError("AHC model is not fitted. Call fit() before clustering.")
        N = self.Z.shape[0] + 1
        flat_clusters = np.zeros((N, N), dtype=int)
        flat_clusters[0] = np.arange(N, dtype=int)
        for i in range(N - 1):
            flat_clusters[i + 1] = flat_clusters[i]
            segm_idx = np.logical_or(
                flat_clusters[i] == self.Z[i, 0], flat_clusters[i] == self.Z[i, 1]
            )
            flat_clusters[i + 1][segm_idx] = N + i

        for i in range(1, N):
            _, flat_clusters[i] = np.unique(flat_clusters[i], return_inverse=True)
        self.flat_clusters = flat_clusters
        return flat_clusters

    def evaluate_homogeneity_completeness_tradeoff(
        self, true_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the curve homogeneity versus completeness where
              Homogeneity: each cluster contains only members of a single class. (cluster purity)
              Completeness: all members of a given class are assigned to the same cluster. (class purity)

        Args:
          true_labels: true cluster labels

        Returns:
            homogeneity vector (num_samples,)
            completenes vector (num_samples,)
        """
        if self.flat_clusters is None:
            self.compute_flat_clusters()

        N = self.flat_clusters.shape[0]
        h = np.zeros((N,), dtype=float_cpu())
        c = np.zeros((N,), dtype=float_cpu())
        for i in range(self.flat_clusters.shape[0]):
            h[i] = homogeneity_score(true_labels, self.flat_clusters[i])
            c[i] = completeness_score(true_labels, self.flat_clusters[i])

        return h, c
