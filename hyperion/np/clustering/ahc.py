"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py
from copy import copy

from scipy.cluster.hierarchy import linkage
from sklearn.metrics import homogeneity_score, completeness_score

from ...hyp_defs import float_cpu
from ..np_model import NPModel


class AHC(NPModel):
    """Agglomerative Hierarchical Clustering class.

    Attributes:
      method: linkage method to calculate the distance between a new agglomerated
              cluster and the rest of clusters.
              This can be ["average", "single", "complete", "weighted", "centroid", "median", "ward"].
              See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
      metric: indicates the type of metric used to calculate the input scores.
              It can be: "llr" (log-likelihood ratios), "prob" (probabilities), "distance": (distance metric).
    """

    def __init__(self, method="average", metric="llr", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.metric = metric
        self.Z = None
        self.flat_clusters = None

    def fit(self, x, mask=None):
        """Performs the clustering.
           It stores the AHC tree in the Z property of the object.

        Args:
          x: input score matrix (num_samples, num_samples).
             It will use the upper triangular matrix only.
          mask: boolean mask where False in position i,j means that
                nodes i and j should not be merged.
        """

        if mask is not None:
            x = copy(x)
            x[mask == False] = -1e10

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

    def get_flat_clusters(self, t, criterion="threshold"):
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
        else:
            return self.get_flat_clusters_from_num_clusters(t)

    def get_flat_clusters_from_num_clusters(self, num_clusters):
        """Computes the flat clusters from the AHC tree using
        num_clusters criterion"
        """
        N = self.Z.shape[0] + 1
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

    def get_flat_clusters_from_thr(self, thr):
        """Computes the flat clusters from the AHC tree using
        threshold criterion"
        """
        if self.metric == "llr" or self.metric == "prob":
            idx = self.Z[:, 2] >= thr
        else:
            idx = self.Z[:, 2] <= thr
        num_clusters = self.Z.shape[0] + 1 - np.sum(idx)
        return self.get_flat_clusters_from_num_clusters(num_clusters)

    def compute_flat_clusters(self):
        """Computes the flat clusters for all possible number of clusters

        Returns:
            numpy matrix (num_samples, num_samples) where row i contains the
            clusters assignments for the case of choosing num_samples - i clusters.
        """
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

    def evaluate_homogeneity_completeness_tradeoff(self, true_labels):
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
