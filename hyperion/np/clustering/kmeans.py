"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import logging
import numpy as np
import h5py

from ...hyp_defs import float_cpu
from ..np_model import NPModel


class KMeans(NPModel):
    """K-Means clustering class.

    Attributes:
      num_clusters: number of clusters.
      mu: cluster centers.
      rtol: minimum delta in loss function used as stopping criterion.
    """

    def __init__(self, num_clusters, mu=None, rtol=0.001, **kwargs):
        super(KMeans, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.mu = mu
        self.rtol = rtol

    def fit(self, x, epochs=100):
        """Performs the clustering.

        Args:
          x: input data (num_samples, feat_dim).
          epochs: max. number of epochs.

        Returns:
          loss: value of loss function (num_epochs,).
          cluster_index: clustering labels as int numpy array with shape=(num_samples,)
        """
        loss = np.zeros((epochs,), dtype=float_cpu())
        self.mu = self._choose_seeds(x)
        cluster_index, err2 = self.predict(x)
        for epoch in range(epochs):
            self.mu = self._compute_centroids(x, cluster_index)
            cluster_index, err2 = self.predict(x)
            loss[epoch] = np.mean(err2)
            if epoch > 0:
                delta = np.abs(loss[epoch - 1] - loss[epoch]) / loss[epoch - 1]
                if delta < self.rtol:
                    loss = loss[: epoch + 1]
                    break

        return loss, cluster_index

    def _choose_seeds(self, x):
        """Chooses the initial seeds for the clustering.

        Args:
          x: input data (num_samples, feat_dim).

        Returns:
          Initial centers (num_clusters, feat_dim)
        """
        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        mu[0] = x[0]
        for i in range(1, self.num_clusters):
            d = np.zeros((x.shape[0],), dtype=float_cpu())
            for j in range(i):
                d += np.sum(np.square(x - mu[j]), axis=-1)
            index = np.argmax(d)
            mu[i] = x[index]
        return mu

    def _compute_centroids(self, x, index):
        """Compute the centroids given cluster assigments.

        Args:
          x: input data (num_samples, feat_dim)
          index: cluster assignments as integers with shape=(num_samples,)

        Returns:
          Cluster centroids (num_clusters, feat_dim)
        """
        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        for k in range(self.num_clusters):
            r = index == k
            if np.sum(r) > 0:
                mu[k] = np.mean(x[index == k], axis=0)
        return mu

    def predict(self, x):
        """Compute the cluster labels for new data.

        Args:
          x: input data (num_samples, feat_dim)

        Returns:
          Cluster assignments as integer array (num_samples,)
          Square distance of each element to the center of its cluster.
        """
        err2 = np.zeros((x.shape[0], self.num_clusters), dtype=float_cpu())
        for k in range(self.num_clusters):
            err2[:, k] = np.sum(np.square(x - self.mu[k]), axis=-1)

        index = np.argmin(err2, axis=-1)
        return index, err2[np.arange(x.shape[0]), index]
