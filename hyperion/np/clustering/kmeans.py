"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import h5py
import numpy as np

from ...hyp_defs import float_cpu
from ..np_model import NPModel


class KMeansInitMethod(str, Enum):
    max_dist = "max_dist"
    random = "random"

    @staticmethod
    def choices():
        return [KMeansInitMethod.max_dist, KMeansInitMethod.random]


class KMeans(NPModel):
    """K-Means clustering class.

    Attributes:
      num_clusters: number of clusters.
      mu: cluster centers.
      rtol: minimum delta in loss function used as stopping criterion.
    """

    def __init__(
        self,
        num_clusters,
        mu=None,
        rtol=0.001,
        epochs=100,
        init_method=KMeansInitMethod.max_dist,
        num_workers=1,
        verbose=True,
        rng_seed=11235813,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.mu = mu
        self.rtol = rtol
        self.epochs = epochs
        self.verbose = verbose
        self.num_workers = num_workers
        self.init_method = init_method
        if self.init_method == KMeansInitMethod.random:
            self.rng = np.random.default_rng(seed=rng_seed)

    def fit(self, x):
        """Performs the clustering.

        Args:
          x: input data (num_samples, feat_dim).
          epochs: max. number of epochs.

        Returns:
          loss: value of loss function (num_epochs,).
          cluster_index: clustering labels as int numpy array with shape=(num_samples,)
        """
        loss = np.zeros((self.epochs,), dtype=float_cpu())
        if self.init_method == KMeansInitMethod.max_dist:
            if self.num_workers == 1:
                self.mu = self._choose_seeds_max_dist(x)
            else:
                self.mu = self._choose_seeds_max_dist_multithread(x)
        else:
            self.mu = self._choose_seeds_random(x)

        cluster_index, err2 = self(x)
        for epoch in range(self.epochs):
            if self.num_workers == 1:
                self.mu = self._compute_centroids(x, cluster_index)
            else:
                self.mu = self._compute_centroids_multithread(x, cluster_index)
            cluster_index, err2 = self(x)
            loss[epoch] = np.mean(err2)
            if epoch > 0:
                delta = np.abs(loss[epoch - 1] - loss[epoch]) / (
                    loss[epoch - 1] + 1e-10
                )
                if self.verbose:
                    logging.info(
                        "epoch: %d loss: %f rdelta: %f", epoch, loss[epoch], delta
                    )
                if delta < self.rtol:
                    loss = loss[: epoch + 1]
                    break
            else:
                if self.verbose:
                    logging.info("epoch: %d loss: %f", epoch, loss[epoch])

        return loss, cluster_index

    def _choose_seeds_random(self, x):
        """Chooses the initial seeds for the clustering randomly.

        Args:
          x: input data (num_samples, feat_dim).

        Returns:
          Initial centers (num_clusters, feat_dim)
        """
        if self.verbose:
            logging.info("choosing seeds")

        mu = self.rng.choice(x, size=(self.num_clusters,), replace=False, shuffle=False)
        if self.verbose:
            logging.info("%d seeds chosen", self.num_clusters)

        return mu

    def _choose_seeds_max_dist(self, x):
        """Chooses the initial seeds for the clustering.

        Args:
          x: input data (num_samples, feat_dim).

        Returns:
          Initial centers (num_clusters, feat_dim)
        """
        if self.verbose:
            logging.info("choosing seeds")
        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        mu[0] = x[0]
        for i in range(1, self.num_clusters):
            d = np.zeros((x.shape[0],), dtype=float_cpu())
            for j in range(i):
                d += np.sum(np.square(x - mu[j]), axis=-1)
            index = np.argmax(d)
            mu[i] = x[index]
        return mu

    @staticmethod
    def _compute_d2(x, mu):
        return np.sum(np.square(x - mu), axis=-1)

    def _choose_seeds_max_dist_multithread(self, x):
        """Chooses the initial seeds for the clustering.

        Args:
          x: input data (num_samples, feat_dim).

        Returns:
          Initial centers (num_clusters, feat_dim)
        """
        if self.verbose:
            logging.info("choosing seeds")

        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            mu[0] = x[0]
            for i in range(1, self.num_clusters):
                d = np.zeros((x.shape[0],), dtype=float_cpu())

                futures = {
                    executor.submit(KMeans._compute_d2, x, mu[j]): j for j in range(i)
                }
                for future in as_completed(futures):
                    d += future.result()

                index = np.argmax(d)
                mu[i] = x[index]
                if self.verbose and (i % 10 == 0 or i == self.num_clusters - 1):
                    logging.info("%d seeds chosen", i + 1)
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
                mu[k] = np.mean(x[r], axis=0)
        return mu

    @staticmethod
    def _compute_centroid(x, index, k):
        r = index == k
        if np.sum(r) > 0:
            return np.mean(x[r], axis=0)
        else:
            return None

    def _compute_centroids_multithread(self, x, index):
        """Compute the centroids given cluster assigments.

        Args:
          x: input data (num_samples, feat_dim)
          index: cluster assignments as integers with shape=(num_samples,)

        Returns:
          Cluster centroids (num_clusters, feat_dim)
        """
        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(KMeans._compute_centroid, x, index, k): k
                for k in range(self.num_clusters)
            }
            for future in as_completed(futures):
                k = futures[future]
                mu_k = future.result()
                if mu_k is not None:
                    mu[k] = mu_k

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

    def predict_multithread(self, x):
        """Compute the cluster labels for new data.

        Args:
          x: input data (num_samples, feat_dim)

        Returns:
          Cluster assignments as integer array (num_samples,)
          Square distance of each element to the center of its cluster.
        """
        err2 = np.zeros((x.shape[0], self.num_clusters), dtype=float_cpu())
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(KMeans._compute_d2, x, self.mu[k]): k
                for k in range(self.num_clusters)
            }
            for future in as_completed(futures):
                k = futures[future]
                err2[:, k] = future.result()

        index = np.argmin(err2, axis=-1)
        return index, err2[np.arange(x.shape[0]), index]

    def __call__(self, x):
        if self.num_workers == 1:
            return self.predict(x)
        else:
            return self.predict_multithread(x)
