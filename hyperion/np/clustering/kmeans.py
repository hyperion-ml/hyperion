"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Optional, Tuple, Union

import numpy as np

from ...hyp_defs import float_cpu
from ..hyper_np_model import HyperNPModel


class KMeansInitMethod(str, Enum):
    max_dist = "max_dist"
    random = "random"

    @staticmethod
    def choices() -> list["KMeansInitMethod"]:
        return [KMeansInitMethod.max_dist, KMeansInitMethod.random]


class KMeans(HyperNPModel):
    """K-Means clustering class.

    Attributes:
      num_clusters: Number of clusters.
      mu: Cluster centers with shape ``(num_clusters, feat_dim)``.
      rtol: Minimum relative change in loss used as stopping criterion.
      epochs: Maximum number of optimization iterations.
      init_method: Seed initialization strategy.
      num_workers: Number of threads for parallel operations.
      verbose: If True, emits training progress logs.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.clustering.kmeans import KMeans
      >>> x = np.array([
      ...     [0.0, 0.1],
      ...     [0.2, 0.0],
      ...     [3.0, 3.1],
      ...     [2.9, 3.0],
      ... ], dtype=np.float32)
      >>> km = KMeans(num_clusters=2, init_method="max_dist", epochs=20, rtol=1e-3)
      >>> loss, labels = km.fit(x)
      >>> pred_labels, err2 = km.predict(x)
    """

    def __init__(
        self,
        num_clusters: int,
        mu: Optional[np.ndarray] = None,
        rtol: float = 0.001,
        epochs: int = 100,
        init_method: Union[KMeansInitMethod, str] = KMeansInitMethod.max_dist,
        num_workers: int = 1,
        verbose: bool = True,
        rng_seed: int = 11235813,
        **kwargs: Any,
    ) -> None:
        """Initializes a ``KMeans`` model.

        Args:
          num_clusters: Number of clusters.
          mu: Optional initial cluster centers.
          rtol: Relative tolerance for convergence based on loss change.
          epochs: Maximum number of epochs.
          init_method: Seed initialization method.
          num_workers: Number of worker threads.
          verbose: If True, logs training progress.
          rng_seed: Random seed used when ``init_method='random'``.
          **kwargs: Extra arguments forwarded to ``HyperNPModel``.
        """
        super().__init__(**kwargs)
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError(
                f"num_clusters must be a positive integer, got {num_clusters!r}"
            )
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(
                f"num_workers must be a positive integer, got {num_workers!r}"
            )
        if rtol < 0:
            raise ValueError(f"rtol must be >= 0, got {rtol!r}")
        try:
            init_method = KMeansInitMethod(init_method)
        except ValueError as err:
            valid = [m.value for m in KMeansInitMethod]
            raise ValueError(
                f"invalid init_method={init_method!r}, expected one of {valid}"
            ) from err

        self.num_clusters = num_clusters
        self.mu = mu
        self.rtol = rtol
        self.epochs = epochs
        self.verbose = verbose
        self.num_workers = num_workers
        self.init_method = init_method
        self.rng: np.random.Generator = np.random.default_rng(seed=rng_seed)

    def fit(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the clustering.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          loss: Loss values with shape ``(num_epochs_used,)``.
          cluster_index: Cluster labels with shape ``(num_samples,)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        if self.num_clusters > x.shape[0]:
            raise ValueError(
                f"num_clusters ({self.num_clusters}) cannot exceed number of samples ({x.shape[0]})"
            )
        if self.mu is not None:
            if self.mu.ndim != 2:
                raise ValueError(
                    f"mu must be 2D with shape (num_clusters, feat_dim), got shape={self.mu.shape}"
                )
            if self.mu.shape[0] != self.num_clusters:
                raise ValueError(
                    f"mu first dimension must equal num_clusters={self.num_clusters}, "
                    f"got shape={self.mu.shape}"
                )
            if self.mu.shape[1] != x.shape[1]:
                raise ValueError(
                    f"mu feature dimension must match x.shape[1]={x.shape[1]}, "
                    f"got shape={self.mu.shape}"
                )
            if not np.all(np.isfinite(self.mu)):
                raise ValueError("mu must contain only finite values")

        loss = np.zeros((self.epochs,), dtype=float_cpu())
        if self.mu is None:
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

    def _choose_seeds_random(self, x: np.ndarray) -> np.ndarray:
        """Chooses the initial seeds for the clustering randomly.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          Initial centers with shape ``(num_clusters, feat_dim)``.
        """
        if self.verbose:
            logging.info("choosing seeds")
        if self.num_clusters > x.shape[0]:
            raise ValueError(
                f"num_clusters ({self.num_clusters}) cannot exceed number of samples ({x.shape[0]})"
            )

        mu = self.rng.choice(x, size=(self.num_clusters,), replace=False, shuffle=False)
        if self.verbose:
            logging.info("%d seeds chosen", self.num_clusters)

        return mu

    def _choose_seeds_max_dist(self, x: np.ndarray) -> np.ndarray:
        """Chooses the initial seeds for the clustering.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          Initial centers with shape ``(num_clusters, feat_dim)``.
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
    def _compute_d2(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Computes squared distances from all points in ``x`` to center ``mu``.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.
          mu: Cluster center with shape ``(feat_dim,)``.

        Returns:
          Squared distances with shape ``(num_samples,)``.
        """
        return np.sum(np.square(x - mu), axis=-1)

    def _choose_seeds_max_dist_multithread(self, x: np.ndarray) -> np.ndarray:
        """Chooses the initial seeds for the clustering.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          Initial centers with shape ``(num_clusters, feat_dim)``.
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

    def _compute_centroids(self, x: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Compute the centroids given cluster assigments.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.
          index: Cluster assignments with shape ``(num_samples,)``.

        Returns:
          Cluster centroids with shape ``(num_clusters, feat_dim)``.
        """
        mu = np.zeros((self.num_clusters, x.shape[-1]), dtype=float_cpu())
        for k in range(self.num_clusters):
            r = index == k
            if np.sum(r) > 0:
                mu[k] = np.mean(x[r], axis=0)
            else:
                mu[k] = self._reinit_empty_centroid_from_data(x)
        return mu

    @staticmethod
    def _compute_centroid(
        x: np.ndarray, index: np.ndarray, k: int
    ) -> Optional[np.ndarray]:
        """Computes one centroid.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.
          index: Cluster assignments with shape ``(num_samples,)``.
          k: Cluster index.

        Returns:
          Cluster centroid with shape ``(feat_dim,)`` or ``None`` when empty.
        """
        r = index == k
        if np.sum(r) > 0:
            return np.mean(x[r], axis=0)
        else:
            return None

    def _reinit_empty_centroid_from_data(self, x: np.ndarray) -> np.ndarray:
        """Reinitializes an empty centroid using an existing sample from ``x``."""
        idx = int(self.rng.integers(0, x.shape[0]))
        return x[idx]

    def _compute_centroids_multithread(
        self, x: np.ndarray, index: np.ndarray
    ) -> np.ndarray:
        """Compute the centroids given cluster assigments.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.
          index: Cluster assignments with shape ``(num_samples,)``.

        Returns:
          Cluster centroids with shape ``(num_clusters, feat_dim)``.
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
                else:
                    mu[k] = self._reinit_empty_centroid_from_data(x)

        return mu

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the cluster labels for new data.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          index: Cluster assignments with shape ``(num_samples,)``.
          err2: Squared distance to the assigned center with shape ``(num_samples,)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        if self.mu is None:
            raise ValueError("KMeans model is not initialized/fitted: mu is None")

        err2 = np.zeros((x.shape[0], self.num_clusters), dtype=float_cpu())
        for k in range(self.num_clusters):
            err2[:, k] = np.sum(np.square(x - self.mu[k]), axis=-1)

        index = np.argmin(err2, axis=-1)
        return index, err2[np.arange(x.shape[0]), index]

    def predict_multithread(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the cluster labels for new data.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          index: Cluster assignments with shape ``(num_samples,)``.
          err2: Squared distance to the assigned center with shape ``(num_samples,)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        if self.mu is None:
            raise ValueError("KMeans model is not initialized/fitted: mu is None")

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

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatches to single-thread or multi-thread prediction.

        Args:
          x: Input data with shape ``(num_samples, feat_dim)``.

        Returns:
          index: Cluster assignments with shape ``(num_samples,)``.
          err2: Squared distance to the assigned center with shape ``(num_samples,)``.
        """
        if self.num_workers == 1:
            return self.predict(x)
        else:
            return self.predict_multithread(x)
