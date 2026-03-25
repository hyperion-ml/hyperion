"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from numpy.random import Generator
from scipy.special import erf

from ....hyp_defs import float_cpu
from ....utils.math_funcs import logsumexp, softmax
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_3D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
)
from ...clustering import KMeans
from .exp_family_mixture import NBestType
from .gmm_diag_cov import GMMDiagCov


class GMMTiedDiagCov(GMMDiagCov):
    """Gaussian mixture model with diagonal covariances tied across components.

    Attributes:
        num_comp: Number of mixture components.
        pi: Mixture weights.
        mu: Component means with shape ``(num_comp, x_dim)``.
        Lambda: Shared diagonal precision vector with shape ``(x_dim,)``.
        var_floor: Variance flooring constant.
        update_mu: Whether to update ``mu`` during training.
        update_Lambda: Whether to update the shared precision during training.
        x_dim: Data dimensionality inferred from ``mu`` if provided.

    Examples:
        >>> import numpy as np
        >>> from hyperion.np.pdfs.mixtures.gmm_tied_diag_cov import GMMTiedDiagCov
        >>> x = np.random.randn(300, 5).astype("float32")
        >>> gmm = GMMTiedDiagCov(num_comp=3, x_dim=5)
        >>> _ = gmm.fit(x, epochs=2)
        >>> llk = gmm.log_prob(x[:4])
        >>> llk.shape
        (4,)
    """

    def __init__(
        self,
        num_comp: int = 1,
        pi: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        Lambda: Optional[np.ndarray] = None,
        var_floor: float = 1e-3,
        update_mu: bool = True,
        update_Lambda: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the tied-covariance GMM.

        Args:
            num_comp: Number of components when ``pi`` is not provided.
            pi: Optional mixture weights.
            mu: Optional component means.
            Lambda: Optional shared diagonal precision per dimension.
            var_floor: Minimum variance allowed during updates.
            update_mu: Whether to update ``mu`` during EM.
            update_Lambda: Whether to update the shared precision during EM.
            **kwargs: Extra keyword arguments forwarded to :class:`GMMDiagCov`.
        """
        super().__init__(
            num_comp=num_comp,
            pi=pi,
            mu=mu,
            Lambda=Lambda,
            var_floor=var_floor,
            update_mu=update_mu,
            update_Lambda=update_Lambda,
            **kwargs,
        )

    def _compute_gmm_nat_std(self) -> None:
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()
            self._compute_nat_params()
        elif self.eta is not None:
            self._validate_eta()
            self.A = self.compute_A_nat(self.eta)
            self._compute_std_params()

    def _initialize_stdnormal(self) -> None:
        """Initializes a single-component standard normal GMM."""
        self.pi = np.array([1], dtype=float_cpu())
        self.mu = np.zeros((1, self.x_dim), dtype=float_cpu())
        self.Lambda = np.ones((self.x_dim,), dtype=float_cpu())

    def _initialize_kmeans(self, num_comp: int, x: np.ndarray) -> None:
        """Initializes the GMM using k-means centroids.

        Args:
            num_comp: Number of components to initialize.
            x: Initialization data with shape ``(num_samples, x_dim)``.
        """
        if num_comp == 1:
            self.pi = np.array([1], dtype=float_cpu())
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.Lambda = 1 / np.std(x, axis=0, keepdims=False) ** 2
            return

        kmeans = KMeans(num_clusters=num_comp, epochs=100)
        loss, cluster_index = kmeans.fit(x)

        self.mu = kmeans.mu
        self.pi = np.zeros((self.num_comp,), dtype=float_cpu())
        C = np.zeros((x.shape[-1],), dtype=float_cpu())
        for k in range(num_comp):
            r = cluster_index == k
            self.pi[k] = np.sum(r) / x.shape[0]
            delta = x[r] - self.mu[k]
            C += np.sum(delta**2, axis=0)

        self.Lambda = x.shape[0] / C

    def Mstep(self, N: np.ndarray, u_x: np.ndarray) -> None:
        """Maximization step for tied-diagonal covariances.

        Args:
            N: Zeroth-order statistics per component.
            u_x: Stacked first- and second-order statistics.
        """
        F, S = self.unstack_suff_stats(u_x)
        N = np.maximum(N, 1e-5)

        if self.update_mu:
            self.mu = F / N[:, None]
        elif self.mu is None:
            raise ValueError("mu must be initialized if update_mu is False")

        if self.update_Lambda:
            S = S / N[:, None] - self.mu**2
            active = N > self.min_N
            if np.any(active):
                S_floor = self.var_floor * np.mean(S[active], axis=0)
            else:
                S_floor = self.var_floor * np.mean(S, axis=0)
            S_floor = np.maximum(S_floor, 1e-10)
            S = np.maximum(S, S_floor)
            Spool = np.sum(N[:, None] * S, axis=0) / np.sum(N)
            self.Lambda = 1 / Spool
            self._Sigma = Spool
            self._cholLambda = None
            self._logLambda = None
        elif self.Lambda is None:
            raise ValueError("Lambda must be initialized if update_Lambda is False")

        if self.update_pi:
            N0 = N < self.min_N
            if np.any(N0):
                N[N0] = 0
                self.mu[N0] = 0

            N_sum = np.sum(N)
            if N_sum <= 0:
                raise ValueError(
                    "all components were pruned (sum(N)==0); reduce min_N"
                )
            self.pi = N / N_sum
            self._log_pi = None
        elif self.pi is None:
            raise ValueError("pi must be initialized if update_pi is False")

        self._compute_nat_params()

    def split_comp(self, K: int = 2) -> "GMMTiedDiagCov":
        """Creates a new GMM with ``K * num_comp`` tied components.

        Args:
            K: Number of splits per component.

        Returns:
            New :class:`GMMTiedDiagCov` whose components are split versions.
        """
        std_dev = 1 / self.cholLambda

        num_comp = self.num_comp * K
        pi = np.repeat(self.pi, K) / K
        mu = np.repeat(self.mu, K, axis=0)

        if K == 2:
            mu[::2] += std_dev
            mu[1::2] -= std_dev
        else:
            for k in range(K):
                factor = 2 * (np.random.uniform(size=std_dev.shape) > 0.5) - 1
                mu[k::K] += factor * std_dev

        config = self.get_config()
        return DiagGMMTiedCov(pi=pi, mu=mu, Lambda=self.Lambda, **config)

    def log_prob_std(self, x: np.ndarray) -> np.ndarray:
        """Computes log-likelihoods using the tied standard parameters.

        Args:
            x: Input data with shape ``(num_samples, x_dim)``.

        Returns:
            Log-likelihood per sample.
        """
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            mah_dist2 = np.sum(((x - self.mu[k]) * self.cholLambda) ** 2, axis=-1)
            llk_k[:, k] = r0[k] - 0.5 * mah_dist2
        return logsumexp(llk_k, axis=-1)

    def log_prob_nbest_std(
        self,
        x: np.ndarray,
        nbest_mode: str = "master",
        nbest: NBestType = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes top-N log-likelihoods using tied standard parameters.

        Args:
            x: Input data with shape ``(num_samples, x_dim)``.
            nbest_mode: If ``"master"``, selects top components per sample.
            nbest: Number of top components or explicit component indices.

        Returns:
            If ``nbest_mode == "master"``, returns ``(llk, top_idx)`` where
            ``top_idx`` has shape ``(num_samples, nbest_eff)``. Otherwise returns
            only ``llk``.
        """
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)

        if nbest_mode == "master":
            llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
            for k in range(self.num_comp):
                mah_dist2 = np.sum(((x - self.mu[k]) * self.cholLambda) ** 2, axis=-1)
                llk_k[:, k] = r0[k] - 0.5 * mah_dist2
            assert isinstance(nbest, int)
            assert nbest > 0
            nbest_eff = min(nbest, self.num_comp)
            if nbest_eff < self.num_comp:
                top_idx = np.argpartition(llk_k, -nbest_eff, axis=1)[:, -nbest_eff:]
            else:
                top_idx = np.tile(
                    np.arange(self.num_comp, dtype=np.intp), (x.shape[0], 1)
                )
            llk_sel = np.take_along_axis(llk_k, top_idx, axis=1)
            sort_idx = np.argsort(llk_sel, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(top_idx, sort_idx, axis=1)
            llk_sel = np.take_along_axis(llk_sel, sort_idx, axis=1)
            llk = logsumexp(llk_sel, axis=-1)
            return llk, top_idx

        nbest_idx = np.asarray(nbest, dtype=np.intp)
        if nbest_idx.ndim != 2 or nbest_idx.shape[0] != x.shape[0]:
            raise ValueError(
                "for nbest_mode!='master', nbest must have shape "
                "(num_samples, nbest)"
            )
        delta = x[:, None, :] - self.mu[nbest_idx]
        llk_sel = r0[nbest_idx] - 0.5 * np.sum((delta * self.cholLambda) ** 2, axis=-1)
        llk = logsumexp(llk_sel, axis=-1)
        return llk

    def log_cdf(self, x: np.ndarray) -> np.ndarray:
        """Computes the log CDF of the tied-covariance mixture."""
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            delta = (x - self.mu[k]) * self.cholLambda
            lk = 0.5 * (1 + erf(delta / np.sqrt(2)))
            llk_k[:, k] = self.log_pi[k] + np.sum(np.log(lk + 1e-20), axis=-1)

        return logsumexp(llk_k)

    def sample(
        self,
        num_samples: int = 1,
        rng: Optional[Generator] = None,
        seed: int = 1024,
        r: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draws samples from the GMM.

        Args:
            num_samples: Number of samples to draw when ``r`` is ``None``.
            rng: Optional :class:`numpy.random.Generator`.
            seed: Used to create a generator if ``rng`` is ``None``.
            r: Optional one-hot selections denoting component indices.

        Returns:
            Samples with shape ``(num_samples, x_dim)``.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        if r is None:
            r = rng.multinomial(1, self.pi, size=(num_samples,))
        else:
            num_samples = len(r)
        x = rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())

        for k in range(self.num_comp):
            index = r[:, k] == 1
            x[index] = 1.0 / self.cholLambda * x[index] + self.mu[k]

        return x

    def _validate_Lambda(self) -> None:
        assert self.Lambda.shape[0] == self.x_dim
        assert np.all(self.Lambda > 0)

    @staticmethod
    def compute_eta(mu, Lambda):
        """Computes nat param. from mean and precision."""
        Lmu = Lambda * mu
        eta = np.hstack((Lmu, -0.5 * np.tile(Lambda, (mu.shape[0], 1))))
        return eta

    @staticmethod
    def compute_std(eta):
        """Computes standard params. from the natural param."""
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2[0]
        return mu, Lambda

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        """Plots one slice of each GMM component in 1d.

        Args:
          feat_idx: feature index.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[feat_idx]
        for k in range(mu.shape[0]):
            plot_gaussian_1D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of each GMM component in 2d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_ellipsoid_2D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_3D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 3 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_ellipsoid_3D(mu[k], C, num_sigmas, num_pts, **kwargs)


DiagGMMTiedCov = GMMTiedDiagCov
