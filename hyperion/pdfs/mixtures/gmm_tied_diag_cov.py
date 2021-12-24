"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py
from scipy.special import erf

from ...hyp_defs import float_cpu
from ...utils.math import softmax, logsumexp
from ...utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
    plot_gaussian_3D,
)
from ...clustering import KMeans

from .gmm_diag_cov import GMMDiagCov


class GMMTiedDiagCov(GMMDiagCov):
    def __init__(
        self,
        mu=None,
        Lambda=None,
        var_floor=1e-3,
        update_mu=True,
        update_Lambda=True,
        **kwargs
    ):
        super().__init__(
            mu=mu,
            Lambda=Lambda,
            var_floor=var_floor,
            update_mu=update_mu,
            update_Lambda=update_Lambda,
            **kwargs
        )

    def _compute_gmm_nat_std(self):
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()
            self._compute_nat_params()
        elif self.eta is not None:
            self._validate_eta()
            self.A = self.compute_A_nat(self.eta)
            self._compute_std_params()

    def _initialize_stdnormal(self):
        self.pi = np.array([1], dtype=float_cpu())
        self.mu = np.zeros((1, self.x_dim), dtype=float_cpu())
        self.Lambda = np.ones((self.x_dim,), dtype=float_cpu())

    def _initialize_kmeans(self, num_comp, x):
        if num_comp == 1:
            self.pi = np.array([1], dtype=float_cpu())
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.Lambda = 1 / np.std(x, axis=0, keepdims=True) ** 2
            return

        kmeans = KMeans(num_clusters=num_comp)
        loss, cluster_index = kmeans.fit(x, epochs=100)

        self.mu = kmeans.mu
        self.pi = np.zeros((self.num_comp,), dtype=float_cpu())
        C = np.zeros((x.shape[-1],), dtype=float_cpu())
        for k in range(num_comp):
            r = cluster_index == k
            self.pi[k] = np.sum(r) / x.shape[0]
            delta = x[r] - self.mu[k]
            C += np.sum(delta ** 2, axis=0)

        self.Lambda = x.shape[0] / C

    def Mstep(self, N, u_x):

        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N[:, None]

        if self.update_Lambda:
            S = S / N[:, None] - self.mu ** 2
            S_floor = self.var_floor * np.mean(S[N > self.min_N], axis=0)
            S = np.maximum(S, S_floor)
            Spool = np.sum(N[:, None] * S, axis=0) / np.sum(N)
            self.Lambda = 1 / Spool
            self._Sigma = Spool
            self._cholLambda = None
            self._logLambda = None

        if self.update_pi:
            N0 = N < self.min_N
            if np.any(N0):
                N[N0] = 0
                mu[N0] = 0
                S[N0] = 1
            self.pi = N / np.sum(N)
            self._log_pi = None

        self._compute_nat_params()

    def split_comp(self, K=2):

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

    def log_prob_std(self, x):
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            mah_dist2 = np.sum(((x - self.mu[k]) * self.cholLambda) ** 2, axis=-1)
            llk_k[:, k] = r0[k] - 0.5 * mah_dist2
        return logsumexp(llk_k, axis=-1)

    def log_cdf(self, x):
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            delta = (x - self.mu[k]) * self.cholLambda
            lk = 0.5 * (1 + erf(delta / np.sqrt(2)))
            llk_k[:, k] = self.log_pi[k] + np.sum(np.log(lk + 1e-20), axis=-1)

        return logsumexp(llk_k)

    def sample(self, num_samples, rng=None, seed=1024):
        if rng is None:
            rng = np.random.RandomState(seed)

        r = rng.multinomial(1, self.pi, size=(num_samples,))
        x = rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())

        for k in range(self.num_comp):
            index = r[:, k] == 1
            x[index] = 1.0 / self.cholLambda * x[index] + self.mu[k]

        return x

    def _validate_Lambda(self):
        assert self.Lambda.shape[0] == self.x_dim
        assert np.all(self.Lambda > 0)

    @staticmethod
    def compute_eta(mu, Lambda):
        Lmu = Lambda * mu
        eta = np.hstack((Lmu, -0.5 * np.tile(Lambda, (mu.shape[0], 1))))
        return eta

    @staticmethod
    def compute_std(eta):
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2[0]
        return mu, Lambda

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[feat_idx]
        for k in range(mu.shape[0]):
            plot_gaussian_1D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_ellipsoid_2D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_3D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = np.diag(1 / self.Lambda[feat_idx])
        for k in range(mu.shape[0]):
            plot_gaussian_ellipsoid_3D(mu[k], C, num_sigmas, num_pts, **kwargs)


DiagGMMTiedCov = GMMTiedDiagCov
