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

from .exp_family_mixture import ExpFamilyMixture


class GMMDiagCov(ExpFamilyMixture):
    def __init__(
        self,
        mu=None,
        Lambda=None,
        var_floor=1e-3,
        update_mu=True,
        update_Lambda=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
        self.Lambda = Lambda
        self.var_floor = var_floor
        self.update_mu = update_mu
        self.update_Lambda = update_Lambda

        self._compute_gmm_nat_std()

        self._logLambda = None
        self._cholLambda = None
        self._Sigma = None

    def _compute_gmm_nat_std(self):
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()
            self._compute_nat_params()
        elif self.eta is not None:
            self._validate_eta()
            self.A = self.compute_A_nat(self.eta)
            self._compute_std_params()

    @property
    def logLambda(self):
        if self._logLambda is None:
            self._logLambda = np.sum(np.log(self.Lambda), axis=-1)
        return self._logLambda

    @property
    def cholLambda(self):
        if self._cholLambda is None:
            self._cholLambda = np.sqrt(self.Lambda)
        return self._cholLambda

    @property
    def Sigma(self):
        if self._Sigma is None:
            self._Sigma = 1.0 / self.Lambda
        return self._Sigma

    def initialize(self, x=None):
        if x is None and self.mu is None and self.eta is None:
            assert self.num_comp == 1
            self._initialize_stdnormal()
        if x is not None:
            self._initialize_kmeans(self.num_comp, x)
        self.validate()
        self._compute_gmm_nat_std()

    def _initialize_stdnormal(self):
        self.pi = np.array([1], dtype=float_cpu())
        self.mu = np.zeros((1, self.x_dim), dtype=float_cpu())
        self.Lambda = np.ones((1, self.x_dim), dtype=float_cpu())

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
        self.Lambda = np.zeros((self.num_comp, x.shape[-1]), dtype=float_cpu())
        for k in range(num_comp):
            r = cluster_index == k
            self.pi[k] = np.sum(r) / x.shape[0]
            self.Lambda[k] = 1 / np.std(x[r], axis=0) ** 2

    def stack_suff_stats(self, F, S=None):
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats):
        F = stats[:, : self.x_dim]
        S = stats[:, self.x_dim :]
        return F, S

    def norm_suff_stats(self, N, u_x, return_order2=False):
        F, S = self.unstack_suff_stats(acc_u_x)
        F_norm = self.cholLambda * (F - N[:, None] * self.mu)
        if return_order2:
            S = S - 2 * self.mu * F + N * self.mu ** 2
            S *= self.Lambda
            return N, self.stack_suff_stats(F_norm, S)

        return N, F_norm

    def Mstep(self, N, u_x):

        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N[:, None]

        if self.update_Lambda:
            S = S / N[:, None] - self.mu ** 2
            S_floor = self.var_floor * np.mean(S[N > self.min_N], axis=0)
            S = np.maximum(S, S_floor)
            self.Lambda = 1 / S
            self._Sigma = S
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
        Lambda = np.repeat(self.Lambda, K, axis=0) * (K ** 2)
        mu = np.repeat(self.mu, K, axis=0)

        if K == 2:
            mu[::2] += std_dev
            mu[1::2] -= std_dev
        else:
            for k in range(K):
                factor = 2 * (np.random.uniform(size=std_dev.shape) > 0.5) - 1
                mu[k::K] += factor * std_dev

        config = self.get_config()
        return GMMDiagCov(pi=pi, mu=mu, Lambda=Lambda, **config)

    def log_prob_std(self, x):
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            mah_dist2 = np.sum(((x - self.mu[k]) * self.cholLambda[k]) ** 2, axis=-1)
            llk_k[:, k] = r0[k] - 0.5 * mah_dist2
        return logsumexp(llk_k, axis=-1)

    def log_cdf(self, x):
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            delta = (x - self.mu[k]) * self.cholLambda[k]
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
            x[index] = 1.0 / self.cholLambda[k] * x[index] + self.mu[k]

        return x

    def get_config(self):
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(GMMDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"pi": self.pi, "mu": self.mu, "Lambda": self.Lambda}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["pi", "mu", "Lambda"]
        params = self._load_params_to_dict(f, config["name"], param_list)
        return cls(
            x_dim=config["x_dim"],
            pi=params["pi"],
            mu=params["mu"],
            Lambda=params["Lambda"],
            var_floor=config["var_floor"],
            min_N=config["min_n"],
            update_pi=config["update_pi"],
            update_mu=config["update_mu"],
            update_Lambda=config["update_lambda"],
            name=config["name"],
        )

    @classmethod
    def load_from_kaldi(cls, file_path):
        pi = None
        eta1 = None
        eta2 = None
        num_comp = 0
        x_dim = 0
        success = False
        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                fields = line.rstrip().split()
                if fields[0] == "<WEIGHTS>":
                    pi = np.array([float(v) for v in fields[2:-1]], dtype=float_cpu())
                    num_comp = len(pi)
                elif fields[0] == "<MEANS_INVVARS>":
                    for k in range(num_comp):
                        line = f.readline()
                        fields = line.split()
                        if x_dim == 0:
                            x_dim = len(fields)
                            eta1 = np.zeros((num_comp, x_dim), dtype=float_cpu())
                            eta2 = np.zeros((num_comp, x_dim), dtype=float_cpu())

                        assert len(fields) == x_dim or len(fields) == x_dim + 1
                        eta1[k] = [float(v) for v in fields[:x_dim]]
                elif fields[0] == "<INV_VARS>":
                    for k in range(num_comp):
                        line = f.readline()
                        fields = line.split()
                        assert len(fields) == x_dim or len(fields) == x_dim + 1
                        eta2[k] = [-0.5 * float(v) for v in fields[:x_dim]]
                        if k == num_comp - 1:
                            success = True
        assert success
        eta = np.hstack((eta1, eta2))
        return cls(x_dim=x_dim, pi=pi, eta=eta)

    def _validate_mu(self):
        assert self.mu.shape[0] == self.num_comp
        assert self.mu.shape[1] == self.x_dim

    def _validate_Lambda(self):
        assert self.Lambda.shape[0] == self.num_comp
        assert self.Lambda.shape[1] == self.x_dim
        assert np.all(self.Lambda > 0)

    def _validate_eta(self):
        assert self.eta.shape[0] == self.num_comp
        assert self.eta.shape[1] == self.x_dim * 2

    def validate(self):
        if self.pi is not None:
            self._validate_pi()

        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu, Lambda):
        Lmu = Lambda * mu
        eta = np.hstack((Lmu, -0.5 * Lambda))
        return eta

    @staticmethod
    def compute_std(eta):
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta):
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -1 / 4 * np.sum(eta1 * eta1 / eta2, axis=-1)
        r3 = -1 / 2 * np.sum(np.log(-2 * eta2), axis=-1)
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu, Lambda):
        x_dim = mu.shape[1]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -0.5 * np.sum(np.log(Lambda), axis=-1)
        r3 = 0.5 * np.sum(mu * mu * Lambda, axis=-1)
        return r1 + r2 + r3

    def _compute_nat_params(self):
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_nat(self.eta)

    def _compute_std_params(self):
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x):
        d = x.shape[-1]
        u = np.zeros((x.shape[0], 2 * d), dtype=float_cpu())
        u[:, :d] = x
        u[:, d:] = x * x
        return u

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            plot_gaussian_1D(mu[k], C[k], num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_ellipsoid_2D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_ellipsoid_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)


DiagGMM = GMMDiagCov
