"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py
import scipy.linalg as la
from scipy.special import erf

from ...hyp_defs import float_cpu
from ...utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
    plot_gaussian_3D,
)
from ...utils.math import (
    invert_pdmat,
    invert_trimat,
    symmat2vec,
    vec2symmat,
    fullcov_varfloor,
    logdet_pdmat,
)

from .exp_family import ExpFamily


class Normal(ExpFamily):
    def __init__(
        self,
        mu=None,
        Lambda=None,
        var_floor=1e-5,
        update_mu=True,
        update_Lambda=True,
        **kwargs
    ):
        super(Normal, self).__init__(**kwargs)
        self.mu = mu
        self.Lambda = Lambda
        self.var_floor = var_floor
        self.update_mu = update_mu
        self.update_Lambda = update_Lambda

        self._compute_nat_std()

        self._logLambda = None
        self._cholLambda = None
        self._Sigma = None

    def _compute_nat_std(self):
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
            assert self.is_init
            f, L, logL = invert_pdmat(self.Lambda, return_logdet=True)
            self._logLambda = logL
            self._cholLambda = L.T
        return self._logLambda

    @property
    def cholLambda(self):
        if self._cholLambda is None:
            assert self.is_init
            f, L, logL = invert_pdmat(self.Lambda, return_logdet=True)
            self._logLambda = logL
            self._cholLambda = L.T
        return self._cholLambda

    @property
    def Sigma(self):
        if self._Sigma is None:
            assert self.is_init
            self._Sigma = invert_pdmat(self.Lambda, return_inv=True)[-1]
        return self._Sigma

    def initialize(self):
        self.validate()
        self._compute_nat_std()

    def stack_suff_stats(self, F, S=None):
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats):
        F = stats[: self.x_dim]
        S = stats[self.x_dim :]
        return F, S

    def accum_suff_stats(self, x, u_x=None, sample_weight=None, batch_size=None):
        if u_x is None:
            if sample_weight is None:
                N = x.shape[0]
                F = np.sum(x, axis=0)
                S = symmat2vec(np.dot(x.T, x))
            else:
                N = np.sum(sample_weight)
                wx = sample_weight[:, None] * x
                F = np.sum(wx, axis=0)
                S = symmat2vec(np.dot(wx.T, x))
            return N, self.stack_suff_stats(F, S)
        else:
            return self._accum_suff_stats_1batch(x, u_x, sample_weight)

    def norm_suff_stats(self, N, u_x, return_order2=False):
        assert self.is_init

        F, S = self.unstack_suff_stats(u_x)
        F_norm = np.dot(F - N * self.mu, self.cholLambda.T)
        if return_order2:
            SS = vec2symat(S)
            Fmu = np.outer(self.F, self.mu)
            SS = SS - Fmu - Fmu.T + N * np.outer(self.mu, self.mu)
            SS = np.dot(self.cholLambda, np.dot(SS, self.cholLambda.T))
            S = symmat2vec(SS)
            return N, self.stack_suff_stats(F_norm, S)
        return N, F_norm

    def Mstep(self, N, u_x):

        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N

        if self.update_Lambda:
            S = vec2symmat(S / N)
            S -= np.outer(self.mu, self.mu)
            # S = fullcov_varfloor(S, self.var_floor)
            self.Lambda = invert_pdmat(S, return_inv=True)[-1]
            self._Sigma = None
            self._logLambda = None
            self._cholLambda = None

        self._compute_nat_params()

    def log_prob_std(self, x):
        assert self.is_init
        mah_dist2 = np.sum(np.dot(x - self.mu, self.cholLambda) ** 2, axis=1)
        return (
            0.5 * self.logLambda
            - 0.5 * self.x_dim * np.log(2 * np.pi)
            - 0.5 * mah_dist2
        )

    # def eval_logcdf(self, x):
    #     delta = np.dot((x-self.mu), self.cholLambda)
    #     lk = 0.5*(1+erf(delta/np.sqrt(2)))
    #     print(x-self.mu)
    #     print(la.cholesky(self.Lambda,lower=True))
    #     print(self.cholLambda)
    #     print(delta)
    #     print(lk)
    #     return np.sum(np.log(lk+1e-20), axis=-1)

    def sample(self, num_samples, rng=None, seed=1024):
        assert self.is_init

        if rng is None:
            rng = np.random.RandomState(seed)
        return rng.multivariate_normal(self.mu, self.Sigma, size=(num_samples,)).astype(
            float_cpu()
        )
        # x=rng.normal(size=(num_samples, self.x_dim))
        # cholS=la.cholesky(self.Sigma, lower=False, overwrite_a=True)
        # return self.mu+np.dot(x, cholS)

    def get_config(self):
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(Normal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):

        assert self.is_init

        params = {"mu": self.mu, "Lambda": self.Lambda}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "Lambda"]
        params = self._load_params_to_dict(f, config["name"], param_list)
        return cls(
            x_dim=config["x_dim"],
            mu=params["mu"],
            Lambda=params["Lambda"],
            var_floor=config["var_floor"],
            update_mu=config["update_mu"],
            update_Lambda=config["update_lambda"],
            name=config["name"],
        )

    def _validate_mu(self):
        assert self.mu.shape[0] == self.x_dim

    def _validate_Lambda(self):
        assert self.Lambda.shape == (self.x_dim, self.x_dim)

    def _validate_eta(self):
        assert self.eta.shape[0] == (self.x_dim ** 2 + 3 * self.x_dim) / 2

    def validate(self):
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu, Lambda):
        Lmu = np.dot(mu, Lambda)
        eta = np.hstack((Lmu, -symmat2vec(Lambda, diag_factor=0.5)))
        return eta

    @staticmethod
    def compute_x_dim_from_eta(eta):
        x_dim = 0.5 * (-3 + np.sqrt(9 + 8 * eta.shape[-1]))
        assert int(x_dim) == x_dim
        return int(x_dim)

    @staticmethod
    def compute_std(eta):
        x_dim = Normal.compute_x_dim_from_eta(eta)
        eta1 = eta[:x_dim]
        eta2 = vec2symmat(eta[x_dim:], diag_factor=2) / 2
        Lambda = -2 * eta2
        f = invert_pdmat(-eta2, right_inv=True)[0]
        mu = 0.5 * f(eta1)
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta):
        x_dim = Normal.compute_x_dim_from_eta(eta)
        eta1 = eta[:x_dim]
        eta2 = vec2symmat(eta[x_dim:], diag_factor=2) / 2
        f, _, log_minus_eta2 = invert_pdmat(-eta2, right_inv=True, return_logdet=True)
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = 0.25 * np.inner(f(eta1), eta1)
        r3 = -0.5 * x_dim * np.log(2) - 0.5 * log_minus_eta2
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu, Lambda):
        x_dim = mu.shape[0]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -0.5 * logdet_pdmat(Lambda)
        r3 = 0.5 * np.inner(np.dot(mu, Lambda), mu)
        return r1 + r2 + r3

    def _compute_nat_params(self):
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_std(self.mu, self.Lambda)
        # self.A = self.compute_A_nat(self.eta)
        # Lmu = np.dot(self.Lambda, self.mu[:, None])
        # muLmu = np.dot(self.mu, Lmu)
        # lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        # Lambda=np.copy(self.Lambda)
        # Lambda[np.diag_indices(self.x_dim)] /= 2
        # self.eta=np.vstack((lnr, Lmu, symmat2vec(Lambda)[:, None]))

    def _compute_std_params(self):
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x):
        d = x.shape[1]
        u = np.zeros((x.shape[0], int(d + d * (d + 1) / 2)), dtype=float_cpu())
        u[:, :d] = x
        k = d
        for i in range(d):
            for j in range(i, d):
                u[:, k] = x[:, i] * x[:, j]
                k += 1
        return u

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        assert self.is_init
        mu = self.mu[feat_idx]
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][feat_idx, feat_idx]
        plot_gaussian_1D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
        plot_gaussian_ellipsoid_2D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
        plot_gaussian_3D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
        plot_gaussian_ellipsoid_3D(mu, C, num_sigmas, num_pts, **kwargs)
