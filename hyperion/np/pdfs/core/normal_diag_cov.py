"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from scipy.special import erf

from ....hyp_defs import float_cpu
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
    plot_gaussian_3D,
)

from .exp_family import ExpFamily


class NormalDiagCov(ExpFamily):
    """Class for Normal distribution with diagonal covariance.

    Attributes:
      mu: mean with shape (x_dim,) or None.
      Lambda: precision with shape (x_dim, x_dim) or None.
      var_floor: variance floor.
      update_mu: whether or not update mu when optimizing.
      update_Lambda: wether or not update Lambda when optimizing.
      x_dim: data dim (infered from mu if present)
    """

    def __init__(
        self,
        mu=None,
        Lambda=None,
        var_floor=1e-5,
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

        self._compute_nat_std()

        self._logLambda = None
        self._cholLambda = None
        self._Sigma = None

    def _compute_nat_std(self):
        """Comptues natural and standard parameters of the distribution."""
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
        """log precision determinant."""
        if self._logLambda is None:
            assert self.is_init
            self._logLambda = np.sum(np.log(self.Lambda))
        return self._logLambda

    @property
    def cholLambda(self):
        """Square root of precision."""
        if self._cholLambda is None:
            assert self.is_init
            self._cholLambda = np.sqrt(self.Lambda)
        return self._cholLambda

    @property
    def Sigma(self):
        "Variance of the distribution."
        if self._Sigma is None:
            assert self.is_init
            self._Sigma = 1.0 / self.Lambda
        return self._Sigma

    def initialize(self):
        """Initializes the distribution."""
        self.validate()
        self._compute_nat_std()
        assert self.is_init

    def stack_suff_stats(self, F, S=None):
        """Stacks F and S suff stats into single vector."""

        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats):
        """Decomposes suff. stats vector into F and S."""
        F = stats[: self.x_dim]
        S = stats[self.x_dim :]
        return F, S

    def norm_suff_stats(self, N, u_x=None, return_order2=False):
        """Normalizes accumlated sufficient statistics with the
        mean and covariance of the distribution.

        Args:
          N: zeroth order sufficient stats.
          u_x: 1st and 2nd order stats.
          return_order2: whether or not return normalizes 2nd order stats.

        Return:
          Normalized N, F or N, [F, S].
        """
        assert self.is_init
        F, S = self.unstack_suff_stats(u_x)
        F_norm = self.cholLambda * (F - N * self.mu)
        if return_order2:
            S = S - 2 * self.mu * F + N * self.mu ** 2
            S *= self.Lambda
            return N, self.stack_suff_stats(F_norm, S)
        return N, F_norm

    def Mstep(self, N, u_x):
        """Maximization step.

        Args:
          N: zeroth order stats.
          u_x: accumlated higher order stats.

        """
        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N

        if self.update_Lambda:
            S = S / N - self.mu ** 2
            S[S < self.var_floor] = self.var_floor
            self.Lambda = 1 / S
            self._Sigma = S
            self._cholLambda = None
            self._logLambda = None

        self._compute_nat_params()

    def log_prob_std(self, x):
        """log p(x) of each data sample computed using the
        standard parameters of the distribution.

        Args:
          x: input data with shape (num_samples, x_dim).

        Returns:
          log p(x) with shape (num_samples,)
        """
        assert self.is_init
        mah_dist2 = np.sum(((x - self.mu) * self.cholLambda) ** 2, axis=1)
        return (
            0.5 * self.logLambda
            - 0.5 * self.x_dim * np.log(2 * np.pi)
            - 0.5 * mah_dist2
        )

    def log_cdf(self, x):
        """Log cumulative distribution function."""
        assert self.is_init
        delta = (x - self.mu) * self.cholLambda
        lk = 0.5 * (1 + erf(delta / np.sqrt(2)))
        return np.sum(np.log(lk + 1e-10), axis=-1)

    def sample(self, num_samples, rng=None, seed=1024):
        """Draws samples from the data distribution.

        Args:
          num_samples: number of samples.
          rng: random number generator.
          seed: random seed used if rng is None.

        Returns:
          Generated samples with shape (num_samples, x_dim).
        """
        assert self.is_init
        if rng is None:
            rng = np.random.RandomState(seed)
        x = rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())
        return self.mu + 1.0 / self.cholLambda * x

    def get_config(self):
        """Returns the model configuration dict."""
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(NormalDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        assert self.is_init
        params = {"mu": self.mu, "Lambda": self.Lambda}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "Lambda"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
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
        assert self.Lambda.shape[0] == self.x_dim
        assert np.all(self.Lambda > 0)

    def _validate_eta(self):
        assert self.eta.shape[0] == self.x_dim * 2

    def validate(self):
        """Validates the parameters of the distribution."""
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu, Lambda):
        """Computes nat param. from mean and precision."""
        Lmu = Lambda * mu
        eta = np.hstack((Lmu, -0.5 * Lambda))
        return eta

    @staticmethod
    def compute_std(eta):
        """Computes standard params. from the natural param."""
        x_dim = int(eta.shape[0] / 2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta):
        """Computes A from the natural param."""
        x_dim = int(eta.shape[0] / 2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -1 / 4 * np.sum(eta1 * eta1 / eta2)
        r3 = -1 / 2 * np.sum(np.log(-2 * eta2))
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu, Lambda):
        """Computes A from the standard params."""
        x_dim = mu.shape[0]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -0.5 * np.sum(np.log(Lambda))
        r3 = 0.5 * np.sum(mu * mu * Lambda)
        return r1 + r2 + r3

    def _compute_nat_params(self):
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_nat(self.eta)
        # Lmu = self.Lambda*self.mu
        # muLmu = np.sum(self.mu*Lmu)
        # lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        # self.eta=np.hstack((lnr, Lmu, -0.5*self.Lambda)).T

    def _compute_std_params(self):
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x):
        """Computes the sufficient stats. for each sample.

        Args:
          x: data samples with shape (num_samples, x_dim).

        Returns:
          Sufficient stats. for each data sample with shape (num_samples, u_dim).
        """
        d = x.shape[1]
        u = np.zeros((x.shape[0], 2 * d), dtype=float_cpu())
        u[:, :d] = x
        u[:, d:] = x * x
        return u

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        """Plots one slice of the Gaussian in 1d.

        Args:
          feat_idx: feature index.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[feat_idx]
        C = 1 / self.Lambda[feat_idx]
        plot_gaussian_1D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of the Gaussian in 2d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_2D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of the Gaussian in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_3D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 3 dimensions of the Gaussian in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_3D(mu, C, num_sigmas, num_pts, **kwargs)


DiagNormal = NormalDiagCov
