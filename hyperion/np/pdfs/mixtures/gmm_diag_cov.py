"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py
from scipy.special import erf

from ....hyp_defs import float_cpu
from ....utils.math import softmax, logsumexp
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
    plot_gaussian_3D,
)
from ...clustering import KMeans

from .exp_family_mixture import ExpFamilyMixture


class GMMDiagCov(ExpFamilyMixture):
    """Class for GMM with diagonal covariance.

    Attributes:
      num_comp: number of components of the mixture (intered from pi).
      pi: weights of the components.
      mu: mean with shape (num_comp, x_dim,) or None.
      Lambda: precision with shape (num_comp, x_dim, x_dim) or None.
      var_floor: variance floor.
      update_mu: whether or not update mu when optimizing.
      update_Lambda: wether or not update Lambda when optimizing.
      x_dim: data dim (infered from mu if present)
    """

    def __init__(
        self,
        num_comp=1,
        pi=None,
        mu=None,
        Lambda=None,
        var_floor=1e-3,
        update_mu=True,
        update_Lambda=True,
        **kwargs
    ):
        if mu is not None:
            assert mu.ndim == 2
            kwargs["x_dim"] = mu.shape[1]

        super().__init__(num_comp=num_comp, pi=pi, **kwargs)
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
        """log precision determinants."""
        if self._logLambda is None:
            self._logLambda = np.sum(np.log(self.Lambda), axis=-1)
        return self._logLambda

    @property
    def cholLambda(self):
        """Cholesqy decomp. of the precisions."""
        if self._cholLambda is None:
            self._cholLambda = np.sqrt(self.Lambda)
        return self._cholLambda

    @property
    def Sigma(self):
        """Covariances."""
        if self._Sigma is None:
            self._Sigma = 1.0 / self.Lambda
        return self._Sigma

    def initialize(self, x=None):
        """Initializes the distribution."""
        if x is None and self.mu is None and self.eta is None:
            assert self.num_comp == 1
            self._initialize_stdnormal()
        if x is not None:
            self._initialize_kmeans(self.num_comp, x)
        self.validate()
        self._compute_gmm_nat_std()

    def _initialize_stdnormal(self):
        """Initializes a single component GMM with std. Normal."""
        self.pi = np.array([1], dtype=float_cpu())
        self.mu = np.zeros((1, self.x_dim), dtype=float_cpu())
        self.Lambda = np.ones((1, self.x_dim), dtype=float_cpu())

    def _initialize_kmeans(self, num_comp, x):
        """Initializes the GMM with K-Means.

        Args:
          num_comp: number of components.
          x: initialization data with shape (num_samples, x_dim).
        """
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
        """Stacks F and S suff stats into single vector."""
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats):
        """Decomposes suff. stats vector into F and S."""
        F = stats[:, : self.x_dim]
        S = stats[:, self.x_dim :]
        return F, S

    def norm_suff_stats(self, N, u_x, return_order2=False):
        """Normalizes accumlated sufficient statistics with the
        mean and covariance of the distribution.

        Args:
          N: zeroth order sufficient stats.
          u_x: 1st and 2nd order stats.
          return_order2: whether or not return normalized 2nd order stats.

        Return:
          Normalized N, F or N, [F, S].
        """
        F, S = self.unstack_suff_stats(u_x)
        F_norm = self.cholLambda * (F - N[:, None] * self.mu)
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
                self.mu[N0] = 0
                self._Sigma[N0] = 1
                self.Lambda[N0] = 1
            self.pi = N / np.sum(N)
            self._log_pi = None

        self._compute_nat_params()

    def split_comp(self, K=2):
        """Creates a new GMM with K x num_componentes.

        Args:
          K: multiplier for the number of components

        Returns:
          GMMDiagConv object.
        """
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
        """log p(x) of each data sample computed using the
        standard parameters of the distribution.

        Args:
          x: input data with shape (num_samples, x_dim).

        Returns:
          log p(x) with shape (num_samples,)
        """
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            mah_dist2 = np.sum(((x - self.mu[k]) * self.cholLambda[k]) ** 2, axis=-1)
            llk_k[:, k] = r0[k] - 0.5 * mah_dist2
        return logsumexp(llk_k, axis=-1)

    def log_cdf(self, x):
        """Log cumulative distribution function."""
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            delta = (x - self.mu[k]) * self.cholLambda[k]
            lk = 0.5 * (1 + erf(delta / np.sqrt(2)))
            llk_k[:, k] = self.log_pi[k] + np.sum(np.log(lk + 1e-20), axis=-1)

        return logsumexp(llk_k)

    def sample(self, num_samples=1, rng=None, seed=1024, r=None):
        """Draws samples from the data distribution.

        Args:
          num_samples: number of samples.
          rng: random number generator.
          seed: random seed used if rng is None.

        Returns:
          Generated samples with shape (num_samples, x_dim).
        """
        if rng is None:
            rng = np.random.RandomState(seed)

        if r is None:
            r = rng.multinomial(1, self.pi, size=(num_samples,))
        else:
            num_samples = len(r)
        x = rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())

        for k in range(self.num_comp):
            index = r[:, k] == 1
            x[index] = 1.0 / self.cholLambda[k] * x[index] + self.mu[k]

        return x

    def get_config(self):
        """Returns the model configuration dict."""
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(GMMDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"pi": self.pi, "mu": self.mu, "Lambda": self.Lambda}
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
        param_list = ["pi", "mu", "Lambda"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
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
        """Loads GMM from Kaldi file.

        Args:
          file_path: kaldi file path.

        Returns:
          Model object.
        """
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
        """Validates the parameters of the distribution."""
        if self.pi is not None:
            self._validate_pi()

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
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta):
        """Computes A from the natural param."""
        x_dim = int(eta.shape[-1] / 2)
        eta1 = eta[:, :x_dim]
        eta2 = eta[:, x_dim:]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -1 / 4 * np.sum(eta1 * eta1 / eta2, axis=-1)
        r3 = -1 / 2 * np.sum(np.log(-2 * eta2), axis=-1)
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu, Lambda):
        """Computes A from the standard params."""
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
        """Computes the sufficient stats. for each sample.

        Args:
          x: data samples with shape (num_samples, x_dim).

        Returns:
          Sufficient stats. for each data sample with shape (num_samples, u_dim).
        """
        d = x.shape[-1]
        u = np.zeros((x.shape[0], 2 * d), dtype=float_cpu())
        u[:, :d] = x
        u[:, d:] = x * x
        return u

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        """Plots one slice of each GMM component in 1d.

        Args:
          feat_idx: feature index.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            plot_gaussian_1D(mu[k], C[k], num_sigmas, num_pts, **kwargs)

    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of each GMM component in 2d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_ellipsoid_2D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 2 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100, **kwargs):
        """Plots 3 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        C = 1 / self.Lambda[:, feat_idx]
        for k in range(mu.shape[0]):
            C_k = np.diag(C[k])
            plot_gaussian_ellipsoid_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)


DiagGMM = GMMDiagCov
