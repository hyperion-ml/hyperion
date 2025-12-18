"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import scipy.linalg as la
from numpy.random import Generator, RandomState
from scipy.special import erf

from ....hyp_defs import float_cpu
from ....utils.math_funcs import (
    fullcov_varfloor,
    invert_pdmat,
    logsumexp,
    symmat2vec,
    vec2symmat,
)
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_3D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
)
from ...clustering import KMeans
from ..core import Normal
from .exp_family_mixture import ExpFamilyMixture


class GMM(ExpFamilyMixture):
    """Gaussian mixture model with full covariance matrices.

    Attributes:
        num_comp: Number of mixture components.
        pi: Mixture weights.
        mu: Component means with shape ``(num_comp, x_dim)``.
        Lambda: Precision matrices with shape ``(num_comp, x_dim, x_dim)``.
        var_floor: Variance flooring constant.
        update_mu: Whether to update ``mu`` during EM.
        update_Lambda: Whether to update precision matrices during EM.
        x_dim: Data dimensionality inferred from ``mu`` if provided.
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
        """Initializes the full-covariance GMM.

        Args:
            num_comp: Number of components when ``pi`` is not provided.
            pi: Optional mixture weights.
            mu: Optional component means.
            Lambda: Optional precision matrices.
            var_floor: Minimum variance allowed during updates.
            update_mu: Whether to update ``mu`` during EM.
            update_Lambda: Whether to update precision matrices during EM.
            **kwargs: Extra keyword arguments forwarded to :class:`ExpFamilyMixture`.
        """
        if mu is not None:
            assert mu.ndim == 2
            kwargs["x_dim"] = mu.shape[1]
        super().__init__(num_comp=num_comp, pi=pi, **kwargs)
        self.mu: Optional[np.ndarray] = mu
        self.Lambda: Optional[np.ndarray] = Lambda
        self.var_floor: float = var_floor
        self.update_mu: bool = update_mu
        self.update_Lambda: bool = update_Lambda

        self._compute_gmm_nat_std()

        self._logLambda: Optional[np.ndarray] = None
        self._cholLambda: Optional[np.ndarray] = None
        self._Sigma: Optional[np.ndarray] = None

    def _compute_gmm_nat_std(self) -> None:
        """Keeps natural and standard parameters synchronized."""
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()
            self._compute_nat_params()
        elif self.eta is not None:
            self._validate_eta()
            self.A = self.compute_A_nat(self.eta)
            self._compute_std_params()

    def compute_Lambda_aux(self) -> None:
        """Computes auxiliary quantities derived from the precision matrices."""
        self._logLambda = np.zeros((self.num_comp,), dtype=float_cpu())
        self._cholLambda = np.zeros(
            (self.num_comp, self.x_dim, self.x_dim), dtype=float_cpu()
        )
        for i, L in enumerate(self.Lambda):
            f, L, logL = invert_pdmat(L, return_logdet=True)
            self._logLambda[i] = logL
            self._cholLambda[i] = L.T

    @property
    def logLambda(self) -> np.ndarray:
        """np.ndarray: Log-determinants of the precision matrices."""
        if self._logLambda is None:
            self.compute_Lambda_aux()
        return self._logLambda

    @property
    def cholLambda(self) -> np.ndarray:
        """np.ndarray: Cholesky factors (upper-triangular) of the precisions."""
        if self._cholLambda is None:
            self.compute_Lambda_aux()
        return self._cholLambda

    @property
    def Sigma(self) -> np.ndarray:
        """np.ndarray: Covariance matrices for each component."""
        if self._Sigma is None:
            self._Sigma = np.zeros(
                (self.num_comp, self.x_dim, self.x_dim), dtype=float_cpu()
            )
            for k in range(self.num_comp):
                self._Sigma[k] = invert_pdmat(self.Lambda[k], return_inv=True)[-1]
        return self._Sigma

    def initialize(self, x: Optional[np.ndarray] = None) -> None:
        """Initializes the GMM parameters.

        Args:
            x: Optional data used for k-means initialization.
        """
        if x is None and self.mu is None and self.eta is None:
            assert self.num_comp == 1
            self._initialize_stdnormal()
        if x is not None:
            self._initialize_kmeans(self.num_comp, x)
        self.validate()
        self._compute_gmm_nat_std()

    def _initialize_stdnormal(self) -> None:
        """Initializes a single-component standard normal GMM."""
        self.pi = np.array([1], dtype=float_cpu())
        self.mu = np.zeros((1, self.x_dim), dtype=float_cpu())
        self.Lambda = np.zeros((1, self.x_dim, self.x_dim), dtype=float_cpu())
        self.Lambda[0] = np.eye(self.x_dim, dtype=float_cpu())

    def _initialize_kmeans(self, num_comp: int, x: np.ndarray) -> None:
        """Initializes the GMM using K-means centroids.

        Args:
            num_comp: Number of components to initialize.
            x: Initialization data with shape ``(num_samples, x_dim)``.
        """
        if num_comp == 1:
            self.pi = np.array([1], dtype=float_cpu())
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.Lambda = np.zeros((1, self.x_dim, self.x_dim), dtype=float_cpu())
            delta = x - self.mu
            S = np.dot(delta.T, delta) / x.shape[0]
            self.Lambda[0] = invert_pdmat(S, return_inv=True)[-1]
            return

        kmeans = KMeans(num_clusters=num_comp, epochs=100)
        loss, cluster_index = kmeans.fit(x)

        self.mu = kmeans.mu
        self.pi = np.zeros((self.num_comp,), dtype=float_cpu())
        self.Lambda = np.zeros(
            (self.num_comp, self.x_dim, self.x_dim), dtype=float_cpu()
        )

        for k in range(num_comp):
            r = cluster_index == k
            self.pi[k] = np.sum(r) / x.shape[0]
            delta = x[r] - self.mu[k]
            S = np.dot(delta.T, delta) / np.sum(r)
            self.Lambda[k] = invert_pdmat(S, return_inv=True)[-1]

    def stack_suff_stats(
        self, F: np.ndarray, S: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Stacks first- and second-order stats into a single vector."""
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Splits stacked statistics into first- and second-order parts."""
        F = stats[:, : self.x_dim]
        S = stats[:, self.x_dim :]
        return F, S

    def norm_suff_stats(
        self, N: np.ndarray, u_x: np.ndarray, return_order2: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Whitens accumulated sufficient statistics.

        Args:
            N: Zeroth-order statistics per component.
            u_x: Stacked first- and second-order statistics.
            return_order2: Whether to also return normalized second-order stats.

        Returns:
            Tuple containing the original ``N`` and normalized statistics.
        """
        F, S = self.unstack_suff_stats(u_x)
        F_norm = F - N[:, None] * self.mu
        for k in range(self.num_comp):
            F_norm[k] = np.dot(F_norm[k], self.cholLambda[k].T)
            if return_order2:
                SS = vec2symmat(S[k])
                Fmu = np.outer(self.F[k], self.mu[k])
                SS = SS - Fmu - Fmu.T + N * np.outer(self.mu[k], self.mu[k])
                SS = np.dot(self.cholLambda[k], np.dot(SS, self.cholLambda[k].T))
                S[k] = symmat2vec(SS)
        if return_order2:
            return N, self.stack_suff_stats(F_norm, S)
        return N, F_norm

    def Mstep(self, N: np.ndarray, u_x: np.ndarray) -> None:
        """Maximization step updating mean, covariance, and weights.

        Args:
            N: Zeroth-order statistics per component.
            u_x: Stacked first- and second-order statistics.
        """
        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N[:, None]

        if self.update_Lambda:
            C = np.zeros((self.num_comp, self.x_dim, self.x_dim), dtype=float_cpu())
            for k in range(self.num_comp):
                C[k] = vec2symmat(S[k] / N[k])
                C[k] -= np.outer(self.mu[k], self.mu[k])
            Sfloor = self.var_floor * np.mean(C, axis=0)
            cholfloor = la.cholesky(Sfloor, overwrite_a=True)
            for k in range(self.num_comp):
                C[k] = fullcov_varfloor(C[k], cholfloor, F_is_chol=True)
                self.Lambda[k] = invert_pdmat(C[k], return_inv=True)[-1]
            self._Sigma = None
            self._logLambda = None
            self._cholLambda = None

        if self.update_pi:
            N0 = N < self.min_N
            if np.any(N0):
                N[N0] = 0
                self.mu[N0] = 0
                self.Lambda[N0] = np.eye(self.x_dim)
            self.pi = N / np.sum(N)
            self._log_pi = None

        self._compute_nat_params()

    def split_comp(self, K: int = 2) -> "GMM":
        """Creates a new GMM whose components are split versions of the current ones.

        Args:
            K: Number of splits per component.

        Returns:
            New :class:`GMM` instance with ``K * num_comp`` components.
        """
        num_comp = self.num_comp * K
        pi = np.repeat(self.pi, K) / K
        Lambda = np.repeat(self.Lambda, K, axis=0) * (K**2)
        mu = np.repeat(self.mu, K, axis=0)

        for g in range(self.num_comp):
            w, v = la.eigh(self.Sigma[g])
            v *= np.sqrt(v)
            if K == 2:
                std_dev = np.sum(v, axis=1)
                mu[2 * g] += std_dev
                mu[2 * g + 1] -= std_dev
            else:
                for k in range(K):
                    factor = 2 * (np.random.uniform(size=(v.shape[1],)) > 0.5) - 1
                    std_dev = np.sum(v * factor, axis=1)
                    mu[K * g + k] += std_dev

        config = self.get_config()
        return GMM(pi=pi, mu=mu, Lambda=Lambda, **config)

    def log_prob_std(self, x: np.ndarray) -> np.ndarray:
        """Computes log-likelihoods using the standard parameters.

        Args:
            x: Input data with shape ``(num_samples, x_dim)``.

        Returns:
            Array of log-likelihoods with shape ``(num_samples,)``.
        """
        r0 = self.log_pi + 0.5 * self.logLambda - 0.5 * self.x_dim * np.log(2 * np.pi)
        llk_k = np.zeros((x.shape[0], self.num_comp), dtype=float_cpu())
        for k in range(self.num_comp):
            mah_dist2 = np.sum(np.dot(x - self.mu[k], self.cholLambda[k]) ** 2, axis=1)
            llk_k[:, k] = r0[k] - 0.5 * mah_dist2

        return logsumexp(llk_k, axis=-1)

    def sample(
        self,
        num_samples: int,
        rng: Optional[Union[Generator, RandomState]] = None,
        seed: int = 1024,
        r: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draws samples from the GMM.

        Args:
            num_samples: Number of samples to generate when ``r`` is ``None``.
            rng: Optional random number generator.
            seed: Used to create a generator if ``rng`` is ``None``.
            r: Optional one-hot selections specifying components.

        Returns:
            Samples with shape ``(num_samples, x_dim)``.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        if r is None:
            r = rng.multinomial(1, self.pi, size=(num_samples,))
        else:
            num_samples = len(r)

        x = np.zeros((num_samples, self.x_dim), dtype=float_cpu())
        for k in range(self.num_comp):
            index = r[:, k] == 1
            n_k = np.sum(index)
            if n_k == 0:
                continue
            x[index] = rng.multivariate_normal(
                self.mu[k], self.Sigma[k], size=(n_k,)
            ).astype(float_cpu())

        return x

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary."""
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: h5py.File) -> None:
        """Saves the model parameters into an HDF5 handle.

        Args:
            f: File handle opened by :mod:`h5py`.
        """
        params = {"pi": self.pi, "mu": self.mu, "Lambda": self.Lambda}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "GMM":
        """Loads parameters from file and constructs a :class:`GMM`.

        Args:
            f: File handle pointing to the serialized model.
            config: Configuration dictionary produced by :meth:`get_config`.

        Returns:
            Fully initialized :class:`GMM`.
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
    def load_from_kaldi(cls, file_path: str) -> "GMM":
        """Loads a GMM from a Kaldi-formatted file."""
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
                elif fields[0] == "<MEANS_INVCOVARS>":
                    for k in range(num_comp):
                        line = f.readline()
                        fields = line.split()
                        if x_dim == 0:
                            x_dim = len(fields)
                            eta1 = np.zeros((num_comp, x_dim), dtype=float_cpu())
                            eta2 = np.zeros(
                                (num_comp, int((x_dim**2 + 3 * x_dim) / 2)),
                                dtype=float_cpu(),
                            )

                        assert len(fields) == x_dim or len(fields) == x_dim + 1
                        eta1[k] = [float(v) for v in fields[:x_dim]]
                elif fields[0] == "<INV_COVARS>":
                    L = np.zeros((x_dim, x_dim), dtype=float_cpu())
                    for k in range(num_comp):
                        L[:, :] = 0
                        for j in range(x_dim):
                            line = f.readline()
                            fields = line.split()
                            if j < x_dim - 1:
                                assert len(fields) == j + 1
                            else:
                                assert len(fields) == x_dim + 1
                            L[j, : j + 1] = [float(v) for v in fields[: j + 1]]
                        eta2[k] = -symmat2vec(L.T, diag_factor=0.5)
                        if k == num_comp - 1:
                            success = True
        assert success
        eta = np.hstack((eta1, eta2))
        return cls(x_dim=x_dim, pi=pi, eta=eta)

    def _validate_mu(self) -> None:
        assert self.mu.shape[0] == self.num_comp
        assert self.mu.shape[1] == self.x_dim

    def _validate_Lambda(self) -> None:
        assert self.Lambda.shape[0] == self.num_comp
        assert self.Lambda.shape[1] == self.x_dim
        assert self.Lambda.shape[2] == self.x_dim

    def _validate_eta(self) -> None:
        assert self.eta.shape[0] == self.num_comp
        assert self.eta.shape[1] == (self.x_dim**2 + 3 * self.x_dim) / 2

    def validate(self) -> None:
        """Validates that parameters have consistent shapes."""
        if self.pi is not None:
            self._validate_pi()

        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Computes natural parameters from mean and precision."""
        x_dim = mu.shape[-1]
        eta_dim = int((x_dim**2 + 3 * x_dim) / 2)
        eta = np.zeros((mu.shape[0], eta_dim), dtype=float_cpu())
        for k in range(mu.shape[0]):
            eta[k] = Normal.compute_eta(mu[k], Lambda[k])

        return eta

    @staticmethod
    def compute_std(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes standard parameters from natural parameters."""
        x_dim = Normal.compute_x_dim_from_eta(eta)
        mu = np.zeros((eta.shape[0], x_dim), dtype=float_cpu())
        Lambda = np.zeros((eta.shape[0], x_dim, x_dim), dtype="float32")
        for k in range(eta.shape[0]):
            mu[k], Lambda[k] = Normal.compute_std(eta[k])

        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta: np.ndarray) -> np.ndarray:
        """Computes the log-normalizer ``A`` from natural parameters."""
        A = np.zeros((eta.shape[0],), dtype=float_cpu())
        for k in range(eta.shape[0]):
            A[k] = Normal.compute_A_nat(eta[k])

        return A

    @staticmethod
    def compute_A_std(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Computes the log-normalizer ``A`` from standard parameters."""
        A = np.zeros((mu.shape[0],), dtype=float_cpu())
        for k in range(mu.shape[0]):
            A[k] = Normal.compute_A_std(mu[k], Lambda[k])

        return A

    def _compute_nat_params(self):
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_nat(self.eta)

    def _compute_std_params(self):
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x: np.ndarray) -> np.ndarray:
        """Computes sufficient statistics for each sample."""
        d = x.shape[1]
        u = np.zeros((x.shape[0], int(d + d * (d + 1) / 2)), dtype=float_cpu())
        u[:, :d] = x
        k = d
        for i in range(d):
            for j in range(i, d):
                u[:, k] = x[:, i] * x[:, j]
                k += 1
        return u

    def plot1D(
        self, feat_idx: int = 0, num_sigmas: int = 2, num_pts: int = 100, **kwargs: Any
    ) -> None:
        """Plots one slice of each GMM component in 1d.

        Args:
          feat_idx: feature index.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        for k in range(mu.shape[0]):
            C = invert_pdmat(self.Lambda[k], return_inv=True)[-1][feat_idx, feat_idx]
            plot_gaussian_1D(mu[k], C, num_sigmas, num_pts, **kwargs)

    def plot2D(
        self,
        feat_idx: Sequence[int] = (0, 1),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots 2 dimensions of each GMM component in 2d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        for k in range(mu.shape[0]):
            C_k = invert_pdmat(self.Lambda[k], return_inv=True)[-1][i, j]
            plot_gaussian_ellipsoid_2D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D(
        self,
        feat_idx: Sequence[int] = (0, 1),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots 2 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        for k in range(mu.shape[0]):
            C_k = invert_pdmat(self.Lambda[k], return_inv=True)[-1][i, j]
            plot_gaussian_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(
        self,
        feat_idx: Sequence[int] = (0, 1, 2),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots 3 dimensions of each GMM component in 3d.

        Args:
          feat_idx: feature indeces.
          num_sigmas: size of the plot in number of standard devs.
          num_pts: number of points in the graph.
          **kwargs: pyplot options.
        """
        mu = self.mu[:, feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        for k in range(mu.shape[0]):
            C_k = invert_pdmat(self.Lambda[k], return_inv=True)[-1][i, j]
            plot_gaussian_ellipsoid_3D(mu[k], C_k, num_sigmas, num_pts, **kwargs)
