"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import Generator, RandomState
from scipy.special import erf

from ....hyp_defs import float_cpu
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_3D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
)
from .exp_family import ExpFamily


class NormalDiagCov(ExpFamily):
    """Multivariate normal distribution constrained to diagonal covariance.

    The model keeps both the standard parameters ``(mu, Lambda)`` and the natural
    parameters of the exponential family representation synchronized so it can
    participate in EM-style training procedures.

    Attributes:
        mu: Optional mean vector of shape ``(x_dim,)``.
        Lambda: Optional diagonal precision matrix stored as a vector of shape
            ``(x_dim,)``.
        var_floor: Minimum variance allowed during re-estimation.
        update_mu: Whether ``mu`` is updated during :meth:`Mstep`.
        update_Lambda: Whether ``Lambda`` is updated during :meth:`Mstep`.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        Lambda: Optional[np.ndarray] = None,
        var_floor: float = 1e-5,
        update_mu: bool = True,
        update_Lambda: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes a diagonal-covariance Gaussian.

        Args:
            mu: Optional mean vector used as initialization.
            Lambda: Optional diagonal precision entries.
            var_floor: Minimum variance enforced during updates.
            update_mu: Whether :meth:`Mstep` adjusts ``mu``.
            update_Lambda: Whether :meth:`Mstep` adjusts ``Lambda``.
            **kwargs: Additional keyword arguments forwarded to :class:`ExpFamily`.
        """
        super().__init__(**kwargs)
        self.mu: Optional[np.ndarray] = mu
        self.Lambda: Optional[np.ndarray] = Lambda
        self.var_floor: float = var_floor
        self.update_mu: bool = update_mu
        self.update_Lambda: bool = update_Lambda

        self._compute_nat_std()

        self._logLambda: Optional[float] = None
        self._cholLambda: Optional[np.ndarray] = None
        self._Sigma: Optional[np.ndarray] = None

    def _compute_nat_std(self) -> None:
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
    def logLambda(self) -> float:
        """float: Log-determinant of the diagonal precision matrix."""
        if self._logLambda is None:
            assert self.is_init
            self._logLambda = np.sum(np.log(self.Lambda))
        return self._logLambda

    @property
    def cholLambda(self) -> np.ndarray:
        """np.ndarray: Element-wise square root of ``Lambda``."""
        if self._cholLambda is None:
            assert self.is_init
            self._cholLambda = np.sqrt(self.Lambda)
        return self._cholLambda

    @property
    def Sigma(self) -> np.ndarray:
        """np.ndarray: Element-wise inverse of ``Lambda`` (the variance)."""
        if self._Sigma is None:
            assert self.is_init
            self._Sigma = 1.0 / self.Lambda
        return self._Sigma

    def initialize(self) -> None:
        """Validates parameters and derives any missing representation."""
        self.validate()
        self._compute_nat_std()
        assert self.is_init

    def stack_suff_stats(
        self, F: np.ndarray, S: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Stacks first- and second-order sufficient statistics.

        Args:
            F: First-order statistics.
            S: Optional vector of second-order statistics.

        Returns:
            Concatenated vector ``[F, S]`` (or ``F`` if ``S`` is ``None``).
        """
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separates stacked statistics into first- and second-order parts.

        Args:
            stats: Vector produced by :meth:`stack_suff_stats`.

        Returns:
            Tuple ``(F, S)`` where both are ``np.ndarray``.
        """
        F = stats[: self.x_dim]
        S = stats[self.x_dim :]
        return F, S

    def norm_suff_stats(
        self,
        N: float,
        u_x: Optional[np.ndarray] = None,
        return_order2: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """Whitens accumulated sufficient statistics using current parameters.

        Args:
            N: Zeroth-order statistic (effective sample count).
            u_x: Stacked sufficient statistics to normalize.
            return_order2: If ``True`` also whiten the second-order term.

        Returns:
            Tuple ``(N, stats)`` where ``stats`` contains the normalized
            first-order vector or the stacked first-/second-order vectors.
        """
        assert self.is_init
        assert u_x is not None
        F, S = self.unstack_suff_stats(u_x)
        F_norm = self.cholLambda * (F - N * self.mu)
        if return_order2:
            S = S - 2 * self.mu * F + N * self.mu**2
            S *= self.Lambda
            return N, self.stack_suff_stats(F_norm, S)
        return N, F_norm

    def Mstep(self, N: float, u_x: np.ndarray) -> None:
        """Maximization step that updates ``mu`` and ``Lambda``.

        Args:
            N: Zeroth-order statistic.
            u_x: Stacked first- and second-order sufficient statistics.
        """
        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N

        if self.update_Lambda:
            S = S / N - self.mu**2
            S[S < self.var_floor] = self.var_floor
            self.Lambda = 1 / S
            self._Sigma = S
            self._cholLambda = None
            self._logLambda = None

        self._compute_nat_params()

    def log_prob_std(self, x: np.ndarray) -> np.ndarray:
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

    def log_cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the log CDF at each sample.

        Args:
            x: Samples of shape ``(num_samples, x_dim)``.

        Returns:
            Array of log-CDF values with shape ``(num_samples,)``.
        """
        assert self.is_init
        delta = (x - self.mu) * self.cholLambda
        lk = 0.5 * (1 + erf(delta / np.sqrt(2)))
        return np.sum(np.log(lk + 1e-10), axis=-1)

    def sample(
        self,
        num_samples: int,
        rng: Optional[Union[Generator, RandomState]] = None,
        seed: int = 1024,
    ) -> np.ndarray:
        """Draws samples from the distribution.

        Args:
            num_samples: Number of samples to generate.
            rng: Optional :class:`numpy.random.Generator` or ``RandomState``.
            seed: Seed used if ``rng`` is ``None``.

        Returns:
            Array of samples with shape ``(num_samples, x_dim)``.
        """
        assert self.is_init
        if rng is None:
            rng = np.random.default_rng(seed)
        x = rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())
        return self.mu + 1.0 / self.cholLambda * x

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary.

        Returns:
            Dictionary of constructor arguments.
        """
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(NormalDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: Any) -> None:
        """Saves the model parameters into an HDF5 handle.

        Args:
            f: File-like object created by :mod:`h5py`.
        """
        assert self.is_init
        params = {"mu": self.mu, "Lambda": self.Lambda}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "NormalDiagCov":
        """Instantiates and populates a :class:`NormalDiagCov` from storage.

        Args:
            f: File handle pointing to the serialized parameters.
            config: Configuration dictionary produced by :meth:`get_config`.

        Returns:
            A fully initialized model.
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

    def _validate_mu(self) -> None:
        assert self.mu.shape[0] == self.x_dim

    def _validate_Lambda(self) -> None:
        assert self.Lambda.shape[0] == self.x_dim
        assert np.all(self.Lambda > 0)

    def _validate_eta(self) -> None:
        assert self.eta.shape[0] == self.x_dim * 2

    def validate(self) -> None:
        """Validates the parameters of the distribution."""
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Computes natural parameters from ``mu`` and ``Lambda``."""
        Lmu = Lambda * mu
        eta = np.hstack((Lmu, -0.5 * Lambda))
        return eta

    @staticmethod
    def compute_std(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts natural parameters into ``(mu, Lambda)``."""
        x_dim = int(eta.shape[0] / 2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        mu = -0.5 * eta1 / eta2
        Lambda = -2 * eta2
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta: np.ndarray) -> float:
        """Evaluates the log-normalizer ``A`` from natural parameters."""
        x_dim = int(eta.shape[0] / 2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -1 / 4 * np.sum(eta1 * eta1 / eta2)
        r3 = -1 / 2 * np.sum(np.log(-2 * eta2))
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu: np.ndarray, Lambda: np.ndarray) -> float:
        """Evaluates the log-normalizer ``A`` from standard parameters."""
        x_dim = mu.shape[0]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -0.5 * np.sum(np.log(Lambda))
        r3 = 0.5 * np.sum(mu * mu * Lambda)
        return r1 + r2 + r3

    def _compute_nat_params(self) -> None:
        """Computes ``eta``/``A`` from ``mu`` and ``Lambda``."""
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_nat(self.eta)
        # Lmu = self.Lambda*self.mu
        # muLmu = np.sum(self.mu*Lmu)
        # lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        # self.eta=np.hstack((lnr, Lmu, -0.5*self.Lambda)).T

    def _compute_std_params(self) -> None:
        """Recomputes ``mu`` and ``Lambda`` from ``eta``."""
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x: np.ndarray) -> np.ndarray:
        """Computes first- and second-order sufficient statistics per sample.

        Args:
            x: Data matrix with shape ``(num_samples, x_dim)``.

        Returns:
            Array of shape ``(num_samples, 2 * x_dim)`` containing ``[x, x^2]``.
        """
        d = x.shape[1]
        u = np.zeros((x.shape[0], 2 * d), dtype=float_cpu())
        u[:, :d] = x
        u[:, d:] = x * x
        return u

    @staticmethod
    def kl_div(p: "NormalDiagCov", q: "NormalDiagCov") -> float:
        """Computes the KL divergence between two NormalDiagCov distributions.

        Args:
          p: first distribution.
          q: second distribution.

        Returns:
          KL divergence between p and q.

        Formula:
            KL(P || Q) = 0.5 * sum_i [
                log(Sq_i / Sp_i) +
                (Sp_i + (mu_p_i - mu_p_i)^2) / S_q_i - 1
            ]
        """
        assert p.is_init and q.is_init
        assert p.x_dim == q.x_dim

        mu_p = p.mu
        Sigma_p = 1 / p.Lambda
        mu_q = q.mu
        Sigma_q = 1 / q.Lambda

        term1 = np.log(Sigma_q / Sigma_p)
        term2 = (Sigma_p + (mu_p - mu_q) ** 2) / Sigma_q
        kl = 0.5 * np.sum(term1 + term2 - 1)
        return kl

    def plot1D(
        self, feat_idx: int = 0, num_sigmas: int = 2, num_pts: int = 100, **kwargs: Any
    ) -> None:
        """Plots a single dimension of the Gaussian.

        Args:
            feat_idx: Feature index to display.
            num_sigmas: Plot extent in standard deviations.
            num_pts: Number of points used to draw the curve.
            **kwargs: Forwarded to :func:`plot_gaussian_1D`.
        """
        mu = self.mu[feat_idx]
        C = 1 / self.Lambda[feat_idx]
        plot_gaussian_1D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot2D(
        self,
        feat_idx: Sequence[int] = (0, 1),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots two dimensions of the Gaussian as a 2-D ellipsoid.

        Args:
            feat_idx: Indices of the dimensions to visualize.
            num_sigmas: Extent of the plot in standard deviations.
            num_pts: Number of samples used to render the ellipse.
            **kwargs: Additional plotting keyword arguments.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_2D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D(
        self,
        feat_idx: Sequence[int] = (0, 1),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots two dimensions of the Gaussian as a 3-D surface.

        Args:
            feat_idx: Indices of the features to display.
            num_sigmas: Extent of the domain in standard deviations.
            num_pts: Resolution of the plotted grid.
            **kwargs: Plotting helper keyword arguments.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_3D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(
        self,
        feat_idx: Sequence[int] = (0, 1, 2),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots a 3-D ellipsoid defined by three features.

        Args:
            feat_idx: Feature indices specifying the ellipsoid.
            num_sigmas: Extent in standard deviations.
            num_pts: Resolution of the sampled mesh.
            **kwargs: Additional plotting keyword arguments.
        """
        mu = self.mu[feat_idx]
        C = np.diag(1.0 / self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_3D(mu, C, num_sigmas, num_pts, **kwargs)


DiagNormal = NormalDiagCov
