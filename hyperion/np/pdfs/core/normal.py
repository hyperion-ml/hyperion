"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.random import Generator, RandomState

from ....hyp_defs import float_cpu
from ....utils.math_funcs import (
    fullcov_varfloor,
    invert_pdmat,
    invert_trimat,
    logdet_pdmat,
    symmat2vec,
    vec2symmat,
)
from ....utils.plotting import (
    plot_gaussian_1D,
    plot_gaussian_3D,
    plot_gaussian_ellipsoid_2D,
    plot_gaussian_ellipsoid_3D,
)
from .exp_family import ExpFamily


class Normal(ExpFamily):
    """Multivariate normal distribution with a full covariance matrix.

    This class exposes both the standard parameters (mean ``mu`` and precision
    matrix ``Lambda``) and the natural parameters used by the `ExpFamily` base
    class, allowing it to participate in EM-style optimization or closed-form
    updates.

    Attributes:
        mu: Mean vector with shape ``(x_dim,)`` or ``None`` if it has not been
            initialized yet.
        Lambda: Precision matrix with shape ``(x_dim, x_dim)`` or ``None`` until
            initialized.
        var_floor: Minimum allowed variance when floor operations are applied.
        update_mu: Whether ``mu`` is updated during the M-step.
        update_Lambda: Whether ``Lambda`` is updated during the M-step.
        x_dim: Data dimensionality inherited from :class:`PDF`.

    Examples:
        Initialize from standard parameters and evaluate log-probabilities:

        >>> import numpy as np
        >>> from hyperion.np.pdfs import Normal
        >>> mu = np.array([0.0, 1.0])
        >>> Lambda = np.array([[2.0, 0.3], [0.3, 1.5]])
        >>> model = Normal(mu=mu, Lambda=Lambda, x_dim=2)
        >>> x = np.array([[0.1, 1.2], [-0.5, 0.7]])
        >>> llk = model.log_prob(x)
        >>> llk.shape
        (2,)

        Fit a model from data and sample from it:

        >>> rng = np.random.default_rng(1)
        >>> x_train = rng.normal(size=(1000, 2))
        >>> model = Normal(x_dim=2)
        >>> _ = model.fit(x_train)
        >>> x_gen = model.sample(5, rng=rng)
        >>> x_gen.shape
        (5, 2)
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
        """Initializes a Normal distribution instance.

        Args:
            mu: Optional mean vector used as the starting point.
            Lambda: Optional precision matrix (inverse covariance).
            var_floor: Minimum variance allowed during covariance updates.
            update_mu: Whether ``mu`` is updated in :meth:`Mstep`.
            update_Lambda: Whether ``Lambda`` is updated in :meth:`Mstep`.
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
        """Keeps natural and standard parameterizations in sync."""
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
        """float: Log-determinant of the precision matrix."""
        if self._logLambda is None:
            assert self.is_init
            f, L, logL = invert_pdmat(self.Lambda, return_logdet=True)
            self._logLambda = logL
            self._cholLambda = L.T
        return self._logLambda

    @property
    def cholLambda(self) -> np.ndarray:
        """np.ndarray: Upper-triangular Cholesky factor of ``Lambda``."""
        if self._cholLambda is None:
            assert self.is_init
            f, L, logL = invert_pdmat(self.Lambda, return_logdet=True)
            self._logLambda = logL
            self._cholLambda = L.T
        return self._cholLambda

    @property
    def Sigma(self) -> np.ndarray:
        """np.ndarray: Covariance matrix (inverse of ``Lambda``)."""
        if self._Sigma is None:
            assert self.is_init
            self._Sigma = invert_pdmat(self.Lambda, return_inv=True)[-1]
        return self._Sigma

    def initialize(self) -> None:
        """Validates parameters and derives the missing representation."""
        self.validate()
        self._compute_nat_std()

    def stack_suff_stats(
        self, F: np.ndarray, S: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Stacks first- and second-order stats into a single vector.

        Args:
            F: First-order sufficient statistics.
            S: Optional flattened second-order statistics.

        Returns:
            Concatenated sufficient statistics.
        """
        if S is None:
            return F
        return np.hstack((F, S))

    def unstack_suff_stats(self, stats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decomposes stacked sufficient statistics.

        Args:
            stats: Vector produced by :meth:`stack_suff_stats`.

        Returns:
            Tuple of ``(F, S)`` where ``F`` is first-order stats and ``S`` contains
            flattened second-order stats.
        """
        F = stats[: self.x_dim]
        S = stats[self.x_dim :]
        return F, S

    def accum_suff_stats(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Accumulates sufficient statistics over several samples.

        Args:
            x: Data matrix of shape ``(num_samples, x_dim)``.
            u_x: Optional pre-computed sufficient statistics per sample.
            sample_weight: Optional weight per sample with shape ``(num_samples,)``.
            batch_size: Unused argument kept for API compatibility.

        Returns:
            Tuple ``(N, stats)`` where ``N`` is the weighted count of samples and
            ``stats`` the accumulated sufficient statistics.
        """
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

    def norm_suff_stats(
        self, N: float, u_x: np.ndarray, return_order2: bool = False
    ) -> Tuple[float, np.ndarray]:
        """Whitens accumulated sufficient statistics.

        Args:
            N: Zeroth-order (weighted count) statistic.
            u_x: Stacked first-/second-order sufficient statistics.
            return_order2: If ``True`` also normalize the second-order term.

        Returns:
            Tuple ``(N, stats)`` where ``stats`` contains the normalized first-order
            vector or the stacked first/second-order vectors depending on
            ``return_order2``.
        """
        assert self.is_init

        F, S = self.unstack_suff_stats(u_x)
        F_norm = np.dot(F - N * self.mu, self.cholLambda.T)
        if return_order2:
            SS = vec2symmat(S)
            Fmu = np.outer(F, self.mu)
            SS = SS - Fmu - Fmu.T + N * np.outer(self.mu, self.mu)
            SS = np.dot(self.cholLambda, np.dot(SS, self.cholLambda.T))
            S = symmat2vec(SS)
            return N, self.stack_suff_stats(F_norm, S)
        return N, F_norm

    def Mstep(self, N: float, u_x: np.ndarray) -> None:
        """Maximization step of EM for the Gaussian parameters.

        Args:
            N: Zeroth-order statistic (effective sample size).
            u_x: Stacked first-/second-order sufficient statistics.
        """
        F, S = self.unstack_suff_stats(u_x)

        if self.update_mu:
            self.mu = F / N
        elif self.mu is None:
            raise ValueError("mu must be initialized if update_mu is False")

        if self.update_Lambda:
            S = vec2symmat(S / N)
            S -= np.outer(self.mu, self.mu)
            S = fullcov_varfloor(S, self.var_floor)
            self.Lambda = invert_pdmat(S, return_inv=True)[-1]
            self._Sigma = None
            self._logLambda = None
            self._cholLambda = None
        elif self.Lambda is None:
            raise ValueError("Lambda must be initialized if update_Lambda is False")

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
        mah_dist2 = np.sum(np.dot(x - self.mu, self.cholLambda) ** 2, axis=1)
        return (
            0.5 * self.logLambda
            - 0.5 * self.x_dim * np.log(2 * np.pi)
            - 0.5 * mah_dist2
        )

    def sample(
        self,
        num_samples: int,
        rng: Optional[Union[Generator, RandomState]] = None,
        seed: int = 1024,
    ) -> np.ndarray:
        """Draws samples from the data distribution.

        Args:
            num_samples: Number of samples to draw.
            rng: Optional :class:`numpy.random.Generator` or :class:`RandomState`.
            seed: Seed used when ``rng`` is ``None``.

        Returns:
            Array of generated samples with shape ``(num_samples, x_dim)``.
        """
        assert self.is_init

        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mu, self.Sigma, size=(num_samples,)).astype(
            float_cpu()
        )

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration for this model.

        Returns:
            Dictionary containing constructor arguments.
        """
        config = {
            "var_floor": self.var_floor,
            "update_mu": self.update_mu,
            "update_lambda": self.update_Lambda,
        }
        base_config = super(Normal, self).get_config()
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
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "Normal":
        """Loads parameters and instantiates a :class:`Normal`.

        Args:
            f: File handle pointing to the serialized parameters.
            config: Configuration dictionary produced by :meth:`get_config`.

        Returns:
            A fully initialized :class:`Normal`.
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
        assert self.Lambda.shape == (self.x_dim, self.x_dim)

    def _validate_eta(self) -> None:
        assert self.eta.shape[0] == (self.x_dim**2 + 3 * self.x_dim) / 2

    def validate(self) -> None:
        """Validates the parameters of the distribution."""
        if self.mu is not None and self.Lambda is not None:
            self._validate_mu()
            self._validate_Lambda()

        if self.eta is not None:
            self._validate_eta()

    @staticmethod
    def compute_eta(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Computes the natural parameters from mean and precision.

        Args:
            mu: Mean vector.
            Lambda: Precision matrix.

        Returns:
            Natural-parameter vector ``eta``.
        """
        Lmu = np.dot(mu, Lambda)
        eta = np.hstack((Lmu, -symmat2vec(Lambda, diag_factor=0.5)))
        return eta

    @staticmethod
    def compute_x_dim_from_eta(eta: np.ndarray) -> int:
        """Infers ``x_dim`` from the length of the natural parameter vector.

        Args:
            eta: Natural-parameter vector.

        Returns:
            Integer dimensionality consistent with ``eta``.
        """
        x_dim = 0.5 * (-3 + np.sqrt(9 + 8 * eta.shape[-1]))
        assert int(x_dim) == x_dim
        return int(x_dim)

    @staticmethod
    def compute_std(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts natural parameters to ``(mu, Lambda)``.

        Args:
            eta: Natural-parameter vector.

        Returns:
            Tuple ``(mu, Lambda)`` with the standard parameters.
        """
        x_dim = Normal.compute_x_dim_from_eta(eta)
        eta1 = eta[:x_dim]
        eta2 = vec2symmat(eta[x_dim:], diag_factor=2) / 2
        Lambda = -2 * eta2
        f = invert_pdmat(-eta2, right_inv=True)[0]
        mu = 0.5 * f(eta1)
        return mu, Lambda

    @staticmethod
    def compute_A_nat(eta: np.ndarray) -> float:
        """Evaluates the log-normalizer ``A`` from natural parameters.

        Args:
            eta: Natural-parameter vector.

        Returns:
            Scalar log-normalizer.
        """
        x_dim = Normal.compute_x_dim_from_eta(eta)
        eta1 = eta[:x_dim]
        eta2 = vec2symmat(eta[x_dim:], diag_factor=2) / 2
        f, _, log_minus_eta2 = invert_pdmat(-eta2, right_inv=True, return_logdet=True)
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = 0.25 * np.inner(f(eta1), eta1)
        r3 = -0.5 * x_dim * np.log(2) - 0.5 * log_minus_eta2
        return r1 + r2 + r3

    @staticmethod
    def compute_A_std(mu: np.ndarray, Lambda: np.ndarray) -> float:
        """Evaluates the log-normalizer ``A`` from standard parameters.

        Args:
            mu: Mean vector.
            Lambda: Precision matrix.

        Returns:
            Scalar log-normalizer.
        """
        x_dim = mu.shape[0]
        r1 = 0.5 * x_dim * np.log(2 * np.pi)
        r2 = -0.5 * logdet_pdmat(Lambda)
        r3 = 0.5 * np.inner(np.dot(mu, Lambda), mu)
        return r1 + r2 + r3

    def _compute_nat_params(self) -> None:
        """Computes ``eta`` and ``A`` from ``mu`` and ``Lambda``."""
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_std(self.mu, self.Lambda)

    def _compute_std_params(self) -> None:
        """Computes ``mu`` and ``Lambda`` from ``eta``."""
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._logLambda = None
        self._Sigma = None

    @staticmethod
    def compute_suff_stats(x: np.ndarray) -> np.ndarray:
        """Computes sufficient statistics for each sample.

        Args:
            x: Data samples of shape ``(num_samples, x_dim)``.

        Returns:
            Array of shape ``(num_samples, u_dim)`` containing concatenated first
            and second-order statistics.
        """
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
        """Plots a 1-D slice of the Gaussian.

        Args:
            feat_idx: Feature index to plot.
            num_sigmas: Extent of the plot in standard deviations.
            num_pts: Number of points used to draw the curve.
            **kwargs: Extra keyword arguments passed to matplotlib helpers.
        """
        assert self.is_init
        mu = self.mu[feat_idx]
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][feat_idx, feat_idx]
        plot_gaussian_1D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot2D(
        self,
        feat_idx: Sequence[int] = (0, 1),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots a two-dimensional ellipse for the Gaussian.

        Args:
            feat_idx: Indices of the two features to plot.
            num_sigmas: Extent of the plot in standard deviations.
            num_pts: Number of points used to parameterize the ellipse.
            **kwargs: Extra keyword arguments forwarded to the plotting helper.
        """
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
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
            feat_idx: Indices of the two features to plot.
            num_sigmas: Extent of the domain in standard deviations.
            num_pts: Resolution of the plotted grid.
            **kwargs: Plotting helper keyword arguments.
        """
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
        plot_gaussian_3D(mu, C, num_sigmas, num_pts, **kwargs)

    def plot3D_ellipsoid(
        self,
        feat_idx: Sequence[int] = (0, 1, 2),
        num_sigmas: int = 2,
        num_pts: int = 100,
        **kwargs: Any,
    ) -> None:
        """Plots a 3-D ellipsoid for the Gaussian marginal.

        Args:
            feat_idx: Indices of the three features to visualize.
            num_sigmas: Extent of the ellipsoid in standard deviations.
            num_pts: Resolution of the mesh used to approximate the ellipsoid.
            **kwargs: Additional plotting keyword arguments.
        """
        assert self.is_init
        mu = self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C = invert_pdmat(self.Lambda, return_inv=True)[-1][i, j]
        plot_gaussian_ellipsoid_3D(mu, C, num_sigmas, num_pts, **kwargs)
