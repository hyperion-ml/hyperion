"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg as la

from ....hyp_defs import float_cpu
from ....utils.math_funcs import invert_pdmat
from ..core.pdf import PDF


class JFATotal(PDF):
    """Joint factor analysis model with a total-variability matrix (i-vectors).

    Attributes:
        K: Number of Gaussian mixture components.
        y_dim: Dimensionality of the latent i-vector subspace.
        T: Total-variability matrix with shape ``(y_dim, K * x_dim)``.

    Examples:
        >>> import numpy as np
        >>> from hyperion.np.pdfs.jfa.jfa_total import JFATotal
        >>> rng = np.random.default_rng(7)
        >>> num_utts, K, x_dim = 300, 4, 20
        >>> N = rng.uniform(1.0, 30.0, size=(num_utts, K)).astype(np.float32)
        >>> F = rng.standard_normal((num_utts, K * x_dim)).astype(np.float32)
        >>> model = JFATotal(K=K, y_dim=64)
        >>> _ = model.fit(N, F, epochs=3, ml_md="ml+md")
        >>> y = model.compute_py_g_x(N[:10], F[:10])
        >>> print(y.shape)  # (10, 64)
    """

    def __init__(
        self,
        K: int,
        y_dim: Optional[int] = None,
        T: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the JFA total-variability model.

        Args:
            K: Number of mixture components.
            y_dim: Latent dimension if ``T`` is not provided.
            T: Optional total-variability matrix used for initialization.
            **kwargs: Additional keyword arguments passed to :class:`PDF`.
        """
        super().__init__(**kwargs)
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if T is not None:
            self._validate_T_shape(T, K)
            y_dim = T.shape[0]
        elif y_dim is None:
            raise ValueError("Either y_dim or T must be provided.")

        self.K: int = K
        self.y_dim: int = y_dim
        self.T: Optional[np.ndarray] = T

        # aux
        self._TT: Optional[np.ndarray] = None
        self.__upptr: Optional[np.ndarray] = None

    def reset_aux(self) -> None:
        """Resets cached matrices derived from ``T``."""
        self._TT = None

    @property
    def is_init(self) -> bool:
        """bool: Whether the model has been initialized."""
        if self._is_init:
            return True
        if self.T is not None:
            self._validate_T_shape(self.T, self.K, expected_y_dim=self.y_dim)
            self._is_init = True
        return self._is_init

    def initialize(self, N: np.ndarray, F: np.ndarray) -> None:
        """Randomly initializes ``T`` using zero/first-order statistics.

        Args:
            N: Zero-order statistics of shape ``(num_utterances, K)``.
            F: First-order statistics of shape ``(num_utterances, K * x_dim)``.
        """
        if N.shape[0] != F.shape[0]:
            raise ValueError(
                f"N.shape[0]={N.shape[0]} does not match F.shape[0]={F.shape[0]}"
            )
        if N.shape[1] != self.K:
            raise ValueError(f"N.shape[1]={N.shape[1]} does not match K={self.K}")
        if F.shape[1] % self.K != 0:
            raise ValueError(
                f"F.shape[1] must be divisible by K. Got F.shape[1]={F.shape[1]}, K={self.K}"
            )
        self.T = np.random.randn(self.y_dim, F.shape[1]).astype(float_cpu(), copy=False)
        self.reset_aux()

    def compute_py_g_x(
        self,
        N: np.ndarray,
        F: np.ndarray,
        G: Optional[np.ndarray] = None,
        return_cov: bool = False,
        return_elbo: bool = False,
        return_acc: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Computes the posterior over latent factors ``p(y | x)``.

        Args:
            N: Zero-order statistics with shape ``(num_utts, K)``.
            F: First-order statistics with shape ``(num_utts, K * x_dim)``.
            G: Optional log-likelihood term to add to the ELBO.
            return_cov: Whether to also return posterior covariances.
            return_elbo: Whether to return the per-utterance ELBO.
            return_acc: Whether to return accumulators for EM updates.

        Returns:
            If ``return_cov``/``return_elbo``/``return_acc`` are all ``False``,
            only returns the posterior means ``y``; otherwise returns a tuple
            containing the requested quantities in order
            ``(y, cov?, elbo?, Ry, Py)``.
        """
        assert self.is_init
        M = F.shape[0]
        y_dim = self.y_dim
        assert self.T is not None
        if N.shape[0] != F.shape[0]:
            raise ValueError(
                f"N.shape[0]={N.shape[0]} does not match F.shape[0]={F.shape[0]}"
            )
        if N.shape[1] != self.K:
            raise ValueError(f"N.shape[1]={N.shape[1]} does not match K={self.K}")
        if F.shape[1] != self.T.shape[1]:
            raise ValueError(
                f"F.shape[1]={F.shape[1]} does not match T.shape[1]={self.T.shape[1]}"
            )

        compute_inv = return_cov or return_acc
        return_tuple = compute_inv or return_elbo

        TF = np.dot(F, self.T.T)
        L = self.compute_L(self.TT, N, self._upptr)
        y = np.zeros((M, y_dim), dtype=float_cpu())

        if return_cov:
            Sy = np.zeros((M, int(y_dim * (y_dim + 1) // 2)), dtype=float_cpu())
        else:
            Sy = None

        if return_elbo:
            elbo = np.zeros((M,), dtype=float_cpu())

        if return_acc:
            Py = np.zeros((y_dim, y_dim), dtype=float_cpu())
            Ry = np.zeros((self.K, int(y_dim * (y_dim + 1) // 2)), dtype=float_cpu())

        Li = np.zeros((self.y_dim, self.y_dim), dtype=float_cpu())
        for i in range(N.shape[0]):
            Li[self._upptr] = L[i]
            r = invert_pdmat(
                Li, right_inv=True, return_logdet=return_elbo, return_inv=compute_inv
            )
            mult_iL = r[0]
            if return_elbo:
                elbo[i] = -r[2] / 2
            if compute_inv:
                iL = r[-1]

            y[i] = mult_iL(TF[i])

            if return_cov:
                Sy[i] = iL[self.__upptr]

            if return_acc:
                iL += np.outer(y[i], y[i])
                Py += iL
                Ry += iL[self.__upptr] * N[i][:, None]

        if not return_tuple:
            return y

        r = [y]

        if return_cov:
            r += [Sy]

        if return_elbo:
            if G is not None:
                elbo += G
            elbo += 0.5 * np.sum(TF * y, axis=-1)
            r += [elbo]

        if return_acc:
            r += [Ry, Py]

        return tuple(r)

    def Estep(
        self, N: np.ndarray, F: np.ndarray, G: Optional[np.ndarray] = None
    ) -> Tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Expectation step computing posterior stats.

        Args:
            N: Zero-order statistics.
            F: First-order statistics.
            G: Optional log-likelihood contribution for the ELBO.

        Returns:
            Tuple ``(elbo, M, y_acc, Ry, Cy, Py)`` with the quantities needed
            for the M-step.
        """
        y, elbo, Ry, Py = self.compute_py_g_x(
            N, F, G, return_elbo=True, return_acc=True
        )

        M = y.shape[0]
        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F.T, y)

        elbo = np.sum(elbo)

        stats = (elbo, M, y_acc, Ry, Cy, Py)
        return stats

    def MstepML(self, stats: Tuple[Any, ...]) -> None:
        """Maximum-likelihood update of ``T``.

        Args:
            stats: Tuple produced by :meth:`Estep`.
        """
        _, M, y_acc, Ry, Cy, _ = stats
        T = np.zeros_like(self.T)
        Ryk = np.zeros((self.y_dim, self.y_dim), dtype=float_cpu())
        x_dim = T.shape[1] // self.K
        for k in range(self.K):
            idx = k * x_dim
            Ryk[self._upptr] = Ry[k]
            iRyk_mult = invert_pdmat(Ryk, right_inv=False)[0]
            T[:, idx : idx + x_dim] = iRyk_mult(Cy[idx : idx + x_dim].T)

        self.T = T
        self.reset_aux()

    def MstepMD(self, stats: Tuple[Any, ...]) -> None:
        """Minimum-divergence adaptation step.

        Args:
            stats: Tuple produced by :meth:`Estep`.
        """
        _, M, y_acc, Ry, Cy, Py = stats
        mu_y = y_acc / M
        Cy = Py / M - np.outer(mu_y, mu_y)
        chol_Cy = la.cholesky(Cy, lower=False, overwrite_a=True)
        self.T = np.dot(chol_Cy, self.T)

        self.reset_aux()

    def fit(
        self,
        N: np.ndarray,
        F: np.ndarray,
        G: Optional[np.ndarray] = None,
        N_val: Optional[np.ndarray] = None,
        F_val: Optional[np.ndarray] = None,
        G_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        ml_md: str = "ml+md",
        md_epochs: Optional[Tuple[int, ...]] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Trains the model via alternating ML/MD steps.

        Args:
            N: Training zero-order stats.
            F: Training first-order stats.
            G: Optional training log-likelihood term.
            N_val: Validation zero-order stats.
            F_val: Validation first-order stats.
            G_val: Optional validation log-likelihood term.
            epochs: Number of training epochs.
            ml_md: Strategy: ``"ml"``, ``"md"``, or ``"ml+md"``.
            md_epochs: Optional tuple of epochs where MD is applied.

        Returns:
            Training ELBO and normalized ELBO, optionally followed by the
            validation counterparts.
        """
        if (N_val is None) != (F_val is None):
            raise ValueError("N_val and F_val must be provided together.")
        if ml_md not in ("ml+md", "ml", "md"):
            raise ValueError(f"ml_md must be 'ml+md', 'ml' or 'md', got '{ml_md}'")
        use_val = N_val is not None and F_val is not None

        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        if not self.is_init:
            self.initialize(N, F)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):

            stats = self.Estep(N, F, G)
            elbo[epoch] = stats[0]
            if use_val:
                _, elbo_val_e = self.compute_py_g_x(
                    N_val, F_val, G_val, return_elbo=True
                )
                elbo_val[epoch] = np.sum(elbo_val_e)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

        elbo_norm = elbo / np.sum(N)
        if not use_val:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(N_val)
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    @property
    def TT(self):
        """np.ndarray: Vectorized ``T_k T_k^T`` matrices for each component."""
        if self._TT is None:
            self._TT = self.compute_TT(self.T, self.K, self._upptr)
        return self._TT

    @property
    def _upptr(self):
        """np.ndarray: Upper-triangular mask used for vectorization."""
        if self.__upptr is None:
            self.__upptr = np.triu(np.ones(self.y_dim, dtype=bool))
        return self.__upptr

    @staticmethod
    def compute_TT(T: np.ndarray, K: int, upptr: np.ndarray) -> np.ndarray:
        """Computes the vectorized ``T_k T_k^T`` matrices for each component.

        Args:
            T: Total-variability matrix.
            K: Number of Gaussian components.
            upptr: Upper-triangular mask.

        Returns:
            Array of shape ``(K, y_dim * (y_dim + 1) / 2)`` containing vectorized
            upper-triangular entries.
        """
        JFATotal._validate_T_shape(T, K)
        x_dim = T.shape[1] // K
        y_dim = T.shape[0]
        TT = np.zeros((K, int(y_dim * (y_dim + 1) / 2)), dtype=float_cpu())
        for k in range(K):
            idx = k * x_dim
            T_k = T[:, idx : idx + x_dim]
            TT_k = np.dot(T_k, T_k.T)
            TT[k] = TT_k[upptr]

        return TT

    @staticmethod
    def compute_L(TT: np.ndarray, N: np.ndarray, upptr: np.ndarray) -> np.ndarray:
        """Computes the posterior precision matrix for each utterance.

        Args:
            TT: Vectorized ``T_k T_k^T`` matrices.
            N: Zero-order statistics.
            upptr: Upper-triangular mask.

        Returns:
            Vectorized precision matrices matching ``upptr``.
        """
        y_dim = upptr.shape[0]
        I = np.eye(y_dim, dtype=float_cpu())[upptr]
        return I + np.dot(N, TT)

    @staticmethod
    def normalize_T(T: np.ndarray, chol_prec: np.ndarray) -> np.ndarray:
        """Normalizes ``T`` by the GMM covariances.

        Args:
            T: Total-variability matrix.
            chol_prec: Cholesky factors of the component precisions.

        Returns:
            Normalized matrix with the same shape as ``T``.
        """
        Tnorm = np.zeros_like(T)
        K = chol_prec.shape[0]
        JFATotal._validate_T_shape(T, K)
        x_dim = T.shape[1] // K
        if chol_prec.ndim != 3:
            raise ValueError(
                f"chol_prec must be 3D with shape (K, x_dim, x_dim), got {chol_prec.shape}"
            )
        if chol_prec.shape[1] != x_dim or chol_prec.shape[2] != x_dim:
            raise ValueError(
                "chol_prec inner dimensions must match x_dim="
                f"{x_dim}, got {chol_prec.shape[1:]}"
            )
        for k in range(K):
            idx = k * x_dim
            Tnorm[:, idx : idx + x_dim] = np.dot(
                T[:, idx : idx + x_dim], chol_prec[k].T
            )

        return Tnorm

    @staticmethod
    def _validate_T_shape(
        T: np.ndarray, K: int, expected_y_dim: Optional[int] = None
    ) -> None:
        """Validates ``T`` layout against ``K`` and optional ``y_dim``."""
        if T.ndim != 2:
            raise ValueError(f"T must be 2D, got shape {T.shape}")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if T.shape[1] % K != 0:
            raise ValueError(
                f"T.shape[1] must be divisible by K. Got T.shape[1]={T.shape[1]}, K={K}"
            )
        if expected_y_dim is not None and T.shape[0] != expected_y_dim:
            raise ValueError(
                f"T.shape[0]={T.shape[0]} does not match y_dim={expected_y_dim}"
            )

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary."""
        config = {"K": self.K}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: Any) -> None:
        """Persists the total-variability matrix into an HDF5 handle.

        Args:
            f: File-like object created by :mod:`h5py`.
        """
        params = {"T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "JFATotal":
        """Loads parameters and instantiates a :class:`JFATotal`.

        Args:
            f: HDF5 handle pointing to the stored parameters.
            config: Configuration dictionary generated by :meth:`get_config`.

        Returns:
            Fully initialized :class:`JFATotal`.
        """
        param_list = ["T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draws samples from the i-vector model."""
        raise NotImplementedError()
