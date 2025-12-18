"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator

from ....hyp_defs import float_cpu
from ...transforms import LNorm
from ..core.pdf import PDF


class PLDALLRNvsMMethod(str, Enum):
    """Scoring strategies for PLDA N-vs-M trials."""

    vavg = "vavg"
    lnorm_vavg = "lnorm-vavg"
    savg = "savg"
    book = "book"

    @staticmethod
    def choices() -> Sequence["PLDALLRNvsMMethod"]:
        return [
            PLDALLRNvsMMethod.vavg,
            PLDALLRNvsMMethod.lnorm_vavg,
            PLDALLRNvsMMethod.savg,
            PLDALLRNvsMMethod.book,
        ]


class PLDABase(PDF):
    """Abstract base class for probabilistic linear discriminant analysis (PLDA).

    Attributes:
        y_dim: Latent speaker-factor dimensionality.
        mu: Global mean vector.
        update_mu: Whether to update ``mu`` during training.
        x_dim: Observed feature dimensionality.
        epochs: Default number of EM epochs.
        ml_md: Default training strategy (``"ml"``, ``"md"``, or ``"ml+md"``).
        md_epochs: Optional iterable of epochs where MD is applied.
    """

    def __init__(
        self,
        y_dim: Optional[int] = None,
        mu: Optional[np.ndarray] = None,
        update_mu: bool = True,
        epochs: int = 20,
        ml_md: str = "ml+md",
        md_epochs: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu: Optional[np.ndarray] = mu
        self.y_dim: Optional[int] = y_dim
        self.update_mu: bool = update_mu
        if mu is not None:
            self.x_dim = mu.shape[0]

        self.epochs: int = epochs
        self.ml_md: str = ml_md
        self.md_epochs: Optional[Sequence[int]] = md_epochs

    def initialize(self, D: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Initializes the PLDA model from sufficient statistics.

        Args:
            D: Tuple ``(N, F, S)`` of zero-, first-, and second-order stats.
        """
        pass

    def compute_py_g_x(
        self, D: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the posterior ``p(y | x)`` from sufficient statistics.

        Args:
            D: Tuple ``(N, F, S)`` of sufficient statistics.

        Returns:
            Posterior means and covariances for the latent variables.
        """
        pass

    def fit(
        self,
        x: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        ptheta: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        class_ids_val: Optional[np.ndarray] = None,
        ptheta_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        ml_md: Optional[str] = None,
        md_epochs: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trains the PLDA model via EM.

        Args:
            x: Training data of shape ``(num_samples, x_dim)``.
            class_ids: Hard class labels for training data.
            ptheta: Soft class assignments for training data.
            sample_weight: Optional per-sample weights.
            x_val: Optional validation data.
            class_ids_val: Validation class labels.
            ptheta_val: Validation soft assignments.
            sample_weight_val: Validation sample weights.
            epochs: Number of EM epochs.
            ml_md: Training strategy: ``"ml"``, ``"md"``, or ``"ml+md"``.
            md_epochs: Optional specific epochs for MD steps.

        Returns:
            Training ELBO trace and normalized ELBO; validation metrics if provided.
        """
        if epochs is None:
            epochs = self.epochs
        if ml_md is None:
            ml_md = self.ml_md
        if md_epochs is None:
            md_epochs = self.md_epochs

        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta, sample_weight=sample_weight)
        else:
            D = self.compute_stats_hard(x, class_ids, sample_weight=sample_weight)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(
                    x_val, ptheta_val, sample_weight=sample_weight_val
                )
            else:
                D_val = self.compute_stats_hard(
                    x_val, class_ids_val, sample_weight=sample_weight_val
                )

        if not self.is_init:
            self.initialize(D)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            stats = self.Estep(D)
            elbo[epoch] = self.elbo(stats)
            if x_val is not None:
                stats_val = self.Estep(D_val)
                elbo_val[epoch] = self.elbo(stats_val)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

        elbo_norm = elbo / np.sum(D[0])
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(D_val[0])
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    def Estep(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Expectation step placeholder.

        Args:
            x: Tuple ``(N, F, S)`` of sufficient statistics.
        """
        pass

    def MstepML(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Maximum-likelihood update placeholder.

        Args:
            x: Tuple ``(N, F, S)`` needed for the ML update.
        """
        pass

    def MstepMD(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Minimum-divergence update placeholder.

        Args:
            x: Tuple ``(N, F, S)`` needed for the MD update.
        """
        pass

    def fit_adapt_weighted_avg_model(
        self,
        x: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        ptheta: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        class_ids_val: Optional[np.ndarray] = None,
        ptheta_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        ml_md: str = "ml+md",
        md_epochs: Optional[Sequence[int]] = None,
        plda0: Optional["PLDABase"] = None,
        w_mu: float = 1,
        w_B: float = 0.5,
        w_W: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adapts a PLDA model and averages it with a prior after each epoch.

        Args:
            x: Training data.
            class_ids: Hard class labels for training data.
            ptheta: Soft class assignments for training data.
            sample_weight: Optional sample weights for training data.
            x_val: Optional validation data.
            class_ids_val: Validation class labels.
            ptheta_val: Validation soft assignments.
            sample_weight_val: Validation sample weights.
            epochs: Number of adaptation epochs.
            ml_md: Adaptation strategy ``"ml"``, ``"md"``, or ``"ml+md"``.
            md_epochs: Optional MD epochs.
            plda0: Prior PLDA model to average with.
            w_mu: Prior weight on the mean.
            w_B: Prior weight on between-class parameters.
            w_W: Prior weight on within-class parameters.

        Returns:
            Training ELBO trace and normalized ELBO, optionally validation metrics.
        """

        assert self.is_init
        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta, sample_weight=sample_weight)
        else:
            D = self.compute_stats_hard(x, class_ids, sample_weight=sample_weight)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(
                    x_val, ptheta_val, sample_weight=sample_weight_val
                )
            else:
                D_val = self.compute_stats_hard(
                    x_val, class_ids_val, sample_weight=sample_weight_val
                )

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            stats = self.Estep(D)
            elbo[epoch] = self.elbo(stats)
            if x_val is not None:
                stats_val = self.Estep(D_val)
                elbo_val[epoch] = self.elbo(stats_val)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

            self.weighted_avg_model(plda0, w_mu, w_B, w_W)

        elbo_norm = elbo / np.sum(D[0])
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(D_val[0])
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    @staticmethod
    def compute_stats_soft(
        x: np.ndarray,
        p_theta: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        scal_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes sufficient statistics using soft class assignments.

        Args:
            x: Samples with shape ``(num_samples, x_dim)``.
            p_theta: Soft class probabilities with shape ``(num_samples, num_classes)``.
            sample_weight: Optional sample weights.
            scal_factor: Optional scaling factor applied to the stats.

        Returns:
            Tuple ``(N, F, S)`` of zero-, first-, and second-order stats.
        """
        if sample_weight is not None:
            p_theta = sample_weight[:, None] * p_theta
        if scal_factor is not None:
            p_theta *= scal_factor
        N = np.sum(p_theta, axis=0)
        F = np.dot(p_theta.T, x)
        wx = np.sum(p_theta, axis=1, keepdims=True) * x
        S = np.dot(x.T, wx)
        return N, F, S

    @staticmethod
    def compute_stats_hard(
        x: np.ndarray,
        class_ids: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        scale_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes sufficient statistics using hard class assignments.

        Args:
            x: Samples with shape ``(num_samples, x_dim)``.
            class_ids: Integer labels in ``[0, num_classes-1]``.
            sample_weight: Optional sample weights.
            scale_factor: Optional scaling factor applied to the stats.

        Returns:
            Tuple ``(N, F, S)`` of zero-, first-, and second-order stats.
        """
        x_dim = x.shape[1]
        num_classes = np.max(class_ids) + 1
        N = np.zeros((num_classes,), dtype=float_cpu())
        F = np.zeros((num_classes, x_dim), dtype=float_cpu())
        if sample_weight is not None:
            wx = sample_weight[:, None] * x
        else:
            wx = x

        for i in range(num_classes):
            idx = class_ids == i
            if sample_weight is None:
                N[i] = np.sum(idx).astype(float_cpu())
                F[i] = np.sum(x[idx], axis=0)
            else:
                N[i] = np.sum(sample_weight[idx])
                F[i] = np.sum(wx[idx], axis=0)

        S = np.dot(x.T, wx)
        if scale_factor is not None:
            N *= scale_factor
            F *= scale_factor
            S *= scale_factor

        return N, F, S

    @staticmethod
    def compute_stats_hard_v0(
        x: np.ndarray,
        class_ids: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        scal_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Variant that converts hard labels into soft assignments."""
        x_dim = x.shape[1]
        num_classes = np.max(class_ids) + 1
        p_theta = np.zeros((x.shape[0], num_classes), dtype=float_cpu())
        p_theta[np.arange(x.shape[0]), class_ids] = 1
        return PLDABase.compute_stats_soft(x, p_theta, sample_weight, scal_factor)

    @staticmethod
    def center_stats(
        D: Tuple[np.ndarray, np.ndarray, np.ndarray], mu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centers sufficient statistics by subtracting the global mean.

        Args:
            D: Tuple ``(N, F, S)`` of sufficient statistics.
            mu: Global mean vector.

        Returns:
            Tuple ``(N, F_c, S_c)`` of centered statistics.
        """
        N, F, S = D
        Fc = F - np.outer(N, mu)
        Fmu = np.outer(np.sum(F, axis=0), mu)
        Sc = S - Fmu - Fmu.T + np.sum(N) * np.outer(mu, mu)
        return N, Fc, Sc

    def llr_1vs1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """LLR for single enrollment vs single test.

        Args:
            x1: Enrollment embeddings with shape ``(num_enroll, x_dim)``.
            x2: Test embeddings with shape ``(num_test, x_dim)``.

        Returns:
            Score matrix with shape ``(num_enroll, num_test)``.
        """
        pass

    def llr_NvsM_book(
        self,
        D1: Tuple[np.ndarray, np.ndarray, np.ndarray],
        D2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Exact N-vs-M PLDA scoring using sufficient statistics.

        Args:
            D1: Enrollment-side stats ``(N1, F1, S1)``.
            D2: Test-side stats ``(N2, F2, S2)``.

        Returns:
            Score matrix of shape ``(num_enroll_sides, num_test_sides)``.
        """
        pass

    def llr_NvsM(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        ids1: Optional[np.ndarray] = None,
        ids2: Optional[np.ndarray] = None,
        method: PLDALLRNvsMMethod = PLDALLRNvsMMethod.lnorm_vavg,
    ) -> np.ndarray:
        """Computes N-vs-M log-likelihood ratios under different strategies.

        Args:
            x1: Enrollment embeddings.
            x2: Test embeddings.
            ids1: Optional mapping from enrollment segments to sides.
            ids2: Optional mapping from test segments to sides.
            method: Scoring strategy to use.

        Returns:
            Score matrix; dimensions depend on the strategy.
        """
        if method == PLDALLRNvsMMethod.savg:
            return self.llr_NvsM_savg(x1, ids1, x2, ids2)

        D1 = x1 if ids1 is None else self.compute_stats_hard(x1, class_ids=ids1)
        D2 = x2 if ids2 is None else self.compute_stats_hard(x2, class_ids=ids2)

        if method == PLDALLRNvsMMethod.book:
            return self.llr_NvsM_book(D1, D2)
        if method == PLDALLRNvsMMethod.vavg:
            return self.llr_NvsM_vavg(D1, D2, do_lnorm=False)
        if method == PLDALLRNvsMMethod.lnorm_vavg:
            return self.llr_NvsM_vavg(D1, D2, do_lnorm=True)

        raise ValueError(f"wrong llr {method}")

    def llr_NvsM_vavg(
        self,
        D1: Tuple[np.ndarray, np.ndarray, np.ndarray],
        D2: Tuple[np.ndarray, np.ndarray, np.ndarray],
        do_lnorm: bool = True,
    ) -> np.ndarray:
        """Vector-averaged N-vs-M scoring with optional length normalization.

        Args:
            D1: Tuple ``(N1, F1, S1)`` for each enrollment side.
            D2: Tuple ``(N2, F2, S2)`` for each test side.
            do_lnorm: Whether to apply length normalization after vector averaging.

        Returns:
            Score matrix of shape ``(num_enroll_sides, num_test_sides)``.
        """
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        x2 = D2[1] / np.expand_dims(D2[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_NvsM_savg(
        self,
        x1: np.ndarray,
        ids1: np.ndarray,
        x2: np.ndarray,
        ids2: np.ndarray,
    ) -> np.ndarray:
        """Score-averaged N-vs-M scoring using segment-level LLRs.

        Args:
            x1: Enrollment embeddings with shape ``(num_segments, x_dim)``.
            ids1: Mapping from enrollment segments to enrollment sides.
            x2: Test embeddings with shape ``(num_segments, x_dim)``.
            ids2: Mapping from test segments to test sides.

        Returns:
            Score matrix with shape ``(num_enroll_sides, num_test_sides)``.
        """
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores_Nvs1 = F / N[:, None]
        N, F, _ = self.compute_stats_hard(scores_Nvs1.T, ids2)
        scores = F.T / N
        return scores

    def llr_Nvs1(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        ids1: Optional[np.ndarray] = None,
        method: PLDALLRNvsMMethod = PLDALLRNvsMMethod.lnorm_vavg,
    ) -> np.ndarray:
        """LLR for N enrollment segments vs single test segments.

        Args:
            x1: Enrollment embeddings with shape ``(num_segments, x_dim)``.
            x2: Test embeddings with shape ``(num_test_segments, x_dim)``.
            ids1: Optional mapping from segments to enrollment sides.
            method: Scoring strategy (``"book"``, ``"vavg"``, ``"lnorm-vavg"``, ``"savg"``).

        Returns:
            Score matrix with shape ``(num_enroll_sides, num_test_segments)``.
        """
        if method == PLDALLRNvsMMethod.savg:
            return self.llr_Nvs1_savg(x1, ids1, x2)

        D1 = x1 if ids1 is None else self.compute_stats_hard(x1, class_ids=ids1)

        if method == PLDALLRNvsMMethod.book:
            D2 = self.compute_stats_hard(x2, np.arange(x2.shape[0]))
            return self.llr_NvsM_book(D1, D2)
        if method == PLDALLRNvsMMethod.vavg:
            return self.llr_Nvs1_vavg(D1, x2, do_lnorm=False)
        if method == PLDALLRNvsMMethod.lnorm_vavg:
            return self.llr_Nvs1_vavg(D1, x2, do_lnorm=True)

        raise ValueError(f"wrong llr {method}")

    def llr_Nvs1_vavg(
        self,
        D1: Tuple[np.ndarray, np.ndarray, np.ndarray],
        x2: np.ndarray,
        do_lnorm: bool = True,
    ) -> np.ndarray:
        """Vector-averaged N-vs-1 scoring.

        Args:
            D1: Tuple ``(N1, F1, S1)`` describing each enrollment side.
            x2: Test embeddings with shape ``(num_test_segments, x_dim)``.
            do_lnorm: Whether to apply length normalization after averaging.

        Returns:
            Score matrix of shape ``(num_enroll_sides, num_test_segments)``.
        """
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_Nvs1_savg(
        self, x1: np.ndarray, ids1: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:
        """Score-averaged N-vs-1 scoring.

        Args:
            x1: Enrollment embeddings.
            ids1: Mapping from enrollment segments to enrollment sides.
            x2: Test embeddings.

        Returns:
            Score matrix with shape ``(num_enroll_sides, num_test_segments)``.
        """
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores = F / N[:, None]
        return scores

    def sample(
        self,
        num_classes: int,
        num_samples_per_class: int,
        rng: Optional[Generator] = None,
        seed: int = 1024,
    ) -> np.ndarray:
        """Draws samples from the PLDA model.

        Args:
            num_classes: Number of classes to simulate.
            num_samples_per_class: Number of samples per class.
            rng: Optional random number generator.
            seed: Used if ``rng`` is ``None``.

        Returns:
            Samples with shape ``(num_classes * num_samples_per_class, x_dim)``.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"y_dim": self.y_dim, "update_mu": self.update_mu}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def weighted_avg_params(self, mu: np.ndarray, w_mu: float) -> None:
        """Averages this model's mean with another mean vector."""
        self.mu = w_mu * mu + (1 - w_mu) * self.mu

    def weighted_avg_model(self, plda: "PLDABase", **kwargs) -> None:
        """Placeholder for averaging this model with another PLDA model."""
        pass
