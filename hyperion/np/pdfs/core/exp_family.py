"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .pdf import PDF


class ExpFamily(PDF):
    """Base class for exponential family distributions.

    Models follow the canonical form ``p(x) = h(x) exp(eta^T u(x) - A)``, where
    ``eta`` are the natural parameters, ``u(x)`` the sufficient statistics, and
    ``A`` the log-normalizer.
    """

    def __init__(self, eta: Optional[np.ndarray] = None, **kwargs: Any) -> None:
        """Initializes an exponential-family model.

        Args:
            eta: Optional natural-parameter vector supplied at construction.
            **kwargs: Extra keyword arguments forwarded to :class:`PDF`.
        """
        super().__init__(**kwargs)
        self.eta: Optional[np.ndarray] = eta
        self.A: Optional[float] = None

    @property
    def is_init(self) -> bool:
        """Indicates whether the model parameters are consistent."""
        if not self._is_init:
            self._compute_nat_std()
            if self.eta is not None and self.A is not None:
                self.validate()
                self._is_init = True
        return self._is_init

    def fit(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Trains the model using sufficient-statistic accumulation.

        Args:
            x: Training samples of shape ``(num_samples, x_dim)``.
            sample_weight: Optional per-sample weights for the training set.
            x_val: Optional validation samples.
            sample_weight_val: Optional validation weights.
            batch_size: If provided, accumulate statistics in mini-batches of
                this size.

        Returns:
            List containing training ELBO, average training ELBO, and
            optionally the validation ELBO and its per-sample value.
        """

        N, u_x = self.Estep(x=x, sample_weight=sample_weight, batch_size=batch_size)
        self.Mstep(N, u_x)
        elbo = self.elbo(x, N=N, u_x=u_x)
        elbo = [elbo, elbo / N]

        if x_val is not None:
            N, u_x = self.Estep(
                x=x_val, sample_weight=sample_weight_val, batch_size=batch_size
            )
            elbo_val = self.elbo(x_val, N=N, u_x=u_x)
            elbo += [elbo_val, elbo_val / N]
        return elbo

    def log_h(self, x: np.ndarray) -> np.ndarray:
        """Computes the log base measure ``log h(x)``."""
        return 0

    def accum_log_h(
        self, x: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Accumulates ``log h(x)`` over samples.

        Args:
            x: Data matrix of shape ``(num_samples, x_dim)``.
            sample_weight: Optional weights to scale the contributions.

        Returns:
            Sum of ``log h(x)`` over the weighted samples.
        """
        if sample_weight is None:
            return np.sum(self.log_h(x))
        return np.sum(sample_weight * self.log_h(x))

    def compute_suff_stats(self, x: np.ndarray) -> np.ndarray:
        """Computes sufficient statistics for a batch of samples."""
        return x

    def accum_suff_stats(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Accumulates sufficient statistics over multiple samples.

        Args:
            x: Samples with shape ``(num_samples, x_dim)``.
            u_x: Optional precomputed sufficient statistics of shape
                ``(num_samples, u_dim)``.
            sample_weight: Optional weights applied during accumulation.
            batch_size: If provided, process the data in mini-batches.

        Returns:
            Weighted count ``N`` and the accumulated statistics ``∑ u(x)``.
        """
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_1batch(x, u_x, sample_weight)
        else:
            return self._accum_suff_stats_nbatches(x, sample_weight, batch_size)

    def _accum_suff_stats_1batch(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Accumulates sufficient statistics for a single batch.

        Args:
            x: Samples in the batch.
            u_x: Optional sufficient statistics computed for ``x``.
            sample_weight: Optional sample weights aligned with ``x``.

        Returns:
            Tuple ``(N, stats)`` with the effective sample count and accumulated
            sufficient statistics.
        """
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        if sample_weight is None:
            N = u_x.shape[0]
        else:
            u_x = u_x * sample_weight[:, None]
            N = np.sum(sample_weight)
        acc_u_x = np.sum(u_x, axis=0)
        return N, acc_u_x

    def _accum_suff_stats_nbatches(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray],
        batch_size: int,
    ) -> Tuple[float, np.ndarray]:
        """Accumulates sufficient statistics across mini-batches.

        Args:
            x: Full dataset of shape ``(num_samples, x_dim)``.
            sample_weight: Optional per-sample weights.
            batch_size: Size of the blocks used to accumulate statistics.

        Returns:
            Tuple ``(N, stats)`` combining statistics from all batches.
        """
        sw_i = None
        for i1 in range(0, x.shape[0], batch_size):
            i2 = np.minimum(i1 + batch_size, x.shape[0])
            x_i = x[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_1batch(x_i, sample_weight=sw_i)
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x

    def sum_suff_stats(
        self, N: Sequence[float], u_x: Sequence[np.ndarray]
    ) -> Tuple[float, np.ndarray]:
        """Aggregates sufficient statistics from multiple workers.

        Args:
            N: Iterable of zeroth-order statistics (counts) per worker.
            u_x: Iterable of accumulated sufficient statistics per worker.

        Returns:
            Tuple ``(N_total, u_total)`` with the combined values.
        """
        assert len(N) == len(u_x)
        acc_N = N[0]
        acc_u_x = u_x[0].copy()
        for i in range(1, len(N)):
            acc_N += N[i]
            acc_u_x += u_x[i]
        return acc_N, acc_u_x

    def Estep(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Expectation step that accumulates sufficient statistics.

        Args:
            x: Input samples.
            u_x: Optional precomputed sufficient statistics.
            sample_weight: Optional weights for each sample.
            batch_size: Optional batch size when aggregating statistics.

        Returns:
            Tuple ``(N, stats)`` analogous to :meth:`accum_suff_stats`.
        """
        return self.accum_suff_stats(x, u_x, sample_weight, batch_size)

    def Mstep(self, N: float, u_x: np.ndarray) -> None:
        """Maximization step (override in subclasses).

        Args:
            N: Zeroth-order statistic (effective sample count).
            u_x: Accumulated sufficient statistics.
        """
        pass

    def elbo(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        N: float = 1,
        log_h: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """Computes the evidence lower bound (ELBO).

        Args:
            x: Samples whose likelihood is evaluated.
            u_x: Optional accumulated sufficient statistics.
            N: Zeroth-order statistic (effective number of samples).
            log_h: Optional accumulated base measure term.
            sample_weight: Optional per-sample weights.
            batch_size: Batch size used if ``u_x`` is not provided.

        Returns:
            Scalar ELBO value ``log p(X)``.
        """
        assert self.is_init
        if u_x is None:
            N, u_x = self.accum_suff_stats(
                x, sample_weight=sample_weight, batch_size=batch_size
            )
        if log_h is None:
            log_h = self.accum_log_h(x, sample_weight=sample_weight)
        return log_h + np.inner(u_x, self.eta) - N * self.A

    def log_prob(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None, method: str = "nat"
    ) -> np.ndarray:
        """Evaluates the log-probability of each sample.

        Args:
            x: Input data of shape ``(num_samples, x_dim)``.
            u_x: Optional sufficient statistics for ``x``.
            method: Whether to use natural (``"nat"``) or standard (``"std"``)
                parameters.

        Returns:
            Array with ``log p(x)`` for each sample.
        """
        if method == "nat":
            return self.log_prob_nat(x, u_x)
        else:
            return self.log_prob_std(x)

    def log_prob_nat(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes ``log p(x)`` using the natural parameters."""
        assert self.is_init
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        return self.log_h(x) + np.inner(u_x, self.eta) - self.A

    @staticmethod
    def compute_A_nat(eta: np.ndarray) -> float:
        """Computes the log-normalizer ``A`` from natural parameters."""
        raise NotImplementedError()

    @staticmethod
    def compute_A_std(params: np.ndarray) -> float:
        """Computes the log-normalizer ``A`` from standard parameters."""
        raise NotImplementedError()

    @staticmethod
    def compute_eta(param: np.ndarray) -> np.ndarray:
        """Converts standard parameters to natural parameters."""
        raise NotImplementedError()

    @staticmethod
    def compute_std(eta: np.ndarray) -> np.ndarray:
        """Converts natural parameters to standard parameters."""
        raise NotImplementedError()

    def _compute_nat_params(self) -> None:
        """Derives ``eta`` and ``A`` from the standard parameters."""
        pass

    def _compute_std_params(self) -> None:
        """Derives the standard parameters from ``eta``."""
        pass

    def _compute_nat_std(self) -> None:
        """Keeps standard and natural parameterizations synchronized."""
        pass

    def validate(self) -> None:
        """Checks that parameters describe a valid distribution."""
        pass
