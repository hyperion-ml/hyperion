"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ....hyp_defs import float_cpu
from ....utils.math_funcs import logsumexp, softmax
from ..core import PDF

NBestType = Union[int, NDArray[np.intp]]


class ExpFamilyMixture(PDF):
    """Mixture of exponential-family distributions.

    Attributes:
        num_comp: Number of mixture components.
        pi: Component weights with shape ``(num_comp,)``.
        eta: Natural parameters for each component.
        min_N: Minimum effective count below which components are pruned.
        update_pi: Whether to update mixture weights during training.
    """

    def __init__(
        self,
        num_comp: int = 1,
        pi: Optional[np.ndarray] = None,
        eta: Optional[np.ndarray] = None,
        min_N: float = 0,
        update_pi: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the mixture model.

        Args:
            num_comp: Number of components when ``pi`` is not provided.
            pi: Optional mixture weights.
            eta: Optional natural parameters for each component.
            min_N: Minimum responsibility mass required to keep a component.
            update_pi: Whether to update ``pi`` during EM.
            **kwargs: Extra arguments forwarded to :class:`PDF`.
        """
        super().__init__(**kwargs)
        if pi is not None:
            num_comp = len(pi)
        self.num_comp: int = num_comp
        self.pi: Optional[np.ndarray] = pi
        self.eta: Optional[np.ndarray] = eta
        self.min_N: float = min_N
        self.A: Optional[np.ndarray] = None
        self._log_pi: Optional[np.ndarray] = None
        self.update_pi: bool = update_pi

    @property
    def is_init(self) -> bool:
        """bool: Whether the mixture parameters have been initialized."""
        if not self._is_init:
            if self.eta is not None and self.A is not None and self.pi is not None:
                self.validate()
                self._is_init = True
        return self._is_init

    @property
    def log_pi(self) -> np.ndarray:
        """np.ndarray: Logarithm of the component weights."""
        if self._log_pi is None:
            self._log_pi = np.log(self.pi + 1e-15)
        return self._log_pi

    def _validate_pi(self) -> None:
        assert len(self.pi) == self.num_comp

    def fit(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: Optional[int] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Trains the mixture via EM.

        Args:
            x: Training data of shape ``(num_samples, x_dim)``.
            sample_weight: Optional per-sample weights.
            x_val: Optional validation data.
            sample_weight_val: Optional validation weights.
            epochs: Number of EM iterations.
            batch_size: Accumulation batch size for statistics.

        Returns:
            Training ELBO trace and per-sample ELBO, optionally followed by the
            validation counterparts.
        """

        if not self.is_init:
            self.initialize(x)

        log_h = self.accum_log_h(x, sample_weight)
        if x_val is not None:
            log_h_val = self.accum_log_h(x_val, sample_weight_val)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            N, u_x = self.Estep(x=x, sample_weight=sample_weight, batch_size=batch_size)
            elbo[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h)
            self.Mstep(N, u_x)

            if x_val is not None:
                N, u_x = self.Estep(
                    x=x_val, sample_weight=sample_weight_val, batch_size=batch_size
                )
                elbo_val[epoch] = self.elbo(None, N=N, u_x=u_x, log_h=log_h_val)

        if x_val is None:
            return elbo, elbo / x.shape[0]
        else:
            return elbo, elbo / x.shape[0], elbo_val, elbo_val / x_val.shape[0]

    def log_h(self, x: np.ndarray) -> np.ndarray:
        """Computes the log base measure ``log h(x)``.

        Args:
            x: Input samples with shape ``(num_samples, x_dim)``.

        Returns:
            Array of ``log h(x)`` evaluations for each sample.
        """
        return 0

    def accum_log_h(
        self, x: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Accumulates ``log h(x)`` over the dataset.

        Args:
            x: Input samples.
            sample_weight: Optional weights per sample.

        Returns:
            Weighted sum of ``log h(x)``.
        """
        if sample_weight is None:
            return np.sum(self.log_h(x))
        return np.sum(sample_weight * self.log_h(x))

    def compute_pz(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None, mode: str = "nat"
    ) -> np.ndarray:
        """Computes posterior responsibilities ``p(z | x)``.

        Args:
            x: Input samples of shape ``(num_samples, x_dim)``.
            u_x: Optional sufficient statistics for ``x``.
            mode: Whether to use natural (``"nat"``) or standard (``"std"``) params.

        Returns:
            Array of responsibilities with shape ``(num_samples, num_comp)``.
        """
        if mode == "nat":
            return self.compute_pz_nat(x, u_x)
        else:
            return self.compute_pz_std(x)

    def compute_pz_nat(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes responsibilities using natural parameters.

        Args:
            x: Samples used to evaluate responsibilities.
            u_x: Optional sufficient statistics of ``x``.

        Returns:
            Responsibility matrix of shape ``(num_samples, num_comp)``.
        """
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        logr = np.dot(u_x, self.eta.T) - self.A + self.log_pi
        return softmax(logr)

    def compute_pz_std(self, x: np.ndarray) -> np.ndarray:
        """Computes responsibilities using standard parameters.

        Args:
            x: Samples whose responsibilities are computed.

        Returns:
            Responsibility matrix of shape ``(num_samples, num_comp)``.
        """
        return self.compute_pz_nat(x)

    def compute_suff_stats(self, x: np.ndarray) -> np.ndarray:
        """Computes sufficient statistics for a batch.

        Args:
            x: Samples with shape ``(num_samples, x_dim)``.

        Returns:
            Array containing sufficient statistics for each sample.
        """
        return x

    def accum_suff_stats(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates sufficient statistics over all samples.

        Args:
            x: Data matrix of shape ``(num_samples, x_dim)``.
            u_x: Optional sufficient statistics with shape ``(num_samples, u_dim)``.
            sample_weight: Optional weights per sample.
            batch_size: If provided, accumulate in mini-batches.

        Returns:
            Tuple ``(N, U)`` with zero-order stats and accumulated suff. stats.
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates statistics for a single batch."""
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        N = np.sum(z, axis=0)
        acc_u_x = np.dot(z.T, u_x)
        return N, acc_u_x

    def _accum_suff_stats_nbatches(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray],
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates statistics across multiple batches."""
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

    def accum_suff_stats_segments(
        self,
        x: np.ndarray,
        segments: Sequence[Sequence[int]],
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates statistics separately for each segment.

        Args:
            x: Input samples.
            segments: Start/end frame indices per segment.
            u_x: Optional sufficient statistics for ``x``.
            sample_weight: Optional weights per frame.
            batch_size: Passed to :meth:`accum_suff_stats`.

        Returns:
            Tuple ``(N, U)`` where each entry has length ``num_segments``.
        """
        K = self.num_comp
        num_segments = len(segments)
        N = np.zeros((num_segments, K), dtype=float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), dtype=float_cpu())
        u_x_i = None
        sw_i = None
        for i in range(num_segments):
            start = int(segments[i][0])
            end = int(segments[i][1]) + 1
            x_i = x[start:end]
            if u_x is not None:
                u_x_i = u_x[start:end]
            if sample_weight is not None:
                sw_i = sample_weight[start:end]
            N_i, acc_u_x_i = self.accum_suff_stats(
                x_i, u_x=u_x_i, sample_weight=sw_i, batch_size=batch_size
            )
            N[i] = N_i
            acc_u_x[i] = acc_u_x_i

        return N, acc_u_x

    def accum_suff_stats_segments_prob(
        self,
        x: np.ndarray,
        prob: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates statistics weighted by segment membership probabilities.

        Args:
            x: Data samples.
            prob: Segment probabilities with shape ``(num_samples, num_segments)``.
            u_x: Optional sufficient statistics.
            sample_weight: Optional sample weights.
            batch_size: Passed to the helper accumulation functions.

        Returns:
            Tuple ``(N, U)`` of per-segment stats.
        """
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_segments_prob_1batch(
                x, prob, u_x, sample_weight
            )
        else:
            return self._accum_suff_stats_segments_prob_nbatches(
                x, prob, sample_weight, batch_size
            )

    def _accum_suff_stats_segments_prob_1batch(
        self,
        x: np.ndarray,
        prob: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        K = len(self.pi)
        num_segments = prob.shape[1]
        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        for i in range(num_segments):
            z_i = z * prob[:, i][:, None]
            N[i] = np.sum(z_i, axis=0)
            acc_u_x[i] = np.dot(z_i.T, u_x)

        return N, acc_u_x

    def _accum_suff_stats_segments_prob_nbatches(
        self,
        x: np.ndarray,
        prob: np.ndarray,
        sample_weight: Optional[np.ndarray],
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sw_i = None
        for i1 in range(0, x.shape[0], batch_size):
            i2 = np.minimum(i1 + batch_size, x.shape[0])
            x_i = x[i1:i2, :]
            prob_i = prob[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_segments_prob_1batch(
                x_i, prob_i, sample_weight=sw_i
            )
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x

    def accum_suff_stats_sorttime(
        self,
        x: np.ndarray,
        frame_length: int,
        frame_shift: int,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulates statistics over sliding windows.

        Args:
            x: Data samples.
            frame_length: Window length in frames.
            frame_shift: Hop size between windows.
            u_x: Optional sufficient statistics.
            sample_weight: Optional sample weights.
            batch_size: Used when delegating to batch accumulation routines.

        Returns:
            Per-window zero-order stats and sufficient statistics.
        """
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_sorttime_1batch(
                x, frame_length, frame_shift, u_x, sample_weight
            )
        else:
            return self._accum_suff_stats_sorttime_nbatches(
                x, frame_length, frame_shift, sample_weight, batch_size
            )

    def _accum_suff_stats_sorttime_1batch(
        self,
        x: np.ndarray,
        frame_length: int,
        frame_shift: int,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = len(self.pi)
        num_frames = x.shape[0]
        num_segments = int(np.floor((num_frames - frame_length) / frame_shift + 1))
        if num_segments <= 1:
            return self._accum_suff_stats_1batch(x, u_x, sample_weight)

        if u_x is None:
            u_x = self.compute_suff_stats(x)
        z = self.compute_pz_nat(x, u_x)
        if sample_weight is not None:
            z *= sample_weight[:, None]

        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        start1 = int(frame_shift - 1)
        end1 = int((num_segments - 1) * frame_shift)
        start2 = int(start1 + frame_length)
        end2 = int(end1 + frame_length)
        cum_N = np.cumsum(z, axis=0)
        N[0] = cum_N[frame_length - 1]
        N[1:] = cum_N[start2:end2:frame_shift] - cum_N[start1:end1:frame_shift]

        for k in range(K):
            cum_u_x_k = np.cumsum(z[:, k][:, None] * u_x, axis=0)
            acc_u_x[0, k] = cum_u_x_k[frame_length - 1]
            acc_u_x[1:, k] = (
                cum_u_x_k[start2:end2:frame_shift] - cum_u_x_k[start1:end1:frame_shift]
            )

        return N, acc_u_x

    def _accum_suff_stats_sorttime_nbatches(
        self,
        x: np.ndarray,
        frame_length: int,
        frame_shift: int,
        sample_weight: Optional[np.ndarray],
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = len(self.pi)
        num_frames = x.shape[0]
        num_segments = int(np.floor((num_frames - frame_length) / frame_shift + 1))
        if num_segments <= 1:
            return self._accum_suff_stats_1batch(x, None, sample_weight)

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        num_segments_per_batch = min(int(batch_size), num_segments)

        N = np.zeros((num_segments, K), float_cpu())
        acc_u_x = np.zeros((num_segments, K, self.eta.shape[1]), float_cpu())

        sw_i = None
        for seg_start in range(0, num_segments, num_segments_per_batch):
            seg_end = min(seg_start + num_segments_per_batch, num_segments)
            num_segments_i = seg_end - seg_start
            i1 = seg_start * frame_shift
            i2 = i1 + (num_segments_i - 1) * frame_shift + frame_length
            x_i = x[i1:i2, :]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_sorttime_1batch(
                x_i, frame_length, frame_shift, sample_weight=sw_i
            )
            N[seg_start:seg_end] = N_i
            acc_u_x[seg_start:seg_end] = u_x_i
        return N, acc_u_x

    def Estep(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expectation step returning accumulated stats."""
        return self.accum_suff_stats(x, u_x, sample_weight, batch_size)

    def sum_suff_stats(
        self, N: Sequence[np.ndarray], u_x: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sums statistics across multiple workers."""
        assert len(N) == len(u_x)
        acc_N = N[0]
        acc_u_x = u_x[0]
        for i in range(1, len(N)):
            acc_N += N[i]
            acc_u_x += u_x[i]
        return acc_N, acc_u_x

    def Mstep(self, stats: Tuple[np.ndarray, np.ndarray]) -> None:
        """Maximization step (override in subclasses)."""
        pass

    def elbo(
        self,
        x: Optional[np.ndarray],
        u_x: Optional[np.ndarray] = None,
        N: Optional[np.ndarray] = None,
        log_h: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """Computes the evidence lower bound."""
        if u_x is None:
            N, u_x = self.accum_suff_stats(
                x, sample_weight=sample_weight, batch_size=batch_size
            )
        if log_h is None:
            log_h = self.accum_log_h(x, sample_weight=sample_weight)
        return log_h + np.sum(u_x * self.eta) + np.inner(N, self.log_pi - self.A)

    def log_prob(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None, mode: str = "nat"
    ) -> np.ndarray:
        """Computes ``log p(x)`` under the mixture.

        Args:
            x: Input samples.
            u_x: Optional sufficient statistics for ``x``.
            mode: Whether to evaluate using natural or standard parameters.

        Returns:
            Array of log-likelihoods with shape ``(num_samples,)``.
        """
        if mode == "nat":
            return self.log_prob_nat(x, u_x)
        else:
            return self.log_prob_std(x)

    def log_prob_nat(
        self, x: np.ndarray, u_x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes ``log p(x)`` using natural parameters.

        Args:
            x: Input samples.
            u_x: Optional sufficient statistics for ``x``.

        Returns:
            Log-likelihood per sample.
        """
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        llk_k = np.dot(u_x, self.eta.T) - self.A + self.log_pi
        llk = logsumexp(llk_k)
        return self.log_h(x) + llk

    def log_prob_std(self, x: np.ndarray) -> np.ndarray:
        """Computes ``log p(x)`` using standard parameters.

        Args:
            x: Input samples whose likelihood is evaluated.

        Returns:
            Log-likelihood per sample.
        """
        raise NotImplementedError()

    def log_prob_nbest(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        mode: str = "nat",
        nbest_mode: str = "master",
        nbest: NBestType = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes ``log p(x)`` using only the top-N components.

        Args:
            x: Input samples.
            u_x: Optional sufficient statistics.
            mode: Whether to use natural or standard parameters.
            nbest_mode: Strategy for selecting the components.
            nbest: In ``"master"`` mode, number of top components. In other
                modes, per-sample component indices with shape
                ``(num_samples, nbest)``.

        Returns:
            In ``nbest_mode="master"``, returns ``(log_likelihood, top_idx)``
            where ``top_idx`` stores the selected component indices per sample.
            In other modes, returns only the log-likelihood per sample.
        """
        if mode == "nat":
            return self.log_prob_nbest_nat(x, u_x, nbest_mode=nbest_mode, nbest=nbest)
        else:
            return self.log_prob_nbest_std(x, nbest_mode=nbest_mode, nbest=nbest)

    def log_prob_nbest_nat(
        self,
        x: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        nbest_mode: str = "master",
        nbest: NBestType = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Top-N log-probability computation using natural parameters.

        Args:
            x: Input samples.
            u_x: Optional sufficient statistics.
            nbest_mode: Selection strategy, e.g., ``"master"`` or ``"ubm"``.
            nbest: Number of components or explicit indices.

        Returns:
            If ``nbest_mode == "master"``, returns ``(llk, top_idx)`` where
            ``top_idx`` are the selected component indices for each sample.
            Otherwise returns only ``llk``.
        """
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        if nbest_mode == "master":
            assert isinstance(nbest, int)
            assert nbest > 0
            llk_k = np.dot(u_x, self.eta.T) - self.A + self.log_pi
            nbest_eff = min(nbest, self.num_comp)
            if nbest_eff < self.num_comp:
                top_idx = np.argpartition(llk_k, -nbest_eff, axis=1)[:, -nbest_eff:]
            else:
                top_idx = np.tile(
                    np.arange(self.num_comp, dtype=np.int64), (x.shape[0], 1)
                )

            llk_sel = np.take_along_axis(llk_k, top_idx, axis=1)
            sort_idx = np.argsort(llk_sel, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(top_idx, sort_idx, axis=1)
            llk_k = np.take_along_axis(llk_sel, sort_idx, axis=1)
        else:
            nbest_idx = np.asarray(nbest, dtype=np.intp)
            if nbest_idx.ndim != 2 or nbest_idx.shape[0] != x.shape[0]:
                raise ValueError(
                    "for nbest_mode!='master', nbest must have shape "
                    "(num_samples, nbest)"
                )
            llk_k = (
                np.einsum("nd,nkd->nk", u_x, self.eta[nbest_idx])
                - self.A[nbest_idx]
                + self.log_pi[nbest_idx]
            )
        llk = logsumexp(llk_k, axis=-1)
        llk = self.log_h(x) + llk
        if nbest_mode == "master":
            return llk, top_idx
        return llk

    def log_prob_nbest_std(
        self,
        x: np.ndarray,
        nbest_mode: str = "master",
        nbest: NBestType = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Top-N log-probability computation using standard parameters.

        Args:
            x: Input samples.
            nbest_mode: Selection strategy matching :meth:`log_prob_nbest_nat`.
            nbest: Number of components or explicit indices.

        Returns:
            In ``nbest_mode="master"``, returns ``(log_likelihood, top_idx)``
            with per-sample selected component indices. In other modes, returns
            only log-likelihood per sample.
        """
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary."""
        config = {"min_n": self.min_N, "update_pi": self.update_pi}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def compute_A_nat(eta):
        """Computes A_theta from the natural param."""
        raise NotImplementedError()

    @staticmethod
    def compute_A_std(params):
        """Computes A_theta from the standard param."""
        raise NotImplementedError()

    @staticmethod
    def compute_eta(param):
        """Computes the natural param. from the standard param."""
        raise NotImplementedError()

    @staticmethod
    def compute_std(eta):
        """Computes the standard param. from the natural param."""
        raise NotImplementedError()

    def _compute_nat_params(self):
        pass

    def _compute_std_params(self):
        pass
