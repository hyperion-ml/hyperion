"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import Generator, RandomState

from ....hyp_defs import float_cpu
from ....utils.math_funcs import logsumexp, softmax
from ..core import PDF


class HMM(PDF):
    """Hidden Markov Model with discrete latent states.

    Attributes:
        num_states: Number of latent states in the chain.
        pi: Initial state distribution with shape ``(num_states,)``.
        trans: Transition probability matrix with shape
            ``(num_states, num_states)``.
        trans_mask: Optional mask that zeroes out prohibited transitions.
        update_pi: Whether to update ``pi`` during training.
        update_trans: Whether to update ``trans`` during training.
        tied_trans: Whether to tie transitions via a loop probability.
        left_to_right: Whether to enforce left-to-right topology.

    Example:
        >>> import numpy as np
        >>> from hyperion.np.pdfs.hmm.hmm import HMM
        >>> rng = np.random.default_rng(1234)
        >>> hmm = HMM(num_states=3, left_to_right=True)
        >>> x_train = np.array(
        ...     [np.log(rng.random((20, 3)).astype(np.float32) + 1e-3) for _ in range(5)],
        ...     dtype=object,
        ... )
        >>> elbo, elbo_per_frame = hmm.fit(x_train, epochs=2)
        >>> x_test = np.log(rng.random((20, 3)).astype(np.float32) + 1e-3)
        >>> pz = hmm.compute_pz(x_test)
        >>> paths, scores = hmm.viterbi_decode(x_test, nbest=2)
        >>> pz.shape, paths.shape, scores.shape
        ((20, 3), (2, 20), (2,))
    """

    def __init__(
        self,
        num_states: int = 1,
        pi: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
        trans_mask: Optional[np.ndarray] = None,
        update_pi: bool = True,
        update_trans: bool = True,
        tied_trans: bool = False,
        left_to_right: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the HMM.

        Args:
            num_states: Number of states when ``pi`` is not provided.
            pi: Optional initial-state probabilities.
            trans: Optional transition matrix.
            trans_mask: Optional boolean mask for transitions.
            update_pi: Whether EM updates ``pi``.
            update_trans: Whether EM updates ``trans``.
            tied_trans: Whether to tie transition probabilities.
            left_to_right: If ``True`` forces upper-triangular transitions.
            **kwargs: Extra keyword arguments forwarded to :class:`PDF`.
        """
        super().__init__(**kwargs)
        if pi is not None:
            num_states = len(pi)

        self.num_states: int = num_states
        self.pi: Optional[np.ndarray] = pi
        self.trans: Optional[np.ndarray] = trans
        self.trans_mask: Optional[np.ndarray] = trans_mask

        self.update_pi: bool = update_pi
        self.update_trans: bool = update_trans
        self.tied_trans: bool = tied_trans
        self.left_to_right: bool = left_to_right

        if left_to_right and (trans_mask is None):
            self.trans_mask = np.triu(
                np.ones((num_states, num_states), dtype=float_cpu())
            )

        self._log_pi: Optional[np.ndarray] = None
        self._log_trans: Optional[np.ndarray] = None

    def reset_aux(self) -> None:
        """Invalidates cached log-probabilities."""
        self._log_pi = None
        self._log_trans = None

    @property
    def is_init(self) -> bool:
        """bool: Whether the transition parameters are valid."""
        if self._is_init:
            return True

        if self.pi is None or self.trans is None:
            return False

        # Route through initialize() so masks/topology constraints are applied
        # even when parameters were provided at construction time.
        self.initialize()

        return self._is_init

    @property
    def log_pi(self) -> np.ndarray:
        """np.ndarray: Cached ``log(pi)``."""
        if self._log_pi is None:
            self._log_pi = np.log(self.pi + 1e-15)
        return self._log_pi

    @property
    def log_trans(self) -> np.ndarray:
        """np.ndarray: Cached ``log(trans)``."""
        if self._log_trans is None:
            self._log_trans = np.log(self.trans + 1e-15)
        return self._log_trans

    def validate(self) -> None:
        """Checks that ``pi``, ``trans`` and masks have consistent shapes."""
        if self.pi is None or self.trans is None:
            raise ValueError("pi and trans must be initialized")
        if self.pi.ndim != 1 or len(self.pi) != self.num_states:
            raise ValueError(
                f"pi shape must be ({self.num_states},), got {self.pi.shape}"
            )
        if self.trans.ndim != 2 or self.trans.shape != (
            self.num_states,
            self.num_states,
        ):
            raise ValueError(
                "trans shape must be "
                f"({self.num_states}, {self.num_states}), got {self.trans.shape}"
            )
        if not np.all(np.isfinite(self.pi)):
            raise ValueError("pi must contain finite values")
        if np.any(self.pi < 0):
            raise ValueError("pi must be non-negative")
        if not np.all(np.isfinite(self.trans)):
            raise ValueError("trans must contain finite values")
        if np.any(self.trans < 0):
            raise ValueError("trans must be non-negative")
        pi_sum = np.sum(self.pi)
        if not np.isclose(pi_sum, 1.0, atol=1e-6):
            raise ValueError(f"pi must sum to 1, got {pi_sum}")
        trans_row_sums = np.sum(self.trans, axis=-1)
        if not np.allclose(trans_row_sums, 1.0, atol=1e-6):
            raise ValueError("each row of trans must sum to 1")
        if self.trans_mask is not None:
            if self.trans_mask.shape != self.trans.shape:
                raise ValueError(
                    "trans_mask shape must match trans, got "
                    f"{self.trans_mask.shape} vs {self.trans.shape}"
                )
            if not np.all(np.isfinite(self.trans_mask)):
                raise ValueError("trans_mask must contain finite values")
            if np.any(self.trans_mask < 0):
                raise ValueError("trans_mask must be non-negative")

    def _validate_seq_llk(self, x: np.ndarray, name: str = "x") -> None:
        """Validates a per-frame log-likelihood matrix."""
        if x.ndim != 2:
            raise ValueError(
                f"{name} must be 2D with shape (T, num_states), got {x.shape}"
            )
        if x.shape[1] != self.num_states:
            raise ValueError(
                f"{name}.shape[1] ({x.shape[1]}) != num_states ({self.num_states})"
            )
        if x.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one frame")

    def _require_initialized(self, caller: str) -> None:
        """Raises an error when HMM parameters are not initialized."""
        if not self.is_init:
            raise ValueError(f"HMM parameters must be initialized before {caller}")

    def initialize(self) -> None:
        """Initializes missing HMM parameters with valid probabilities."""
        if self.pi is None:
            self.pi = np.ones((self.num_states,), dtype=float_cpu()) / self.num_states

        if self.trans is None:
            self.trans = (
                np.ones((self.num_states, self.num_states), dtype=float_cpu())
                / self.num_states
            )

        if self.trans_mask is not None:
            self.trans *= self.trans_mask
            row_sums = np.sum(self.trans, axis=-1, keepdims=True)
            if np.any(row_sums == 0):
                raise ValueError(
                    "trans_mask creates states with no outgoing transitions"
                )
            self.trans /= row_sums

        self.validate()
        self._is_init = True
        self.reset_aux()

    def fit(
        self,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        epochs: int = 10,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Trains the HMM via EM.

        Args:
            x: Array of training sequences, each entry holding emission log-likelihoods.
            sample_weight: Currently unused placeholder for per-sequence weights.
            x_val: Optional validation sequences.
            sample_weight_val: Currently unused placeholder for validation weights.
            epochs: Number of EM passes over the data.

        Returns:
            ``(elbo, elbo_per_frame)`` if ``x_val`` is ``None``; otherwise also
            returns ``(elbo_val, elbo_val_per_frame)``.
        """
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sequence")
        for i in range(x.shape[0]):
            self._validate_seq_llk(x[i], name=f"x[{i}]")

        if x_val is not None:
            for i in range(x_val.shape[0]):
                self._validate_seq_llk(x_val[i], name=f"x_val[{i}]")

        if not self.is_init:
            self.initialize()

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):
            stats = None
            prev_Nzz = np.zeros((self.num_states, self.num_states), dtype=float_cpu())
            for i in range(x.shape[0]):
                pz, stats = self.Estep(x[i], stats_0=stats)
                _, Nzz = stats
                cur_Nzz = Nzz - prev_Nzz
                prev_Nzz = Nzz.copy()
                elbo[epoch] += self.elbo(x[i], pz=pz, Nzz=cur_Nzz)

            self.Mstep(stats)

            if x_val is not None:
                for i in range(x_val.shape[0]):
                    pz, stats = self.Estep(x_val[i])
                    _, Nzz = stats
                    elbo_val[epoch] += self.elbo(x_val[i], pz=pz, Nzz=Nzz)

        N_tot = np.sum([x_i.shape[0] for x_i in x])
        if N_tot <= 0:
            raise ValueError("x must contain at least one frame")
        if x_val is None:
            return elbo, elbo / N_tot
        else:
            N_val_tot = np.sum([x_i.shape[0] for x_i in x_val])
            if N_val_tot <= 0:
                raise ValueError("x_val must contain at least one frame")
            return elbo, elbo / N_tot, elbo_val, elbo_val / N_val_tot

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Runs the forward recursion.

        Args:
            x: Per-frame log-likelihoods with shape ``(T, num_states)``.

        Returns:
            Log forward messages ``alpha`` with the same shape.
        """
        self._require_initialized("forward")
        self._validate_seq_llk(x)
        # x = log P(x|z)
        N = x.shape[0]
        log_alpha = np.zeros((N, self.num_states), dtype=float_cpu())
        log_alpha[0] = self.log_pi + x[0]
        for n in range(1, N):
            log_alpha[n] = x[n] + logsumexp(
                log_alpha[n - 1][:, None] + self.log_trans, axis=0
            )

        return log_alpha

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Runs the backward recursion.

        Args:
            x: Per-frame log-likelihoods with shape ``(T, num_states)``.

        Returns:
            Log backward messages ``beta`` with the same shape.
        """
        self._require_initialized("backward")
        self._validate_seq_llk(x)
        N = x.shape[0]
        log_beta = np.zeros((N, self.num_states), dtype=float_cpu())
        # In log-domain, beta at the final frame is log(1)=0.
        log_beta[-1] = 0.0
        for n in range(N - 2, -1, -1):
            r = log_beta[n + 1] + x[n + 1] + self.log_trans
            log_beta[n] = logsumexp(r.T, axis=0)

        return log_beta

    def compute_pz(
        self,
        x: np.ndarray,
        return_Nzz: bool = False,
        return_log_px: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Computes posterior over states and optional statistics.

        Args:
            x: Per-frame log-likelihoods with shape ``(T, num_states)``.
            return_Nzz: Whether to also return expected transition counts.
            return_log_px: Whether to also return ``log p(x)``.

        Returns:
            Posterior over states ``pz`` and optionally ``Nzz`` and ``log_px``.
        """
        self._require_initialized("compute_pz")
        self._validate_seq_llk(x)
        log_alpha = self.forward(x)
        log_beta = self.backward(x)
        log_px = logsumexp(log_alpha[-1])

        pz = softmax(log_alpha + log_beta, axis=-1)

        if not (return_Nzz or return_log_px):
            return pz

        r = [pz]
        if return_Nzz:
            x_e = np.expand_dims(x, axis=1)
            log_alpha_e = np.expand_dims(log_alpha, axis=-1)
            log_beta_e = np.expand_dims(log_beta, axis=1)
            log_trans_e = np.expand_dims(self.log_trans, axis=0)
            log_zz = log_alpha_e[:-1] + x_e[1:] + log_trans_e + log_beta_e[1:]
            log_zz_norm = logsumexp(log_zz.reshape((log_zz.shape[0], -1)), axis=-1)
            zz = np.exp(log_zz - log_zz_norm[:, None, None])
            Nzz = np.sum(zz, axis=0)
            r.append(Nzz)

        if return_log_px:
            r.append(log_px)

        return tuple(r)

    def elbo(
        self,
        x: np.ndarray,
        pz: Optional[np.ndarray] = None,
        Nzz: Optional[np.ndarray] = None,
    ) -> float:
        """Computes the ELBO for a sequence.

        Args:
            x: Per-frame log-likelihoods.
            pz: Optional posterior over states.
            Nzz: Optional expected transition counts.

        Returns:
            Evidence lower bound for ``x``.
        """
        self._require_initialized("elbo")
        if pz is None and Nzz is None:
            pz, Nzz = self.compute_pz(x, return_Nzz=True)
        elif pz is None or Nzz is None:
            raise ValueError("pz and Nzz must be provided together")

        Nz = pz[0]
        elbo = np.sum(Nz * self.log_pi) + np.sum(Nzz * self.log_trans) + np.sum(pz * x)
        return elbo

    def Estep(
        self, x: np.ndarray, stats_0: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Expectation step for a single sequence.

        Args:
            x: Per-frame log-likelihoods.
            stats_0: Optional running totals ``(Nz, Nzz)``.

        Returns:
            Posterior ``pz`` and updated statistics ``(Nz, Nzz)``.
        """
        self._require_initialized("Estep")
        if stats_0 is None:
            Nz = np.zeros((self.num_states,), dtype=float_cpu())
            Nzz = np.zeros((self.num_states, self.num_states), dtype=float_cpu())
        else:
            Nz, Nzz = stats_0

        pz, cur_Nzz = self.compute_pz(x, return_Nzz=True)
        Nz += pz[0]
        Nzz += cur_Nzz
        stats = (Nz, Nzz)

        return pz, stats

    def Mstep(self, stats: Tuple[np.ndarray, np.ndarray]) -> None:
        """Maximization step that updates model parameters.

        Args:
            stats: Tuple ``(Nz, Nzz)`` from :meth:`Estep`.
        """
        Nz, Nzz = stats

        if self.update_pi:
            self.pi = Nz / np.sum(Nz)

        if self.update_trans:
            prev_trans = self.trans.copy()
            row_sums = np.sum(Nzz, axis=-1, keepdims=True)
            self.trans = np.divide(
                Nzz,
                row_sums,
                out=np.zeros_like(Nzz, dtype=float_cpu()),
                where=row_sums > 0,
            )
            zero_rows = np.where(row_sums[:, 0] <= 0)[0]
            if zero_rows.size > 0:
                self.trans[zero_rows] = prev_trans[zero_rows]

            if self.tied_trans:
                p_loop = np.mean(np.diag(self.trans))
                if self.num_states == 1:
                    self.trans[:] = 1.0
                else:
                    p_off = (1 - p_loop) / (self.num_states - 1)
                    self.trans[:] = p_off
                    self.trans[np.diag_indices(self.num_states)] = p_loop

            if self.trans_mask is not None:
                self.trans *= self.trans_mask
                row_sums = np.sum(self.trans, axis=-1, keepdims=True)
                zero_rows = np.where(row_sums[:, 0] <= 0)[0]
                if zero_rows.size > 0:
                    prev_trans_m = prev_trans * self.trans_mask
                    prev_row_sums = np.sum(prev_trans_m, axis=-1, keepdims=True)
                    if np.any(prev_row_sums[zero_rows] <= 0):
                        raise ValueError(
                            "trans_mask creates states with no outgoing transitions"
                        )
                    prev_trans_m = np.divide(
                        prev_trans_m,
                        prev_row_sums,
                        out=np.zeros_like(prev_trans_m, dtype=float_cpu()),
                        where=prev_row_sums > 0,
                    )
                    self.trans[zero_rows] = prev_trans_m[zero_rows]
                    row_sums = np.sum(self.trans, axis=-1, keepdims=True)
                self.trans /= row_sums

        self.reset_aux()

    def log_predictive(self, x: np.ndarray) -> np.ndarray:
        """Computes ``log p(x_{t+1} | x_{1:t})`` for each time step.

        Args:
            x: Per-frame log-likelihoods.

        Returns:
            Array of predictive log-likelihoods across time.
        """
        # log p(x_{N+1}|x_1,..,x_N}
        if not self.is_init:
            raise ValueError("HMM parameters must be initialized before log_predictive")
        self._validate_seq_llk(x)

        log_alpha = self.forward(x)[:-1]
        log_px = logsumexp(log_alpha, axis=-1)

        log_alpha_e = np.expand_dims(log_alpha, axis=-1)
        log_trans_e = np.expand_dims(self.log_trans, axis=0)

        log_pred = logsumexp(log_alpha_e + log_trans_e, axis=1)
        log_pred = logsumexp(log_pred + x[1:], axis=-1) - log_px

        return log_pred

    def viterbi_decode(
        self, x: np.ndarray, nbest: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the most likely latent state sequences.

        Args:
            x: Per-frame log-likelihoods.
            nbest: Number of best paths to return.

        Returns:
            Tuple ``(paths, log_probs)`` with shapes
            ``(nbest_eff, T)`` and ``(nbest_eff,)`` respectively, where
            ``nbest_eff <= nbest`` if fewer distinct paths are available.
        """
        if not self.is_init:
            raise ValueError("HMM parameters must be initialized before viterbi_decode")
        if nbest < 1:
            raise ValueError(f"nbest must be >= 1, got {nbest}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2D with shape (T, num_states), got {x.shape}")
        if x.shape[1] != self.num_states:
            raise ValueError(
                f"x.shape[1] ({x.shape[1]}) != num_states ({self.num_states})"
            )

        T = x.shape[0]
        if T == 0:
            raise ValueError("x must contain at least one frame")
        S = self.num_states
        neg_inf = -np.inf

        # scores[t, j, r] is the score of the r-th best path ending at state j at time t
        scores = np.full((T, S, nbest), neg_inf, dtype=float_cpu())
        prev_state = np.zeros((T, S, nbest), dtype=int)
        prev_rank = np.zeros((T, S, nbest), dtype=int)

        scores[0, :, 0] = self.log_pi + x[0]

        for t in range(1, T):
            prev_scores = scores[t - 1]
            for j in range(S):
                cand = prev_scores + self.log_trans[:, j][:, None] + x[t, j]
                cand_flat = cand.reshape(-1)
                valid = np.isfinite(cand_flat)
                num_valid = int(np.sum(valid))
                if num_valid == 0:
                    continue
                k = min(nbest, num_valid)
                valid_idx = np.nonzero(valid)[0]
                top_local = np.argpartition(cand_flat[valid_idx], -k)[-k:]
                top_idx = valid_idx[top_local]
                top_scores = cand_flat[top_idx]
                order = np.argsort(top_scores)[::-1]
                top_idx = top_idx[order]
                top_scores = top_scores[order]

                pred_state = top_idx // nbest
                pred_rank = top_idx % nbest

                scores[t, j, :k] = top_scores
                prev_state[t, j, :k] = pred_state
                prev_rank[t, j, :k] = pred_rank

        final_scores = scores[-1].reshape(-1)
        valid = np.isfinite(final_scores)
        num_valid = int(np.sum(valid))
        if num_valid == 0:
            raise ValueError("viterbi_decode found no valid paths")

        nbest_eff = min(nbest, num_valid)
        valid_idx = np.nonzero(valid)[0]
        top_local = np.argpartition(final_scores[valid_idx], -nbest_eff)[-nbest_eff:]
        best_idx = valid_idx[top_local]
        log_pxz = final_scores[best_idx]
        order = np.argsort(log_pxz)[::-1]
        best_idx = best_idx[order]
        log_pxz = log_pxz[order]

        best_state = best_idx // nbest
        best_rank = best_idx % nbest

        paths = np.zeros((nbest_eff, T), dtype=int)
        for n in range(nbest_eff):
            cur_state = best_state[n]
            cur_rank = best_rank[n]
            paths[n, -1] = cur_state
            for t in range(T - 1, 0, -1):
                p_state = prev_state[t, cur_state, cur_rank]
                p_rank = prev_rank[t, cur_state, cur_rank]
                paths[n, t - 1] = p_state
                cur_state, cur_rank = p_state, p_rank

        return paths, log_pxz

    def sample(
        self,
        num_seqs: int,
        num_steps: int,
        rng: Optional[Union[Generator, RandomState]] = None,
        seed: int = 1024,
    ) -> np.ndarray:
        """Draws sequences of state indicators.

        Args:
            num_seqs: Number of independent sequences.
            num_steps: Length of each sequence.
            rng: Optional random generator.
            seed: Seed used when ``rng`` is ``None``.

        Returns:
            One-hot encoded states with shape ``(num_seqs, num_steps, num_states)``.
        """
        if not self.is_init:
            raise ValueError("HMM parameters must be initialized before sample")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if rng is None:
            rng = np.random.default_rng(seed)

        x = np.zeros((num_seqs, num_steps, self.num_states), dtype=float_cpu())
        x[:, 0, :] = rng.multinomial(1, self.pi, size=(num_seqs,))
        for t in range(1, num_steps):
            for k in range(self.num_states):
                index = x[:, t - 1, k] == 1
                n_k = np.sum(index)
                if n_k == 0:
                    continue
                x[index, t, :] = rng.multinomial(1, self.trans[k], size=(n_k,))

        return x

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary.

        Returns:
            Dictionary of constructor arguments.
        """
        config = {
            "num_states": self.num_states,
            "update_pi": self.update_pi,
            "update_trans": self.update_trans,
            "tied_trans": self.tied_trans,
            "left_to_right": self.left_to_right,
        }
        base_config = super(HMM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: Any) -> None:
        """Persists model parameters into an HDF5 group.

        Args:
            f: File handle opened by :mod:`h5py`.
        """
        params = {"pi": self.pi, "trans": self.trans, "trans_mask": self.trans_mask}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "HMM":
        """Loads parameters from storage and instantiates an :class:`HMM`.

        Args:
            f: HDF5 file handle pointing to the saved parameters.
            config: Configuration dictionary generated by :meth:`get_config`.

        Returns:
            Fully initialized HMM instance.
        """
        param_list = ["pi", "trans", "trans_mask"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            pi=params["pi"],
            trans=params["trans"],
            trans_mask=params["trans_mask"],
            **config,
        )
