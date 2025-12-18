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

        if self.pi is not None and self.trans is not None:
            self.validate()
            self._is_init = True

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
        assert len(self.pi) == self.num_states
        assert self.trans.shape[0] == self.num_states
        assert self.trans.shape[1] == self.num_states
        if self.trans_mask is not None:
            assert self.trans_mask.shape == self.trans.shape

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

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for epoch in range(epochs):
            for i in range(x.shape[0]):
                pz, stats = self.Estep(x[i], stats)
                _, Nzz = stats
                elbo[epoch] += self.elbo(x[i], pz=pz, Nzz=Nzz)

            self.Mstep(stats)

            if x_val is not None:
                for i in range(x_val.shape[0]):
                    pz, stats = self.Estep(x_val[i])
                    _, Nzz = stats
                    elbo_val[epoch] += self.elbo(x[i], pz=pz, Nzz=Nzz)

        N_tot = np.sum([x_i.shape[0] for x_i in x])
        if x_val is None:
            return elbo, elbo / N_tot
        else:
            N_val_tot = np.sum([x_i.shape[0] for x_i in x_val])
            return elbo, elbo / N_tot, elbo_val, elbo_val / N_val_tot

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Runs the forward recursion.

        Args:
            x: Per-frame log-likelihoods with shape ``(T, num_states)``.

        Returns:
            Log forward messages ``alpha`` with the same shape.
        """
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
        N = x.shape[0]
        log_beta = np.zeros((N, self.num_states), dtype=float_cpu())
        log_beta[-1] = 1
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
            zz = log_alpha_e[:-1] + x_e[1:] + log_trans_e + log_beta_e[1:]
            zz = softmax(zz, axis=-1)
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
        if pz is None:
            pz, Nzz = self.compute_pz(x, return_Nzz=True)

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

        self.pi = Nz / np.sum(Nz)
        self.trans = Nzz / np.sum(Nzz, axis=-1, keepdims=True)

        if self.tied_trans:
            p_loop = np.mean(np.diag(self.trans))
            self.trans[:] = (1 - p_loop) / self.num_states
            self.trans[np.diag_indices(self.num_states)] = p_loop

        if self.trans_mask is not None:
            self.trans *= self.trans_mask
            self.trans /= np.sum(self.trans, axis=-1, keepdims=True)

        self.reset_aux()

    def log_predictive(self, x: np.ndarray) -> np.ndarray:
        """Computes ``log p(x_{t+1} | x_{1:t})`` for each time step.

        Args:
            x: Per-frame log-likelihoods.

        Returns:
            Array of predictive log-likelihoods across time.
        """
        # log p(x_{N+1}|x_1,..,x_N}
        assert self.is_init

        log_alpha = self.forward(x)[:-1]
        log_px = np.sum(log_alpha, axis=-1, keepdims=True)

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
            ``(nbest, T)`` and ``(nbest,)`` respectively.
        """
        assert self.is_init
        idx_aux = np.arange(self.num_states)
        phi = np.zeros((x.shape[0], self.num_states), dtype=int)
        # phi[n, j] stores the argmax predecessor for state j at time n
        w = self.log_pi + x[0]
        for i in range(1, x.shape[0]):
            u = w[:, None] + self.log_trans
            k_max = np.argmax(u, axis=0)
            w = x[i] + u[k_max, idx_aux]
            phi[i - 1] = k_max

        best = np.fliplr(np.argsort(w))[:nbest]
        log_pxz = w[best]
        paths = np.zeros((nbest, x.shape[0]), dtype=int)
        for n in range(nbest):
            k_max = best[n]
            paths[n, -1] = k_max
            for i in range(x.shape[0] - 2, -1, -1):
                k_max = phi[i, k_max]
                paths[n, i] = k_max

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
                x[index] = rng.multinomial(1, self.trans[k], size=(n_k,))

        return x

    def get_config(self) -> Dict[str, Any]:
        """Builds a serializable configuration dictionary.

        Returns:
            Dictionary of constructor arguments.
        """
        config = {
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
        params = {"pi": self.pi, "trans": self.trans}
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
        param_list = ["pi", "trans"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(pi=params["pi"], trans=params["trans"], **config)
