"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ....hyp_defs import float_cpu
from ....utils.math import softmax, logsumexp
from ..core import PDF


class HMM(PDF):
    def __init__(
        self,
        num_states=1,
        pi=None,
        trans=None,
        trans_mask=None,
        update_pi=True,
        update_trans=True,
        tied_trans=False,
        left_to_right=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if pi is not None:
            num_states = len(pi)

        self.num_states = num_states
        self.pi = pi
        self.trans = trans
        self.trans_mask = trans_mask

        self.update_pi = update_pi
        self.update_trans = update_trans
        self.tied_trans = tied_trans
        self.left_to_right = left_to_right

        if left_to_right and (trans_mask is None):
            self.trans_mask = np.triu(np.ones_like(self.trans))

        self._log_pi = None
        self._log_trans = None

    def reset_aux(self):
        self._log_pi = None
        self._log_trans = None

    @property
    def is_init(self):
        if self._is_init:
            return True

        if self.pi is not None and self.trans is not None:
            self.validate()
            self._is_init = True

        return self._is_init

    @property
    def log_pi(self):
        if self._log_pi is None:
            self._log_pi = np.log(self.pi + 1e-15)
        return self._log_pi

    @property
    def log_trans(self):
        if self._log_trans is None:
            self._log_trans = np.log(self.trans + 1e-15)
        return self._log_trans

    def validate(self):
        assert len(self.pi) == self.num_states
        assert self.trans.shape[0] == self.num_states
        assert self.trans.shape[1] == self.num_states
        if self.trans_mask is not None:
            assert self.trans_mask.shape == self.trans.shape

    def fit(self, x, sample_weight=None, x_val=None, sample_weight_val=None, epochs=10):

        N_val_tot = 0
        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        stats = None
        for epoch in range(epochs):
            for i in range(x.shape[0]):
                stats = self.Estep(x[i], stats)
                pz, Nzz = stats
                elbo[epoch] += self.elbo(x[i], pz=pz, Nzz=Nzz)

            self.Mstep(stats)

            if x_val is not None:
                for i in range(x_val.shape[0]):
                    pz, Nzz = self.Estep(x_val[i])
                    elbo_val[epoch] += self.elbo(x[i], pz=pz, Nzz=Nzz)

        N_tot = np.sum([x_i.shape[0] for x_i in x])
        if x_val is None:
            return elbo, elbo / N_tot
        else:
            N_val_tot = np.sum([x_i.shape[0] for x_i in x_val])
            return elbo, elbo / N_tot, elbo_val, elbo_val / N_val_tot

    def forward(self, x):
        # x = log P(x|z)
        N = x.shape[0]
        log_alpha = np.zeros((N, self.num_states), dtype=float_cpu())
        log_alpha[0] = self.log_pi + x[0]
        for n in range(1, N):
            log_alpha[n] = x[n] + logsumexp(
                log_alpha[n - 1][:, None] + self.log_trans, axis=0
            )

        return log_alpha

    def backward(self, x):

        N = x.shape[0]
        log_beta = np.zeros((N, self.num_states), dtype=float_cpu())
        log_beta[-1] = 1
        for n in range(N - 2, -1, -1):
            r = log_beta[n + 1] + x[n + 1] + self.log_trans
            log_beta[n] = logsumexp(r.T, axis=0)

        return log_beta

    def compute_pz(self, x, return_Nzz=False, return_log_px=False):
        log_alpha = self.forward(x)
        log_beta = self.backward(x)
        log_px = np.sum(log_alpha[-1])

        pz = softmax(log_alpha + log_beta, axis=-1)

        if not (return_Nzz or return_log_px):
            return pz

        r = [pz]
        if return_Nzz:
            x_e = np.expand_dims(axis=1)
            log_alpha_e = np.expand_dims(axis=-1)
            log_beta_e = np.expand_dims(axis=1)
            log_trans_e = np.expand_dims(axis=0)
            zz = log_alpha_e + x_e + log_trans_e + log_beta_e
            zz = softmax(zz, axis=-1)
            Nzz = np.sum(zz, axis=0)
            r.append(Nzz)

        if return_log_px:
            r.append(log_px)

        return tuple(r)

    def elbo(self, x, pz=None, Nzz=None):
        if pz is None:
            pz, Nzz = self.compute_pz(x, return_Nzz=True)

        Nz = pz[0]
        elbo = np.sum(Nz * self.log_pi) + np.sum(Nzz * self.log_trans) + np.sum(pz * x)
        return elbo

    def Estep(self, x, stats_0=None):

        if stats_0 is None:
            Nz = np.zeros((self.num_states,), dtype=float_cpu())
            Nzz = np.zeros((self.num_states, self.num_states), dtype=float_cpu())
        else:
            Nz, Nzz = stats_0

        pz, Nzz = self.compute_pz(x, return_Nzz=True)
        Nz += pz[0]
        Nzz += Nzz
        stats = (Nz, Nzz)

        return pz, stats

    def Mstep(self, stats):
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

    def log_predictive(self, x):
        # log p(x_{N+1}|x_1,..,x_N}
        assert self.is_init

        log_alpha = self.forward(x)[:-1]
        log_px = np.sum(log_alpha, axis=-1, keepdims=True)

        log_alpha_e = np.expand_dims(log_alpha, axis=-1)
        log_trans_e = np.expand_dims(self.log_trans, axis=0)

        log_pred = logsumexp(log_alpha_e + log_trans_e, axis=1)
        log_pred = logsumexp(log_pred + x[1:], axis=-1) - log_px

        return log_pred

    def viterbi_decode(self, x, nbest=1):
        assert self.is_init
        idx_aux = np.arange(self.num_states)
        phi = np.zeros((x.shape[0], self.num_states), dtype=int)
        w = self.log_pi + x[0]
        for i in range(x.shape[0]):
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

    def sample(self, num_seqs, num_steps, rng=None, seed=1024):
        if rng is None:
            rng = np.random.RandomState(seed)

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

    def get_config(self):
        config = {
            "update_pi": self.update_pi,
            "update_trans": self.update_trans,
            "tied_trans": self.tied_trans,
            "left_to_right": self.left_to_right,
        }
        base_config = super(HMM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"pi": self.pi, "trans": self.trans}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["pi", "trans"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(pi=params["pi"], trans=params["trans"], **config)
