"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from scipy import linalg as sla

from ...hyp_defs import float_cpu
from ...utils.math import (
    invert_pdmat,
    invert_trimat,
    logdet_pdmat,
    vec2symmat,
    symmat2vec,
)
from ..core.pdf import PDF


class JFATotal(PDF):
    def __init__(self, K, y_dim=None, T=None, **kwargs):
        super(JFATotal, self).__init__(**kwargs)
        if T is not None:
            y_dim = T.shape[0]

        self.K = K
        self.y_dim = y_dim
        self.T = T

        # aux
        self._TT = None
        self.__upptr = None

    def reset_aux(self):
        self._TT = None

    @property
    def is_init():
        if self._is_init:
            return True
        if self.T is not None:
            self._is_init = True
        return self._is_init

    def initialize(self, N, F):
        assert N.shape[0] == self.K

        self.T = np.random.randn(self.y_dim, F.shape[1]).astype(float_cpu(), copy=False)

    def compute_py_g_x(
        self, N, F, G=None, return_cov=False, return_elbo=False, return_acc=False
    ):
        assert self.is_init
        x_dim = int(F.shape[1] / self.K)
        M = F.shape[0]
        y_dim = self.y_dim

        compute_inv = return_cov or return_acc
        return_tuple = compute_inv or return_elbo

        TF = np.dot(F, self.T.T)
        L = self.compute_L(self.TT, N, self._upptr)
        y = np.zeros((M, y_dim), dtype=float_cpu())

        if return_cov:
            Sy = np.zeros((M, y_dim * (y_dim + 1) / 2), dtype=float_cpu())
        else:
            Sy = None

        if return_elbo:
            elbo = np.zeros((M,), dtype=float_cpu())

        if return_acc:
            Py = np.zeros((y_dim, y_dim), dtype=float_cpu())
            Ry = np.zeros((self.K, y_dim * (y_dim + 1) / 2), dtype=float_cpu())

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
                Ry += iL[self.__uppr] * N[i][:, None]

        if not return_tuple:
            return y

        r = [y]

        if return_cov:
            r += [Sy]

        if return_elbo:
            if G is not None:
                elbo += G
            elbo += 0.5 * np.sum(VF * y, axis=-1)
            r += [elbo]

        if return_acc:
            r += [Ry, Py]

        return tuple(r)

    def Estep(self, N, F, G=None):

        y, elbo, Ry, Py = self.compute_py_g_x(
            N, F, G, return_elbo=True, return_acc=True
        )

        M = y.shape[0]
        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F, y)

        elbo = np.sum(elbo)

        stats = (elbo, M, y_acc, Ry, Cy, Py)
        return stats

    def MstepML(self, stats):
        _, M, y_acc, Ry, Cy, _ = stats
        T = np.zeros_like(self.T)
        Ryk = np.zeros((self.y_dim, self.y_dim), dtype=float_cpu())
        x_dim = T.shape[1] / self.K
        for k in range(self.K):
            idx = k * x_dim
            Ryk[self._upptr] = Ry[k]
            iRyk_mult = invert_pdmat(Ryk, right_inv=False)[0]
            T[:, idx : idx + x_dim] = iRyk_mult(Cy[idx : idx + x_dim].T)

        self.T = T
        self.reset_aux()

    def MstepMD(self, stats):
        _, M, y_acc, Ry, Cy, Py = stats
        mu_y = y_acc / M
        Cy = Py / M - np.outer(my_y, mu_y)
        chol_Cy = la.cholesky(Cy, lower=False, overwrite_a=True)
        self.T = np.dot(chol_Cy, self.T)

        self.reset_aux()

    def fit(
        self,
        N,
        F,
        G=None,
        N_val=None,
        F_val=None,
        epochs=20,
        ml_md="ml+md",
        md_epochs=None,
    ):

        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        if not self.is_init:
            self.initialize(N, F)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):

            stats = self.Estep(N, F, G)
            elbo[epoch] = stats[0]
            if N_val is not None and F_val is not None:
                _, elbo_val_e = self.compute_py_x(N, F, G, return_elbo=True)
                elbo_val[epoch] = np.sum(elbo_val_e)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

        elbo_norm = elbo / np.sum(N)
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(N_val)
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    @property
    def TT(self):
        if self._TT is None:
            self._TT = self.compute_TT(self.T, self.K)
        return self._TT

    @property
    def _upptr(self):
        if self.__upptr is None:
            I = np.eye(self.y_dim, dtype=float_cpu())
            self.__upptr = np.triu(I).ravel()
        return self.__upptr

    @staticmethod
    def compute_TT(self, T, K, upptr):
        x_dim = int(T.shape[1] / K)
        y_dim = T.shape[0]
        TT = np.zeros((K, y_dim * (y_dim + 1) / 2), dtype=float_cpu())
        for k in range(K):
            idx = k * x_dim
            T_k = T[:, idx : idx + x_dim]
            TT_k = np.dot(T_k, T_k.T)
            TT[k] = TT_k[self._upptr]

        return TT

    @staticmethod
    def compute_L(TT, N, upptr):
        y_dim = self._upptr.shape[0]
        I = np.eye(y_dim, dtype=float_cpu())[self._upptr]
        return I + np.dot(N, TT)

    @staticmethod
    def normalize_T(T, chol_prec):
        Tnorm = np.zeros_like(T)
        K = chol_prec.shape[0]
        x_dim = int(T.shape[1] / K)
        for k in range(K):
            idx = k * x_dim
            Tnorm[:, idx : idx + x_dim] = np.dot(
                T[:, idx : idx + x_dim], chol_prec[k].T
            )

        return Tnorm

    def get_config(self):
        config = {"K": self.K}
        base_config = super(JFATotal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def sample(self, num_samples):
        pass
