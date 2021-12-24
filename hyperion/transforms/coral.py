"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel


class CORAL(HypModel):
    """Class to do CORAL"""

    def __init__(
        self,
        mu=None,
        T_col=None,
        T_white=None,
        update_mu=True,
        update_T=True,
        alpha_mu=1,
        alpha_T=1,
        **kwargs
    ):
        super(CORAL, self).__init__(**kwargs)
        self.mu = mu
        self.T_col = T_col
        self.T_white = T_white
        self.T = None
        self.update_mu = update_mu
        self.update_T = update_T
        self.alpha_mu = alpha_mu
        self.alpha_T = alpha_T

    def get_config(self):
        config = {
            "update_mu": self.update_mu,
            "update_t": self.update_T,
            "pca_dim": self.pca_dim,
        }
        base_config = super(CORAL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_T(self):
        if self.T_col is not None and self.T_white is not None:
            self.T = np.dot(self.T_white, self.T_col)

    def predict(self, x):
        if self.T is None:
            self._compute_T()
        if self.mu is not None:
            x = x - self.mu

        if self.T is not None:
            x = np.dot(x, self.T)

        return x

    def fit(self, x, sample_weight=None, x_out=None, sample_weight_out=None):

        if x_out is None:
            assert self.T_white is not None
        else:
            mu_out = np.mean(x_out, axis=0)
            if self.update_T:
                delta = x_out - mu_out
                S_out = np.dot(delta.T, delta) / x_out.shape[0]
                # zero-phase component analysis (ZCA)
                d, V = la.eigh(S_out)
                self.T_white = np.dot(V * (1 / np.sqrt(d)), V.T)

        mu_in = np.mean(x, axis=0)
        if self.update_T:
            delta = x - mu_in
            S_in = np.dot(delta.T, delta) / x.shape[0]
            if self.alpha_T < 1:
                S_in = self.alpha_T * S_in + (1 - self.alpha_T) * S_out
            # zero-phase component analysis (ZCA)
            d, V = la.eigh(S_in)
            d[d < 0] = 0
            self.T_col = np.dot(V * np.sqrt(d), V.T)

        if self.update_mu:
            self.mu = self.alpha_mu * (mu_out - mu_in)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "T_col", "T_white"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            T_col=params["T_col"],
            T_white=params["T_white"],
            name=config["name"],
        )

    def save_params(self, f):
        params = {
            "mu": self.mu,
            "T_col": self.T_col,
            "T_white": self.T_white,
            "alpha_mu": self.alpha_mu,
            "alpha_T": self.alpha_T,
        }
        self._save_params_from_dict(f, params)
