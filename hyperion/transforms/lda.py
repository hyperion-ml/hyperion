"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel
from .sb_sw import SbSw


class LDA(HypModel):
    """Class to do linear discriminant analysis."""

    def __init__(
        self, mu=None, T=None, lda_dim=None, update_mu=True, update_T=True, **kwargs
    ):
        super(LDA, self).__init__(**kwargs)
        self.mu = mu
        self.T = T
        if T is None:
            self.lda_dim = lda_dim
        else:
            self.lda_dim = T.shape[1]
        self.update_mu = update_mu
        self.update_T = update_T

    def predict(self, x):
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    def fit(self, x, y, mu=None, Sb=None, Sw=None):

        if mu is None or Sb is None or Sw is None:
            sbsw = SbSw()
            sbsw.fit(x, y)
            mu = sbsw.mu
            Sb = sbsw.Sb
            Sw = sbsw.Sw

        if self.update_mu:
            self.mu = mu

        if not self.update_T:
            return

        assert Sb.shape == Sw.shape

        try:
            d, V = la.eigh(Sb, Sw)
        except:
            alpha = 1e-2 * np.max(np.diag(Sw))
            d, V = la.eigh(Sb, alpha * np.eye(Sw.shape[0]) + Sw)
        V = np.fliplr(V)

        p = V[0, :] < 0
        V[:, p] *= -1

        if self.lda_dim is not None:
            assert self.lda_dim <= V.shape[1]
            V = V[:, : self.lda_dim]

        self.T = V

    def get_config(self):
        config = {
            "lda_dim": self.lda_dim,
            "update_mu": self.update_mu,
            "update_t": self.update_T,
        }
        base_config = super(LDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"mu": self.mu, "T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], T=params["T"], name=config["name"])

    # @classmethod
    # def load(cls, file_path):
    #     with h5py.File(file_path, 'r') as f:
    #         config = self.load_config_from_json(f['config'])
    #         param_list = ['mu', 'T']
    #         params = self._load_params_to_dict(f, config['name'], param_list)
    #         return cls(mu=params['mu'], T=params['T'], name=config['name'])

    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)
