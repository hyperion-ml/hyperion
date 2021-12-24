"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel
from ..hyp_defs import float_cpu


class NDA(HypModel):
    """Class to do nearest-neighbors discriminant analysis"""

    def __init__(self, mu=None, T=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T

    def predict(self, x):
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    def fit(self, mu, Sb, Sw, nda_dim=None):
        self.mu = mu

        assert Sb.shape == Sw.shape

        d, V = la.eigh(Sb, Sw)
        V = np.fliplr(V)

        p = V[0, :] < 0
        V[:, p] *= -1

        if nda_dim is not None:
            assert nda_dim <= V.shape[1]
            V = V[:, :nda_dim]

        self.T = V

    def save_params(self, f):
        params = {"mu": self.mu, "T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], T=params["T"], name=config["name"])

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
