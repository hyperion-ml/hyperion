"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel


class MVN(HypModel):
    """Class to do global mean and variance normalization."""

    def __init__(self, mu=None, s=None, **kwargs):
        super(MVN, self).__init__(**kwargs)
        self.mu = mu
        self.s = s

    def predict(self, x):
        if self.mu is not None:
            x = x - self.mu
        if self.s is not None:
            x = x / self.s
        return x

    def fit(self, x):
        self.mu = np.mean(x, axis=0)
        self.s = np.std(x, axis=0)

    def save_params(self, f):
        params = {"mu": self.mu, "s": self.s}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "s"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], s=params["s"], name=config["name"])
