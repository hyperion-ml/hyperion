"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel


class NAP(HypModel):
    """Class to do nussance attribute projection."""

    def __init__(self, U=None, **kwargs):
        super(NAP, self).__init__(**kwargs)
        self.U = U

    def predict(self, x):
        return x - np.dot(np.dot(x, self.U.T), self.U)

    def fit(self, x, U_dim, class_ids):
        x_hat = np.zeros_like(x)
        u_ids = np.uniqe(class_ids)
        M = np.sqrt(len(u_ids))
        for i in u_ids:
            idx = np.nonzero(i == class_ids)
            N = np.sqrt(len(idx))
            mu_i = np.mean(x[idx, :], axis=0)
            xx[idx, :] = (x[idx, :] - mu_i) / N
        xx /= M
        _, s, Vt = np.svd(xx, full_matrices=False, overwrite_a=True)
        idx = (np.argsort(s)[::-1])[:U_dim]
        self.U = Vt[idx, :]

    def save_params(self, f):
        params = {"U": self.U}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["U"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(U=params["U"], name=config["name"])

    # @classmethod
    # def load(cls, file_path):
    #     with h5py.File(file_path, 'r') as f:
    #         config = self.load_config_from_json(f['config'])
    #         param_list = ['U']
    #         params = self._load_params_to_dict(f, config['name'], param_list)
    #         return cls(U=params['U'], name=config['name'])

    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, "r") as f:
            U = np.asarray(f["U"], dtype="float32")
            return cls(U)

    def save_mat(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("U", data=self.U)
