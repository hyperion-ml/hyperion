"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np
import h5py

import scipy.linalg as la
from scipy.special import erfinv

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel


class Gaussianizer(HypModel):
    """Class to make i-vector distribution standard Normal."""

    def __init__(self, max_vectors=None, r=None, **kwargs):
        super(Gaussianizer, self).__init__(**kwargs)
        self.max_vectors = max_vectors
        self.r = r

    def predict(self, x):
        px_cum = np.linspace(0, 1, self.r.shape[0] + 2)[1:-1]
        y_map = erfinv(2 * px_cum - 1) * np.sqrt(2)

        r = self.r[1:]
        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            y_index = np.searchsorted(r[:, i], x[:, i])
            logging.debug(y_index)
            y[:, i] = y_map[y_index]

        return y

    def fit(self, x):

        r = np.sort(x, axis=0, kind="heapsort")

        x = np.zeros((1, x.shape[-1]), dtype=float_cpu())

        if r.shape[0] > self.max_vectors:
            index = np.round(
                np.linspace(0, r.shape[0] - 1, self.max_vectors, dtype=float)
            ).astype(int)
            r = r[index, :]

        self.r = np.vstack((x, r))

    def get_config(self):
        config = {"max_vectors": self.max_vectors}

        base_config = super(Gaussianizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"r": self.r}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["r"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            r=params["r"], max_vectors=config["max_vectors"], name=config["name"]
        )

    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, "r") as f:
            r = np.asarray(f["r"], dtype="float32")
            return cls(r=r)

    def save_mat(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("r", data=self.r)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("max_vectors", "name")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "max-vectors",
            default=None,
            type=int,
            help=("maximum number of background vectors"),
        )

        parser.add_argument(p1 + "name", default="gauss")

    add_arparse_args = add_class_args
