"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..np_model import NPModel


class NAP(NPModel):
    """Class to do nuissance attribute projection.

    Attributes:
      U: NAP projection.
    """

    def __init__(self, U=None, U_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.U = U
        if U is None:
            self.U_dim = U_dim
        else:
            self.U_dim = U.shape[0]

    def __call__(self, x):
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def forward(self, x):
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def predict(self, x):
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return x - np.dot(np.dot(x, self.U.T), self.U)

    def fit(self, x, y):
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          y: training labels as integers in [0, num_classes-1] with shape (num_samples,)
        """
        u_ids = np.unique(y)
        M = np.sqrt(len(u_ids))
        for i in u_ids:
            idx = y == i
            N = np.sqrt(len(idx))
            mu_i = np.mean(x[idx, :], axis=0)
            xx[idx, :] = (x[idx, :] - mu_i) / N
        xx /= M
        _, s, Vt = np.svd(xx, full_matrices=False, overwrite_a=True)
        idx = (np.argsort(s)[::-1])[: self.U_dim]
        self.U = Vt[idx, :]

    def get_config(self):
        """Returns the model configuration dict."""
        config = {
            "U_dim": self.U_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"U": self.U}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["U"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(U=params["U"], name=config["name"])

    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, "r") as f:
            U = np.asarray(f["U"], dtype="float32")
            return cls(U)

    def save_mat(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("U", data=self.U)
