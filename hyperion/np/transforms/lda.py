"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..np_model import NPModel
from .sb_sw import SbSw


class LDA(NPModel):
    """Class to do linear discriminant analysis.

    Attributes:
      mu: data mean vector
      T: LDA projection.
      lda_dim: LDA dimension.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.
    """

    def __init__(
        self, mu=None, T=None, lda_dim=None, update_mu=True, update_T=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        if T is None:
            self.lda_dim = lda_dim
        else:
            self.lda_dim = T.shape[1]
        self.update_mu = update_mu
        self.update_T = update_T

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
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    def fit(self, x, y, mu=None, Sb=None, Sw=None):
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          y: training labels as integers in [0, num_classes-1] with shape (num_samples,)
          mu: precomputed mean.
          Sb: precomputed between-class covariance.
          Sw: precomputed within-class covariance.
        """
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
        """Returns the model configuration dict."""
        config = {
            "lda_dim": self.lda_dim,
            "update_mu": self.update_mu,
            "update_t": self.update_T,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "T": self.T}
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
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], T=params["T"], **config)

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
