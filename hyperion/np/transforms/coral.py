"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..np_model import NPModel


class CORAL(NPModel):
    """Class to do CORAL.

    https://arxiv.org/abs/1612.01939

    Attributes:
      mu: mean shift between both domains.
      T_col: recoloring projection.
      T_white: whitening projection.
      update_mu: whether or not to update mu when training.
      update_T: wheter or not to update T_col and T_white when training.
      alpha_mu: weight of the in-domain data when computing in-domain mean.
      alpha_T: weight of the in-domain data when computing in-domain covariance.
    """

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
        super().__init__(**kwargs)
        self.mu = mu
        self.T_col = T_col
        self.T_white = T_white
        self.T = None
        self.update_mu = update_mu
        self.update_T = update_T
        self.alpha_mu = alpha_mu
        self.alpha_T = alpha_T

    def get_config(self):
        """Returns the model configuration dict."""
        config = {
            "update_mu": self.update_mu,
            "update_t": self.update_T,
            "alpha_mu": self.alpha_mu,
            "alpha_T": self.alpha_T,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_T(self):
        if self.T_col is not None and self.T_white is not None:
            self.T = np.dot(self.T_white, self.T_col)

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
        if self.T is None:
            self._compute_T()
        if self.mu is not None:
            x = x - self.mu

        if self.T is not None:
            x = np.dot(x, self.T)

        return x

    def fit(self, x, sample_weight=None, x_out=None, sample_weight_out=None):
        """Trains the model.

        Args:
          x:  in-domain data samples with shape (num_samples, x_dim).
          sample_weight: weight for each in-domain training sample.
          x_out:  out-domain data samples with shape (num_samples, x_dim).
          sample_weight_out: weight for each out-domain training sample.
        """
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
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "T_col", "T_white"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            T_col=params["T_col"],
            T_white=params["T_white"],
            **config,
        )

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {
            "mu": self.mu,
            "T_col": self.T_col,
            "T_white": self.T_white,
        }
        self._save_params_from_dict(f, params)
