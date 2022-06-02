"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..np_model import NPModel


class MVN(NPModel):
    """Class to do global mean and variance normalization.

    Attributes:
      mu: data mean vector
      s: standard deviation vector.

    """

    def __init__(self, mu=None, s=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.s = s

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
        if self.s is not None:
            x = x / self.s
        return x

    def fit(self, x):
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
        """
        self.mu = np.mean(x, axis=0)
        self.s = np.std(x, axis=0)

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "s": self.s}
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
        param_list = ["mu", "s"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(mu=params["mu"], s=params["s"], name=config["name"])
