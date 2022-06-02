"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

import scipy.linalg as la

from ..np_model import NPModel
from ..pdfs import Normal
from .cent_whiten import CentWhiten


class CentWhitenUP(CentWhiten):
    """Class to do centering and whitening with uncertainty propagation.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.
    """

    def __init__(self, mu=None, T=None, update_mu=True, update_T=True, **kwargs):
        super().__init__(mu, T, update_mu, update_T, **kwargs)

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
        x_dim = int(x.shape[-1] / 2)
        m_x = x[:, :x_dim]
        s2_x = x[:, x_dim:]
        m_x = super().predict(m_x)
        for i in range(x.shape[0]):
            s2_x[i] = np.diag(np.dot(self.T.T * s2_x[i], self.T))
        return np.hstack((m_x, s2_x))

    def fit(self, x, sample_weight=None):
        """Trains the transformation parameters.

        Args:
          x: training samples with shape (num_samples, x_dim)
        """
        x = x[:, : int(x.shape[-1] / 2)]
        super().fit(x, sample_weight=sample_weight)
