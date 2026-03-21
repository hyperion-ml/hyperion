"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import h5py
import numpy as np
import scipy.linalg as la
from typing import Any, Optional

from ..hyper_np_model import HyperNPModel
from .cent_whiten import CentWhiten


class CentWhitenUP(CentWhiten):
    """Class to do centering and whitening with uncertainty propagation.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        update_mu: bool = True,
        update_T: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(mu, T, update_mu, update_T, **kwargs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "x.shape[-1] must be even (concatenated mean and variance), "
                f"got {x.shape[-1]}"
            )
        if self.T is None:
            raise ValueError("self.T must be initialized before calling predict")
        if self.T.ndim != 2:
            raise ValueError(f"self.T must be 2D, got ndim={self.T.ndim}")

        x_dim = int(x.shape[-1] / 2)
        if self.T.shape[0] != x_dim or self.T.shape[1] != x_dim:
            raise ValueError(
                "self.T must be square with shape (x_dim, x_dim) for CentWhitenUP, "
                f"got self.T.shape={self.T.shape}, x_dim={x_dim}"
            )
        m_x = x[:, :x_dim]
        s2_x = x[:, x_dim:]
        m_x = super().predict(m_x)
        for i in range(x.shape[0]):
            s2_x[i] = np.diag(np.dot(self.T.T * s2_x[i], self.T))
        return np.hstack((m_x, s2_x))

    def fit(self, x: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """Trains the transformation parameters.

        Args:
          x: training samples with shape (num_samples, x_dim)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "x.shape[-1] must be even (concatenated mean and variance), "
                f"got {x.shape[-1]}"
            )
        x = x[:, : int(x.shape[-1] / 2)]
        super().fit(x, sample_weight=sample_weight)
