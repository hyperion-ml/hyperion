"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py

from .cent_whiten import CentWhiten


class LNorm(CentWhiten):
    """Class to do length normalization.

    Attributes:
      mu: data mean vector
      T: whitening projection.
      update_mu: whether or not to update the mean when training.
      update_T: wheter or not to update T when training.
    """

    def predict(self, x):
        x = super().predict(x)
        mx = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)) + 1e-10
        return np.sqrt(x.shape[1]) * x / mx
