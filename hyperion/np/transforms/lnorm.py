"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py

from .cent_whiten import CentWhiten


class LNorm(CentWhiten):
    """Class to do length normalization."""

    def predict(self, x):
        x = super().predict(x)
        mx = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)) + 1e-10
        return np.sqrt(x.shape[1]) * x / mx
