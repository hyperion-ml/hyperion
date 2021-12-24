"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import numpy as np
import h5py

from .cent_whiten_up import CentWhitenUP


class LNormUP(CentWhitenUP):
    """Class to do Lenght Normalization with uncertainty propagation"""

    def predict(self, x):
        x = super(LNormUP, self).predict(x)
        x_dim = int(x.shape[-1] / 2)
        m_x = x[:, :x_dim]
        s2_x = x[:, x_dim:]

        mx2 = np.sum(m_x ** 2, axis=1, keepdims=True) + 1e-10
        m_x /= np.sqrt(mx2)
        s2_x /= mx2

        return np.hstack((m_x, s2_x))
