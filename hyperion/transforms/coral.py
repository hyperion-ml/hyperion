"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel

class CORAL(HypModel):
    """Class to do CORA
    """
    def __init__(self, T_col=None, T_white=None, **kwargs):
        super(CORAL, self).__init__(**kwargs)
        self.T_col = T_col
        self.T_white = T_white
        self.T = None


    def _compute_T(self):
        if self.T_col is not None and self.T_white is not None:
            self.T = np.dot(self.T_white, self.T_col)


    def predict(self, x):
        if self.T is None:
            self._compute_T()
        return np.dot(x, self.T)


    def fit(self, x, sample_weight=None, x_out=None, sample_weight_out=None):

        if x_out is None:
            assert self.T_white is not None
        else:
            mu = np.mean(x_out, axis=0)
            delta = x_out - mu
            S_out = np.dot(delta.T, delta)/x_out.shape[0]
            # zero-phase component analysis (ZCA)
            d, V = la.eigh(S_out)
            self.T_white = np.dot(np.dot(V,1/np.sqrt(d)), V.T)
            
            
        mu = np.mean(x, axis=0)
        delta = x - mu
        S_in = np.dot(delta.T, delta)/x.shape[0]
        # zero-phase component analysis (ZCA)
        d, V = la.eigh(S_in)
        self.T_col = np.dot(np.dot(V, np.sqrt(d)), V.T)

        
    @classmethod
    def load_params(cls, f, config):
        param_list = ['T_col', 'T_white']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        return cls(T_col=params['T_col'], T_white=params['T_white'], name=config['name'])
