"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..metrics import dcf

from .binary_logistic_regression import BynaryLogisticRegression as BLR

class GreedyFusionBinaryLR(HypModel):

    def __init__(self, weights=None, bias=None,
                 system_idx=None, system_names=None, max_systems=None,
                 prioritize_positive=True,
                 penalty='l2', lambda_reg=1e-6,
                 bias_scaling=1, prior=0.5, prior_eval=None,
                 solver='liblinear', max_iter=100,
                 dual=False, tol=0.0001, verbose=0, lr_seed=1024, **kwargs):

        super(GreedyFusionBinaryLR, self).__init__(**kwargs)

        self.weights = weights
        self.bias = bias
        self.system_idx = system_idx
        self.system_names = system_names
        self.max_systems = max_systems
        self.prioritize_positive = prioritize_positive
        
        self.lr = BLR(penalty=penalty, lambda_reg=lambda_reg,
                      use_bias=True, bias_scaling=bias_scaling,
                      prior=prior, solver=solver, max_iter=max_iter,
                      dual=dual, tol=tol, verbose=verbose, warm_start=False,
                      lr_seed=lr_seed)
        


    def get_fusion_params(self, idx):
        return self.weights[idx], self.bias[idx], self.system_idx[idx]

    
    def fit(x, class_ids, sample_weights=None):
        
        num_systems = x.shape[1]
        if self.max_systems = None:
            self.max_systems = min(10, num_systems)

        self.weights = []
        self.bias = []
        for i in xrange(self.max_systems):
            
            
        
