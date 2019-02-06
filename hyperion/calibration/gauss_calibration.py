"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from ..hyp_model import HypModel

class GaussCalibration(HypModel):
    """Class for supervised Gaussian calibration
    """
    def __init__(self, mu1=None, mu2=None, sigma2=None, prior=0.5, **kwargs):
        super(GaussCalibration, self).__init__(**kwargs)
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.prior = prior
        self.a = None
        self.b = None
        if self.is_init():
            self._compute_scale_bias()


    def is_init(self):
        return self.mu1 is not None and self.mu2 is not None and self.sigma2 is not None

            
    def _compute_scale_bias(self):
        self.a = (self.mu1 - self.mu2)/self.sigma2
        self.b = 0.5*(self.mu2**2 - self.mu1**2)/self.sigma2
        

        
    def fit(self, x, y, sample_weight=None):
        non = x[y==0]
        tar = x[y==1]
        if sample_weight is None:
            sw_tar = 1
            sw_non = 1
            sample_weight = 1
            self.prior = float(len(tar))/len(x)
        else:
            sw_non = sample_weight[y==0]
            sw_tar = sample_weight[y==1]
            self.prior = np.sum(sw_tar)/np.sum(sample_weight)

        self.mu1 = np.mean(sw_tar*tar)/np.mean(sw_tar)
        self.mu2 = np.mean(sw_non*non)/np.mean(sw_non)

        self.sigma2 = (np.sum(sw_tar*(tar - self.mu1)**2) + np.sum(sw_non*(non - self.mu2)**2))/len(x)/np.mean(sample_weight)
        
        self._compute_scale_bias()


        
    def predict(self, x):
        return self.a*x+self.b


    def save_params(self, f):
        params = {'mu1': self.mu1,
                  'mu2': self.mu2,
                  'sigma2': self.sigma2}
        self._save_params_from_dict(f, params)

        

    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu1', 'mu2', 'sigma2']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        return cls(mu1=params['mu1'], mu2=params['mu2'], sigma2=config['sigma2'], name=config['name'])

    
