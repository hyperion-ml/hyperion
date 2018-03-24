from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from abc import ABCMeta, abstractmethod
from ...hyp_model import HypModel

class PDF(HypModel):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(PDF, self).__init__(**kwargs)


    # def get_config(self):
    #     config = {'x_dim': self.x_dim }
    #     base_config = super(PDF, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    
    @abstractmethod
    def log_prob(self, x):
        pass


    def log_cdf(self, x):
        raise NotImplementedError

    
    @abstractmethod
    def sample(self, num_samples):
        pass

