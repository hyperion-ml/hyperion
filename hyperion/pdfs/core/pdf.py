"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from abc import ABCMeta, abstractmethod
from ...hyp_model import HypModel


class PDF(HypModel):
    __metaclass__ = ABCMeta

    def __init__(self, x_dim=1, **kwargs):
        super(PDF, self).__init__(**kwargs)
        self.x_dim = x_dim

    def get_config(self):
        config = {"x_dim": self.x_dim}
        base_config = super(PDF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @abstractmethod
    def log_prob(self, x):
        pass

    def eval_llk(self, x):
        return self.log_prob(x)

    @abstractmethod
    def sample(self, num_samples):
        pass

    def generate(self, num_samples, **kwargs):
        return self.generate(num_samples, **kwargs)
