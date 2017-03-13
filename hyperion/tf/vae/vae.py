"""
Variational Autoencoder as is defined in 

Diederik P Kingma, Max Welling, Auto-Encoding Variational Bayes
https://arxiv.org/abs/1312.6114
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import tensorflow as tf

class VAE(PDF):

    def __init__(self, qz_net, px_net, x_distribution):
        self.qz_net = qz_net
        self.px_net = px_net
        self.x_distribution = x_distribution
        self.x_dim = 0
        self.z_dim = 0
        self.z_param = None
        self.model = None

    def build(self):
        self.x_dim = 
        self.z_dim = 
        self._build_model()
        self._build_loss()

        
