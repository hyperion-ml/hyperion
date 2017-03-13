""""
Tied Variational Autoencoder with 2 latent variables:
 - y: variable tied across all the samples in the segment
 - z: untied variable,  it has a different value for each frame

Factorization of the posterior:
   q(y_i,Z_i)=q(y_i) \prod_j q(z_{ij} | y_i)

The parameters \phi of all the  variational distributions are given by 
two NN.
"""

from __future__ import absolute_import
from __future__ import print_function

from six.moves import xrange

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Merge
from keras.models import Model

from ..layers.sampling import *
from .. import objectives as hyp_obj

from .tied_cvae_qyqz import TiedCVAE_qYqZ

class TiedCVAE_qYqZgY(TiedCVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net, x_distribution):
        super(TiedCVAE_qYqZgY,self).__init__([], decoder_net, x_distribution)
        self.qy_net = qy_net
        self.qz_net = qz_net

    def build(self, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.r_dim = self.qy_net.internal_input_shapes[1][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        assert(self.r_dim==self.decoder_net.internal_input_shapes[2][-1])
        self._build_model()
        self._build_loss()

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        r = Input(shape=(self.max_seq_length, self.r_dim,))
        y_param = self.qy_net([x, r])
        y = DiagNormalSamplerFromSeqLevel(self.max_seq_length)(y_param)
        z_param = self.qz_net([x, y, r])
        self.encoder_net = Model([x, r], y_param+z_param)
        super(TiedCVAE_qYqZgY, self)._build_model()
        
