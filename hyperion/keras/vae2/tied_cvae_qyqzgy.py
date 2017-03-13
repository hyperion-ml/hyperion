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

from .. import objectives as hyp_obj
from ..layers.core import Repeat
from ..layers.sampling2 import *

from .tied_cvae_qyqz import TiedCVAE_qYqZ
from .tied_vae_qyqzgy import TiedVAE_qYqZgY

class TiedCVAE_qYqZgY(TiedCVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net, x_distribution):
        super(TiedCVAE_qYqZgY,self).__init__([], decoder_net, x_distribution)
        self.qy_net = qy_net
        self.qz_net = qz_net

    def build(self, nb_samples_y=1, nb_samples_z=1, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.r_dim = self.qy_net.internal_input_shapes[1][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        assert(self.r_dim==self.decoder_net.internal_input_shapes[2][-1])
        self.nb_samples_y = nb_samples_y
        self.nb_samples_z = nb_samples_z
        self.nb_samples = nb_samples_y*nb_samples_z
        self._build_model()
        self._build_loss()

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        r = Input(shape=(self.max_seq_length, self.r_dim,))
        
        self.y_param = self.qy_net([x, r])
        y = DiagNormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                          nb_samples=self.nb_samples_y,
                                          one_sample_per_seq=False)(self.y_param)

        if self.nb_samples_y > 1:
            x_rep = Repeat(self.nb_samples_y, axis=0)(x)
            r_rep = Repeat(self.nb_samples_y, axis=0)(r)
        else:
            x_rep = x
            r_rep = r

        self.z_param = self.qz_net([x_rep, y, r_rep])
        
        self.encoder_net = Model([x, r], self.y_param+self.z_param)

        z = DiagNormalSampler(nb_samples = self.nb_samples_z)(self.z_param)

        if self.nb_samples_z > 1:
            y_rep = Repeat(self.nb_samples_z, axis=0)(y)
            r_rep = Repeat(self.nb_samples, axis=0)(r)
        else:
            y_rep = y

        x_dec_param=self.decoder_net([y_rep, z, r_rep])
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model([x, r], x_dec_param)

        
    def _build_loss(self):
        if self.x_distribution == 'bernoulli':
            self.loss=lambda x, y : TiedVAE_qYqZgY._get_loss_bernoulli(
                x, y, self.y_param, self.z_param, self.nb_samples, self.nb_samples_y)
        else:
            self.loss=lambda x, y : TiedVAE_qYqZgY._get_loss_normal(
                x, y, self.y_param, self.z_param, self.nb_samples, self.nb_samples_y)

