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
from __future__ import division

from six.moves import xrange

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Merge
from keras.models import Model

from ..layers.sampling import *
from .. import objectives as hyp_obj

from .tied_vae_qyqz import TiedVAE_qYqZ

class TiedVAE_qYqZgY(TiedVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net, x_distribution):
        super(TiedVAE_qYqZgY,self).__init__([], decoder_net, x_distribution)
        self.qy_net = qy_net
        self.qz_net = qz_net

    def build(self, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        self._build_model()
        self._build_loss()

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        self.y_param = self.qy_net(x)
        y = DiagNormalSamplerFromSeqLevel(self.max_seq_length, one_sample_per_seq=False)(self.y_param)
        self.z_param = self.qz_net([x, y])
        self.encoder_net = Model(x, self.y_param+self.z_param)
        z = DiagNormalSampler()(self.z_param)
        x_dec_param=self.decoder_net([y, z])
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model(x, x_dec_param)

        
class TiedVAE_qYqZgY2(TiedVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net, x_distribution):
        super(TiedVAE_qYqZgY2,self).__init__(None, decoder_net, x_distribution)
        self.qy_net = qy_net
        self.qz_net = qz_net
        self.model_y = None
        self.loss_y = None
        self.loss_mdy = None

    def build(self, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        self._build_model()
        self._build_loss()
        self._build_loss_y()

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        self.y_param = self.qy_net(x)
        y = DiagNormalSamplerFromSeqLevel(self.max_seq_length)(self.y_param)
        self.z_param = self.qz_net([x, y])
        self.encoder_net = Model(x, self.y_param+self.z_param)
        z = DiagNormalSampler()(self.z_param)
        x_dec_param=self.decoder_net([y, z])
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model(x, x_dec_param)

        z0 = Lambda(lambda x:0*x, output_shape=(self.max_seq_length, self.z_dim))(z)
        x_dec_param=self.decoder_net([y, z0])
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model_y=Model(x, x_dec_param)

    def _build_loss_y(self):
        if self.x_distribution == 'bernoulli':
            self.loss_y=lambda x, y : self._get_loss_bernoulli_y(
                x, y, self.y_param)
        else:
            self.loss_y=lambda x, y : self._get_loss_normal_y(
                x, y, self.y_param)
        self.loss_mdy=lambda x, y : self._get_loss_mdy(
                x, y, self.y_param)

        
    @staticmethod
    def _get_loss_bernoulli_y(x, x_dec_param, y_param):
        n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
                               K.floatx()), axis=-1)
        logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
        kl_y = K.expand_dims(
            hyp_obj.kl_diag_normal_vs_std_normal(y_param)/n_samples, dim=1)
        return logPx_g_z + kl_y

    

    @staticmethod
    def _get_loss_normal_y(x, x_dec_param, y_param):
        n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
                               K.floatx()), axis=-1)
        x_dim=K.cast(K.shape(x)[-1],'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        kl_y = K.expand_dims(
            hyp_obj.kl_diag_normal_vs_std_normal(y_param)/n_samples, dim=1)
        return logPx_g_z + kl_y

    
    @staticmethod
    def _get_loss_mdy(x, x_dec_param, y_param):
        n_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
                               K.floatx()), axis=-1)
        x_dim=K.cast(K.shape(x)[-1],'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        kl_y = K.expand_dims(
            hyp_obj.kl_diag_normal_vs_std_normal(y_param)/n_samples, dim=1)
        return 1e-10*logPx_g_z + kl_y


    def fit_y(self, x_train, x_val=None, optimizer=None,
            sample_weight_train=None, sample_weight_val=None, **kwargs):

        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model_y.compile(optimizer=optimizer, loss=self.loss_y,
                             sample_weight_mode='temporal')
        
        if isinstance(x_val, np.ndarray):
            if sample_weight_val is None:
                x_val=(x_val, x_val)
            else:
                x_val=(x_val, x_val, sample_weight_val)
            
        if isinstance(x_train, np.ndarray):
            return self.model_y.fit(x_train, x_train,
                                  sample_weight=sample_weight_train,
                                  validation_data=x_val, **kwargs)
        else:
            return self.model_y.fit_generator(x_train, validation_data=x_val, **kwargs)

    def fit_mdy(self, x_train, x_val=None, optimizer=None,
                sample_weight_train=None, sample_weight_val=None, **kwargs):

        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model_y.compile(optimizer=optimizer, loss=self.loss_mdy,
                             sample_weight_mode='temporal')
        
        if isinstance(x_val, np.ndarray):
            if sample_weight_val is None:
                x_val=(x_val, x_val)
            else:
                x_val=(x_val, x_val, sample_weight_val)
            
        if isinstance(x_train, np.ndarray):
            return self.model_y.fit(x_train, x_train,
                                  sample_weight=sample_weight_train,
                                  validation_data=x_val, **kwargs)
        else:
            return self.model_y.fit_generator(x_train, validation_data=x_val, **kwargs)
