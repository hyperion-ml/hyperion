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
from keras.layers import Input, Lambda, Concatenate, Reshape
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers  import *

from .vae import VAE
from .tied_vae_qyqz import TiedVAE_qYqZ


class TiedVAE_qYqZgY(TiedVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net,
                 px_cond_form='diag_normal',
                 qy_form='diag_normal',
                 qz_form='diag_normal',
                 py_prior_form='std_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2):

        super(TiedVAE_qYqZgY,self).__init__(
            [], decoder_net, px_cond_form=px_cond_form,
            qy_form=qy_form, qz_form=qz_form,
            py_prior_form=py_prior_form, pz_prior_form=pz_prior_form,
            min_kl=min_kl)

        self.qy_net = qy_net
        self.qz_net = qz_net
        self.num_samples_y = 1
        self.num_samples_z = 1

        
    def build(self, num_samples_y=1, num_samples_z=1, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        self.num_samples_y = num_samples_y
        self.num_samples_z = num_samples_z
        self.num_samples = num_samples_y*num_samples_z
        self._build_model()
        self._build_loss()
        self.is_compiled = False
        self.elbo_function = None

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        self.qy_param = self.qy_net(x)

        if self.qy_form == 'diag_normal':
            y = DiagNormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                              num_samples=self.num_samples_y,
                                              one_sample_per_seq=False)(self.qy_param)
        else:
            y = NormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                          num_samples=self.num_samples_y,
                                          one_sample_per_seq=False)(self.qy_param)

        if self.num_samples_y > 1:
            x_rep = Repeat(self.num_samples_y, axis=0)(x)
        else:
            x_rep = x
        
        self.qz_param = self.qz_net([x_rep, y])
        
        self.encoder_net = Model(x, self.qy_param+self.qz_param)

        if self.qz_form == 'diag_normal':
            z = DiagNormalSampler(num_samples = self.num_samples_z)(self.qz_param)
        else:
            z = NormalSampler(num_samples = self.num_samples_z)(self.qz_param)

        if self.num_samples_z > 1:
            y_rep = Repeat(self.num_samples_z, axis=0)(y)
        else:
            y_rep = y
        # print('dims', self.qy_param[0].ndim,
        #       self.qy_param[1].ndim,
        #       self.qy_param[2].ndim,
        #       y_rep.ndim, z.ndim)
        x_dec_param=self.decoder_net([y_rep, z])
        if self.px_cond_form != 'bernoulli':
            if self.px_cond_form == 'normal':
                x_chol = Reshape((self.max_seq_length, self.x_dim**2))(x_dec_param[2])
                x_dec_param = [x_dec_param[0], x_dec_param[1], x_chol]
            if self.px_cond_form == 'normal_1chol':
                self.x_chol = x_dec_param[2]
                x_dec_param = [x_dec_param[0], x_dec_param[1]]
            x_dec_param=Concatenate(axis=-1)(x_dec_param)

        self.model=Model(x, x_dec_param)

        
    def _build_loss(self):
        
        seq_length = lambda x: K.sum(
            K.cast(K.any(K.not_equal(x, 0), axis=-1), K.floatx()), axis=-1)
            
        if self.px_cond_form == 'bernoulli':
            logPx_f = self._get_loss_bernoulli
        elif self.px_cond_form == 'diag_normal':
            logPx_f = self._get_loss_diag_normal
        elif self.px_cond_form == 'normal':
            logPx_f = self._get_loss_normal
        elif self.px_cond_form == 'normal_1chol':
            logPx_f = lambda x,y,z: self._get_loss_normal_1chol(x, y, self.x_chol, z)
        else:
            raise ValueError('Invalid Px cond %s' % self.px_cond_form)

        if self.pz_prior_form == 'std_normal':
            kl_z_f = hyp_obj.kl_normal_vs_std_normal
        else:
            kl_z_f = hyp_obj.kl_normal_vs_diag_normal
        kl_z = K.clip(kl_z_f(self.qz_param), self.min_kl, None)
        if self.num_samples_y > 1:
            r = K.reshape(kl_z, (-1, self.num_samples_y, self.max_seq_length))
            kl_z = K.mean(r, axis=1)

        
        if self.py_prior_form == 'std_normal':
            kl_y_f = hyp_obj.kl_normal_vs_std_normal
        else:
            kl_y_f = hyp_obj.kl_normal_vs_diag_normal
        kl_y = lambda x: K.expand_dims(
            K.clip(kl_y_f(self.qy_param), self.min_kl, None)/seq_length(x), axis=1)

        self.loss=(lambda x, y: logPx_f(x, y, self.num_samples) +
                   kl_z + kl_y(x))



    def get_config(self):
        qy_config = self.qy_net.get_config()
        qz_config = self.qz_net.get_config()
        config = {
            'qy_net': qy_config,
            'qz_net': qz_config}
        base_config = super(TiedVAE_qYqZgY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.qy.h5' % (file_path)
        self.qy_net.save(file_model)
        file_model = '%s.qz.h5' % (file_path)
        self.qz_net.save(file_model)
        file_model = '%s.dec.h5' % (file_path)
        self.decoder_net.save(file_model)

        
    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=VAE.load_config_from_json(f.read())

        file_model = '%s.qy.h5' % (file_path)
        qy_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.qz.h5' % (file_path)
        qz_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.dec.h5' % (file_path)
        decoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        k_args = ('px_cond_form', 'qz_form', 'pz_prior_form',
                  'qy_form', 'py_prior_form', 'kl_min')
        args = {k: config[k] for k in k_args if k in config }
        return cls(qy_net, qz_net, decoder_net, **args)

