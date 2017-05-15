"""
Conditional VAE
"""

from __future__ import absolute_import
from __future__ import print_function

from six.moves import xrange

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Concatenate, Reshape
from keras.models import Model


from .. import objectives as hyp_obj
from ..layers import *
from .tied_vae_qyqz import TiedVAE_qYqZ


class TiedCVAE_qYqZ(TiedVAE_qYqZ):

    def __init__(self, encoder_net, decoder_net,
                 px_cond_form='diag_normal',
                 qy_form='diag_normal',
                 qz_form='diag_normal',
                 py_prior_form='std_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2):

        super(TiedCVAE_qYqZ,self).__init__(
            encoder_net, decoder_net, px_cond_form=px_cond_form,
            qy_form=qy_form, qz_form=qz_form,
            py_prior_form=py_prior_form, pz_prior_form=pz_prior_form,
            min_kl=min_kl)

        self.r_dim=0

        
    def build(self, num_samples=1, max_seq_length=None):
        self.x_dim=self.encoder_net.internal_input_shapes[0][-1]
        self.r_dim=self.encoder_net.internal_input_shapes[1][-1]
        self.y_dim=self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim=self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length=self.encoder_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        assert(self.r_dim==self.decoder_net.internal_input_shapes[2][-1])
        self.num_samples = num_samples
        self._build_model()
        self._build_loss()
        self.is_compiled = False

        
    def _build_model(self):
        x=Input(shape=(self.max_seq_length, self.x_dim,))
        r=Input(shape=(self.max_seq_length, self.r_dim,))
        qyz_param=self.encoder_net([x, r])

        if self.qz_form == 'diag_normal':
            self.qy_param=qyz_param[:2]
            self.qz_param=qyz_param[2:]
            z = DiagNormalSampler(num_samples=self.num_samples)(self.qz_param)
        else:
            self.qy_param=qyz_param[:3]
            self.qz_param=qyz_param[3:]
            z = NormalSampler(num_samples=self.num_samples)(self.qz_param)

        if self.qy_form == 'diag_normal':
            y = DiagNormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                              num_samples=self.num_samples)(self.qy_param)
        else:
            y = NormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                          num_samples=self.num_samples)(self.qy_param)
        
        if self.num_samples > 1:
            r_rep = Repeat(self.num_samples, axis=0)(r)
        else:
            r_rep = r

        x_dec_param=self.decoder_net([y, z, r_rep])
        # hack for keras to work
        if self.px_cond_form != 'bernoulli':
            if self.px_cond_form == 'normal':
                x_chol = Reshape((self.max_seq_length, self.x_dim**2))(x_dec_param[2])
                x_dec_param = [x_dec_param[0], x_dec_param[1], x_chol]
            x_dec_param=Concatenate(axis=-1)(x_dec_param)

        self.model=Model([x, r], x_dec_param)


    def fit(self, x_train, r_train, x_val=None, r_val=None,
            optimizer=None,
            sample_weight_train=None, sample_weight_val=None,
            **kwargs):
        if not self.is_compiled:
            self.compile(optimizer)
            
        if x_val is not None:
            assert(r_val is not None)
            if sample_weight_val is None:
                x_val = ([x_val, r_val], x_val)
            else:
                x_val = ([x_val, r_val], x_val, sample_weight_val)

        return self.model.fit([x_train, r_train], x_train,
                              sample_weight=sample_weight_train,
                              validation_data=x_val, **kwargs)
        
    
    def compute_qyz_x(self, x, r, batch_size):
        return self.encoder_net.predict([x, r], batch_size=batch_size)

    
    def compute_px_yz(self, y, z, r, batch_size):
        return self.decoder_net.predict([y, z, r], batch_size=batch_size)

    
    def decode_yz(self, y, z, r, batch_size, sample_x=True):
        if y.ndim == 2:
            y=np.expand_dims(y, axis=1)
        if y.shape[1]==1:
            y=np.tile(y, (1, z.shape[1],1))

        if not sample_x:
            x_param=self.decoder_net.predict([y, z, r], batch_size=batch_size)
            if self.px_cond_form=='bernoulli':
                return x_param
            return x_param[:,:,:self.x_dim]

        y_input = Input(shape=(self.max_seq_length, self.y_dim,))
        z_input = Input(shape=(self.max_seq_length, self.z_dim,))
        r_input = Input(shape=(self.max_seq_length, self.r_dim,))
        x_param = self.decoder_net([y_input, z_input, r_input])

        if self.px_cond_form == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        elif self.px_cond_form == 'diag_normal' :
            x_sampled = DiagNormalSampler()(x_param)
        elif self.px_cond_form == 'normal' :
            x_sampled = NormalSampler()(x_param)
        else:
            raise ValueError()

        generator = Model([y_input, z_input, r_input], x_sampled)
        return generator.predict([y, z, r], batch_size=batch_size)


    def generate(self, r, batch_size,sample_x=True):
        num_seqs=r.shape[0]
        num_samples=r.shape[1]
        y=np.random.normal(loc=0.,scale=1.,size=(num_seqs, 1, self.y_dim))
        z=np.random.normal(loc=0.,scale=1.,size=(num_seqs, num_samples,self.z_dim))
        return self.decode_yz(y, z, r, batch_size,sample_x)


    def generate_x_g_y(self, y, r, batch_size,sample_x=True):
        num_seqs=r.shape[0]
        num_samples=r.shape[1]
        z=np.random.normal(loc=0.,scale=1.,size=(num_seqs, num_samples, self.z_dim))
        return self.decode_yz(y, z, r, batch_size, sample_x)

