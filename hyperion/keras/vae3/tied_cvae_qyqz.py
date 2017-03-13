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
from keras.layers import Input, Lambda, Merge
from keras.models import Model


from .. import objectives as hyp_obj
from ..layers.core import Repeat
from ..layers.sampling2 import *
from .tied_vae_qyqz import TiedVAE_qYqZ


class TiedCVAE_qYqZ(TiedVAE_qYqZ):

    def __init__(self, encoder_net, decoder_net, x_distribution):
        super(TiedCVAE_qYqZ,self).__init__(encoder_net, decoder_net, x_distribution)
        self.r_dim=0

        
    def build(self, nb_samples=1, max_seq_length=None):
        self.x_dim=self.encoder_net.internal_input_shapes[0][-1]
        self.r_dim=self.encoder_net.internal_input_shapes[1][-1]
        self.y_dim=self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim=self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length=self.encoder_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        assert(self.r_dim==self.decoder_net.internal_input_shapes[2][-1])
        self.nb_samples = nb_samples
        self._build_model()
        self._build_loss()
        self.is_compile = False

    def _build_model(self):
        x=Input(shape=(self.max_seq_length, self.x_dim,))
        r=Input(shape=(self.max_seq_length, self.r_dim,))
        yz_param=self.encoder_net([x, r])
        self.y_param=yz_param[:2]
        self.z_param=yz_param[2:]

        z = DiagNormalSampler(nb_samples=self.nb_samples)(self.z_param)
        y = DiagNormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                          nb_samples=self.nb_samples)(self.y_param)

        if self.nb_samples > 1:
            r_rep = Repeat(self.nb_samples, axis=0)(r)
        else:
            r_rep = r

        x_dec_param=self.decoder_net([y, z, r_rep])
        # hack for keras to work
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model([x, r], x_dec_param)

                
    def fit(self, x_train, r_train=None, sample_weight_train=None,
              x_val=None, r_val=None, sample_weight_val=None,
              optimizer=None, **kwargs):
        if not self.is_compiled:
            self.compile(optimizer)

        if isinstance(x_val, np.ndarray):
            if sample_weight_val is None:
                x_val=([x_val, r_val], x_val)
            else:
                x_val=([x_val, r_val], x_val, sample_weight_val)
            
        if isinstance(x_train, np.ndarray):
            assert(r_train is not None)
            return self.model.fit(
                [x_train, r_train], x_train, validation_data=x_val,
                sample_weight=sample_weight_train, **kwargs)
        else:
            return self.model.fit_generator(
                x_train, validation_data=x_val, **kwargs)
        
    
    def compute_qyz_x(self, x, r, batch_size):
        return self.encoder_net.predict([x, r], batch_size=batch_size)

    
    def compute_px_yz(self, y, z, r, batch_size):
        return self.decoder_net.predict([y, z, r], batch_size=batch_size)

    
    def decode_yz(self, y, z, r, batch_size, sample_x=True):
        if y.ndim == 2:
            y=np.expand_dims(y,axis=1)
        if y.shape[1]==1:
            y=np.tile(y, (1, z.shape[1],1))

        if not(sample_x):
            x_param=self.decoder_net.predict([y, z, r], batch_size=batch_size)
            if self.x_distribution=='bernoulli':
                return x_param
            return x_param[:,:,:self.x_dim]

        y_input=Input(shape=(self.max_seq_length, self.y_dim,))
        z_input=Input(shape=(self.max_seq_length, self.z_dim,))
        r_input=Input(shape=(self.max_seq_length, self.r_dim,))
        x_param=self.decoder_net([y_input, z_input, r_input])
        if self.x_distribution == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        else:
            x_sampled = DiagNormalSampler()(x_param)
        generator = Model([y_input, z_input, r_input], x_sampled)
        return generator.predict([y, z, r], batch_size=batch_size)


    def generate(self, r, batch_size,sample_x=True):
        n_signals=r.shape[0]
        n_samples=r.shape[1]
        y=np.random.normal(loc=0.,scale=1.,size=(n_signals, 1, self.y_dim))
        z=np.random.normal(loc=0.,scale=1.,size=(n_signals, n_samples,self.z_dim))
        return self.decode_yz(y, z, r, batch_size,sample_x)


    def generate_x_g_y(self, y, r, batch_size,sample_x=True):
        n_signals=r.shape[0]
        n_samples=r.shape[1]
        z=np.random.normal(loc=0.,scale=1.,size=(n_signals, n_samples, self.z_dim))
        return self.decode_yz(y, z, r, batch_size, sample_x)

