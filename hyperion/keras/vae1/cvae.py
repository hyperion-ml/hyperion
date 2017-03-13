"""
Conditional VAE
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

from ..layers.sampling import BernoulliSampler, DiagNormalSampler
from .. import objectives as hyp_obj
from .vae import VAE


class CVAE(VAE):

    def __init__(self, encoder_net, decoder_net, x_distribution):
        super(CVAE, self).__init__(encoder_net, decoder_net, x_distribution)
        self.r_dim=0

        
    def build(self):
        self.x_dim=self.encoder_net.internal_input_shapes[0][-1]
        self.r_dim=self.encoder_net.internal_input_shapes[1][-1]
        self.z_dim=self.decoder_net.internal_input_shapes[0][-1]
        assert(self.r_dim==self.decoder_net.internal_input_shapes[1][-1])
        self._build_model()
        self._build_loss()


    def _build_model(self):
        x=Input(shape=(self.x_dim,))
        r=Input(shape=(self.r_dim,))
        self.z_param=self.encoder_net([x, r])
        z=DiagNormalSampler()(self.z_param)
        x_dec_param=self.decoder_net([z, r])
        # hack for keras to work
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model([x, r], x_dec_param)

                
    def fit(self, x_train, r_train=None, x_val=None, r_val=None,
            optimizer=None, **kwargs):
        if not self.is_compiled:
            self._compile(optimizer)

        if isinstance(x_val, np.ndarray):
            x_val=([x_val, r_val], x_val)
            
        if isinstance(x_train, np.ndarray):
            assert(r_train is not None)
            return self.model.fit([x_train, r_train], x_train, validation_data=x_val, **kwargs)
        else:
            return self.model.fit_generator(x_train, validation_data=x_val, **kwargs)
        
    
    def compute_qz_x(self, x, r, batch_size):
        return self.encoder_net.predict([x, r], batch_size=batch_size)

    
    def compute_px_z(self, z, batch_size):
        return self.decoder_net.predict([z, r], batch_size=batch_size)

    def decode_z(self, z, r, batch_size, sample_x=True):
        if not(sample_x):
            return super(self.__class__,self).decode_z([z, r], batch_size, sample_x)

        z_input = Input(shape=(self.z_dim,))
        r_input = Input(shape=(self.r_dim,))
        zr_input = [z_input, r_input]
        x_param = self.decoder_net(zr_input)
        if self.x_distribution == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        else:
            x_sampled = DiagNormalSampler()(x_param)
        generator = Model(zr_input, x_sampled)
        return generator.predict([z, r],batch_size=batch_size)


    def sample(self, r, batch_size, sample_x=True):
        n_samples = r.shape[0]
        z = np.random.normal(loc=0., scale=1.,
                             size=(n_samples, self.z_dim))
        return self.decode_z(z, r, batch_size, sample_x)
