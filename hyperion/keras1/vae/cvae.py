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
from keras.layers import Input, Lambda, Merge, Reshape
from keras.models import Model

from .. import objectives as hyp_obj
from ..layers.core import Repeat
from ..layers.sampling import BernoulliSampler, DiagNormalSampler, NormalSampler
from .vae import VAE


class CVAE(VAE):

    def __init__(self, encoder_net, decoder_net,
                 px_cond_form='diag_normal', qz_form='diag_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2):
        super(CVAE, self).__init__(encoder_net, decoder_net,
                                   px_cond_form=px_cond_form,
                                   qz_form=qz_form,
                                   pz_prior_form=pz_prior_form,
                                   min_kl=min_kl)
        self.r_dim=0

        
    def build(self, nb_samples=1):
        self.x_dim=self.encoder_net.internal_input_shapes[0][-1]
        self.r_dim=self.encoder_net.internal_input_shapes[1][-1]
        self.z_dim=self.decoder_net.internal_input_shapes[0][-1]
        assert(self.r_dim==self.decoder_net.internal_input_shapes[1][-1])
        self.nb_samples = nb_samples
        self._build_model()
        self._build_loss()
        self.is_compiled=False

        
    def _build_model(self):
        x = Input(shape=(self.x_dim,))
        r = Input(shape=(self.r_dim,))
        self.qz_param = self.encoder_net([x, r])
        if self.qz_form == 'diag_normal':
            z = DiagNormalSampler(nb_samples=self.nb_samples)(self.qz_param)
        else:
            z = NormalSampler(nb_samples=self.nb_samples)(self.qz_param)

        if self.nb_samples > 1:
            r_rep = Repeat(self.nb_samples, axis=0)(r)
        else:
            r_rep = r
        x_dec_param=self.decoder_net([z, r_rep])
        # hack for keras to work
        if self.px_cond_form != 'bernoulli':
            if self.px_cond_form == 'normal':
                x_chol = Reshape((self.x_dim**2,))(x_dec_param[2])
                x_dec_param = [x_dec_param[0], x_dec_param[1], x_chol]
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

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
        

    def elbo(self, x, r, nb_samples=1, batch_size=None):
        if not self.is_compiled:
            self.compile()

        if self.elbo_function is None:
            self.elbo_function = make_eval_function(self.model, self.loss)

        if batch_size is None:
            batch_size = x.shape[0]
            
        elbo = - eval_loss(self.model, self.elbo_function, [x, r], x,
                           batch_size=batch_size)
        for i in xrange(1, nb_samples):
            elbo -= eval_loss(self.model, self.elbo_function, [x, r], x,
                              batch_size=batch_size)
        return elbo/nb_samples

    
    def compute_qz_x(self, x, r, batch_size):
        return self.encoder_net.predict([x, r], batch_size=batch_size)

    
    def compute_px_z(self, z, batch_size):
        return self.decoder_net.predict([z, r], batch_size=batch_size)

    
    def decode_z(self, z, r, batch_size, sample_x=True):
        if not sample_x:
            return super(CVAE, self).decode_z([z, r], batch_size, sample_x)

        z_input = Input(shape=(self.z_dim,))
        r_input = Input(shape=(self.r_dim,))
        zr_input = [z_input, r_input]
        x_param = self.decoder_net(zr_input)
        
        if self.px_cond_form == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        elif self.px_cond_form == 'diag_normal' :
            x_sampled = DiagNormalSampler()(x_param)
        elif self.px_cond_form == 'normal' :
            x_sampled = NormalSampler()(x_param)
        else:
            raise ValueError()

        generator = Model(zr_input, x_sampled)
        return generator.predict([z, r],batch_size=batch_size)


    def generate(self, r, batch_size, sample_x=True):
        n_samples = r.shape[0]
        z = np.random.normal(loc=0., scale=1.,
                             size=(n_samples, self.z_dim))
        return self.decode_z(z, r, batch_size, sample_x)
