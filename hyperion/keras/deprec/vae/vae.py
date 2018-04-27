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

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Concatenate, Reshape
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ...pdfs.core import PDF
from ..layers.sampling import BernoulliSampler, DiagNormalSampler, NormalSampler



class VAE(PDF):

    def __init__(self, encoder_net, decoder_net, pt_net=None,
                 px_cond_form='diag_normal', pt_cond_form='categorical',
                 qz_form='diag_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2, **kwargs):
        
        super(VAE, self).__init__(x_dim=0, **kwargs)
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.pt_net = pt_net
        self.px_cond_form = px_cond_form
        self.pt_cond_form = pt_cond_form
        self.qz_form = qz_form
        self.pz_prior_form = pz_prior_form
        self.min_kl = min_kl
        self.x_dim = 0
        self.z_dim = 0
        self.qz_param = None
        self.model = None
        self.is_compiled = False        
        self.elbo_function = None
        self.num_samples = 1
        self.x_chol = None

        
    def build(self, num_samples=1):
        self.x_dim = self.encoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.num_samples = num_samples
        self._build_model()
        self._build_loss()
        self.is_compiled = False

        
    def _build_model(self):
        x = Input(shape=(self.x_dim,))
        self.qz_param = self.encoder_net(x)
        if self.qz_form == 'diag_normal':
            z = DiagNormalSampler(num_samples=self.num_samples)(self.qz_param)
        else:
            z = NormalSampler(num_samples=self.num_samples)(self.qz_param)
            
        x_dec_param = self.decoder_net(z)
        # hack for keras to work
        if self.px_cond_form != 'bernoulli':
            if self.px_cond_form == 'normal':
                x_chol = Reshape((self.x_dim**2,))(x_dec_param[2])
                x_dec_param = [x_dec_param[0], x_dec_param[1], x_chol]
            if self.px_cond_form == 'normal_1chol':
                self.x_chol = x_dec_param[2]
                x_dec_param = [x_dec_param[0], x_dec_param[1]]
            x_dec_param=Concatenate(axis=-1)(x_dec_param)

        if self.pt_net is not None:
            t_param = self.pt_net(z)
            x_dec_param=Concatenate(axis=-1)([x_dec_param, t_param])
            
        self.model=Model(x, x_dec_param)

        
    def _build_loss(self):
        if self.px_cond_form == 'bernoulli':
            logPx_f = VAE._get_loss_bernoulli
        elif self.px_cond_form == 'diag_normal':
            logPx_f = VAE._get_loss_diag_normal
        elif self.px_cond_form == 'normal':
            logPx_f = VAE._get_loss_normal
        elif self.px_cond_form == 'normal_1chol':
            logPx_f = lambda x, y, z: VAE._get_loss_normal_1chol(x, y, self.x_chol, z)
        else:
            raise ValueError('Invalid Px cond %s' % self.px_cond_form)

        if self.pz_prior_form == 'std_normal':
            kl_f = hyp_obj.kl_normal_vs_std_normal
        else:
            kl_f = hyp_obj.kl_normal_vs_diag_normal

        self.loss=(lambda x, y: logPx_f(x, y, self.num_samples) +
                   K.clip(kl_f(self.qz_param), self.min_kl, None))
        
            
    def compile(self, optimizer=None):
        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss=self.loss)
        self.is_compiled=True
            

    def fit(self, x_train, x_val=None, optimizer=None,
            sample_weight_train=None, sample_weight_val=None, **kwargs):
        if not self.is_compiled:
            self.compile(optimizer)

        if x_val is not None:
            if sample_weight_val is None:
                x_val = (x_val, x_val)
            else:
                x_val = (x_val, x_val, sample_weight_val)
            
        return self.model.fit(x_train, x_train,
                              sample_weight=sample_weight_train,
                              validation_data=x_val, **kwargs)

        
    def fit_generator(self, x_train, x_val=None, optimizer=None, **kwargs):
        if not self.is_compiled:
            self.compile(optimizer)

        return self.model.fit_generator(x_train, validation_data=x_val,
                                        **kwargs)

        
    def elbo(self, x, num_samples=1, batch_size=None):
        if not self.is_compiled:
            self.compile()

        if self.elbo_function is None:
            self.elbo_function = make_eval_function(self.model, self.loss)

        if batch_size is None:
            batch_size = x.shape[0]
            
        elbo = - eval_loss(self.model, self.elbo_function, x, x,
                           batch_size=batch_size)
        for i in xrange(1, num_samples):
            elbo -= eval_loss(self.model, self.elbo_function, x, x,
                              batch_size=batch_size)
        return elbo/num_samples

    
    def compute_qz_x(self, x, batch_size):
        return self.encoder_net.predict(x, batch_size=batch_size)

    
    def compute_px_z(self, z, batch_size):
        return self.decoder_net.predict(z, batch_size=batch_size)

    
    def decode_z(self, z, batch_size, sample_x=True):

        if not(sample_x):
            x_param = self.decoder_net.predict(z, batch_size=batch_size)
            if self.px_cond_form == 'bernoulli':
                return x_param
            return x_param[:,:self.x_dim]
        
        z_input = Input(shape=(self.z_dim,))
        x_param = self.decoder_net(z_input)
        
        if self.px_cond_form == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        elif self.px_cond_form == 'diag_normal' :
            x_sampled = DiagNormalSampler()(x_param)
        elif self.px_cond_form == 'normal' :
            x_sampled = NormalSampler()(x_param)
        else:
            raise ValueError()
        
        generator = Model(z_input, x_sampled)
        return generator.predict(z,batch_size=batch_size)


    def generate(self, num_samples, batch_size, sample_x=True):
        z = np.random.normal(loc=0., scale=1.,
                             size=(num_samples, self.z_dim))
        return self.decode_z(z, batch_size, sample_x)

    
    def get_config(self):
        enc_config = self.encoder_net.get_config()
        dec_config = self.decoder_net.get_config()
        config = {
            'encoder_net': enc_config,
            'decoder_net': dec_config,
            'px_cond_form': self.px_cond_form,
            'qz_form': self.qz_form,
            'pz_prior_form': self.pz_prior_form,
            'min_kl': self.min_kl }
        base_config = super(VAE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

        
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.enc.h5' % (file_path)
        self.encoder_net.save(file_model)
        file_model = '%s.dec.h5' % (file_path)
        self.decoder_net.save(file_model)

        
    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=VAE.load_config_from_json(f.read())

        file_model = '%s.enc.h5' % (file_path)
        encoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.dec.h5' % (file_path)
        decoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        k_args = ('px_cond_form', 'qz_form', 'pz_prior_form', 'min_kl')
        args = { k:config[k] for k in k_args if k in config}
        return cls(enconder_net, decoder_net, **args)


    @staticmethod
    def _get_loss_bernoulli(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
            
        logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
        if num_samples > 1:
            r = K.reshape(logPx_g_z, (-1, num_samples))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z

    
    @staticmethod
    def _get_loss_diag_normal(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
            
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        x_dec_param = [x_dec_param[:,:x_dim], x_dec_param[:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        if num_samples > 1:
            r = K.reshape(logPx_g_z, (-1, num_samples))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z

    
    @staticmethod
    def _get_loss_normal(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        x_dec_param = [x_dec_param[:,:x_dim], x_dec_param[:,x_dim:2*x_dim],
                       K.reshape(x_dec_param[:,2*x_dim:], (-1, x_dim, x_dim))]
        logPx_g_z = hyp_obj.normal(x, x_dec_param)
        if num_samples > 1:
            r = K.reshape(logPx_g_z, (-1, num_samples))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z
            
    
    @staticmethod
    def _get_loss_normal_1chol(x, x_dec_param, x_chol, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        x_dec_param = [x_dec_param[:,:x_dim], x_dec_param[:,x_dim:], x_chol]
        logPx_g_z = hyp_obj.normal_1chol(x, x_dec_param)
        if num_samples > 1:
            r = K.reshape(logPx_g_z, (-1, num_samples))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z
            
    
