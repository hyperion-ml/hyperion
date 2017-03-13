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
from keras.layers import Input, Lambda, Merge
from keras.models import Model, load_model, model_from_json

from ...distributions.core import PDF
from ..layers.sampling import BernoulliSampler, DiagNormalSampler
from .. import objectives as hyp_obj
from ..keras_utils import *

class VAE(PDF):

    def __init__(self, encoder_net, decoder_net, x_distribution):
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.x_distribution = x_distribution
        self.x_dim = 0
        self.z_dim = 0
        self.z_param = None
        self.model = None
        self.is_compiled = False        
        self.elbo_function = None
        
    def build(self):
        self.x_dim = self.encoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[0][-1]
        self._build_model()
        self._build_loss()

        
    def _build_model(self):
        x=Input(shape=(self.x_dim,))
        self.z_param = self.encoder_net(x)
        z=DiagNormalSampler()(self.z_param)
        x_dec_param=self.decoder_net(z)
        # hack for keras to work
        if self.x_distribution != 'bernoulli':
            x_dec_param=Merge(mode='concat', concat_axis=-1)(x_dec_param)

        self.model=Model(x, x_dec_param)

        
    def _build_loss(self):
        if self.x_distribution == 'bernoulli':
            self.loss=lambda x, y : VAE._get_loss_bernoulli(x, y, self.z_param)
        else:
            self.loss=lambda x, y : VAE._get_loss_normal(x, y, self.z_param)

            
    def _compile(self, optimizer=None):
        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss=self.loss)
        self.is_compiled=True
            

    def fit(self, x_train, x_val=None, optimizer=None,
            sample_weight_train=None, sample_weight_val=None, **kwargs):
        if not self.is_compiled:
            self._compile(optimizer)

        if isinstance(x_val, np.ndarray):
            if sample_weight_val is None:
                x_val=(x_val, x_val)
            else:
                x_val=(x_val, x_val, sample_weight_val)
            
        if isinstance(x_train, np.ndarray):
            return self.model.fit(x_train, x_train,
                                  sample_weight=sample_weight_train,
                                  validation_data=x_val, **kwargs)
        else:
            return self.model.fit_generator(x_train, validation_data=x_val, **kwargs)

        
    def elbo(self, x, nb_samples, batch_size=None):
        if not self.is_compiled:
            self._compile()

        if self.elbo_function is None:
            self.elbo_function = make_eval_function(self.model, self.loss)

        if batch_size is None:
            batch_size = x.shape[0]
            
        elbo = eval_loss(self.model, self.elbo_function, x, x, batch_size=batch_size)
        for i in xrange(1, nb_samples):
            elbo += eval_loss(self.model, self.elbo_function, x, x, batch_size=batch_size)
        return elbo/nb_samples

    
    def compute_qz_x(self, x, batch_size):
        return self.encoder_net.predict(x, batch_size=batch_size)

    
    def compute_px_z(self, z, batch_size):
        return self.decoder_net.predict(z, batch_size=batch_size)

    
    def decode_z(self, z, batch_size, sample_x=True):

        if not(sample_x):
            x_param = self.decoder_net.predict(z, batch_size=batch_size)
            if self.x_distribution == 'bernoulli':
                return x_param
            return x_param[:,:self.x_dim]
        
        z_input = Input(shape=(self.z_dim,))
        x_param = self.decoder_net(z_input)
        if self.x_distribution == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        else:
            x_sampled = DiagNormalSampler()(x_param)
        generator = Model(z_input, x_sampled)
        return generator.predict(z,batch_size=batch_size)


    def sample(self, nb_samples, batch_size, sample_x=True):
        z = np.random.normal(loc=0., scale=1.,
                             size=(nb_samples, self.z_dim))
        return self.decode_z(z, batch_size, sample_x)

    
    def get_config(self):
        enc_config = self.encoder_net.get_config()
        dec_config = self.decoder_net.get_config()
        config = {
            'class_name': self.__class__.__name__,
            'encoder_net': enc_config,
            'decoder_net': dec_config,
            'x_distribution': self.x_distribution }
        return config

        
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

        return cls(enconder_net, decoder_net, config['x_distribution'])

    
    # def _stack_output(self, x):
    #     if self.x_distribution == 'bernoulli':
    #         return x
    #     return Merge(mode='concat', concat_axis=-1)(x)

    
    # def _unstack_output(self, x):
    #     if self.x_distribution == 'bernoulli':
    #         return x
    #     return x[:,:self.x_dim], x[:,self.x_dim:]

    
    # @staticmethod
    # def _sample_normal(params):
    #     mu, logvar = params
    #     epsilon = K.random_normal(shape=K.shape(mu), mean=0., std=1.)
    #     return mu + K.exp(logvar / 2) * epsilon

    
    # @staticmethod
    # def _sample_bernoulli(p):
    #     return K.random.binomial(p=p,shape=K.shape(p))

    
    @staticmethod
    def _get_loss_bernoulli(x, x_dec_param, z_param):
        logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
        kl = hyp_obj.kl_diag_normal_vs_std_normal(z_param)
        return logPx_g_z + kl
        # z_mean, z_logvar=z_param
        # x_dim=K.cast(K.shape(x)[-1],'float32')
        # logPx_g_z_loss = x_dim * objectives.binary_crossentropy(x, x_dec_param)
        # kl_loss = - 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1)
        # return logPx_g_z_loss + kl_loss

    
    @staticmethod
    def _get_loss_normal(x, x_dec_param, z_param):
        x_dim=K.cast(K.shape(x)[-1],'int32')
        x_dec_param = [x_dec_param[:,:x_dim], x_dec_param[:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        kl = hyp_obj.kl_diag_normal_vs_std_normal(z_param)
        return logPx_g_z + kl
        
        # # x_decoded_mean, x_decoded_logvar = x_decoded
        # z_mean, z_logvar=z_param
        # x_dim=K.cast(K.shape(x)[-1],'int32')
        # log2pi=np.log(2*np.pi).astype('float32')
        # x_mean=x_dec_param[:,:x_dim]
        # x_logvar=x_dec_param[:,x_dim:]
        # x_var=K.exp(x_logvar)
        # dist2=K.square(x-x_mean)/x_var
        # logPx_g_z_loss=0.5*K.sum(log2pi+x_logvar+dist2, axis=-1)
        # kl_loss=-0.5*K.sum(1+z_logvar-K.square(z_mean)-K.exp(z_logvar), axis=-1)
        # return logPx_g_z_loss + kl_loss
    
