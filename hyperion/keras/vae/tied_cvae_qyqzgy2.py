""""
Tied Variational Autoencoder with 2 latent variables:
 - y: variable tied across all the samples in the segment
 - z: untied variable,  it has a different value for each frame

Factorization of the posterior:
   q(y_i,Z_i)=q(y_i) \prod_j q(z_{ij} | y_i)
n
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
from ..layers import *

from .tied_cvae_qyqz import TiedCVAE_qYqZ
from .tied_vae_qyqzgy import TiedVAE_qYqZgY

class TiedCVAE_qYqZgY2(TiedCVAE_qYqZ):

    def __init__(self, qy_net, qz_net, decoder_net,
                 pt_net=None,
                 px_cond_form='diag_normal',
                 qy_form='diag_normal',
                 qz_form='diag_normal',
                 py_prior_form='std_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2, loss_weights=None):

        super(TiedCVAE_qYqZgY2,self).__init__(
            [], decoder_net, px_cond_form=px_cond_form,
            qy_form=qy_form, qz_form=qz_form,
            py_prior_form=py_prior_form, pz_prior_form=pz_prior_form,
            min_kl=min_kl)
                                             
        self.qy_net = qy_net
        self.qz_net = qz_net
        self.pt_net = pt_net
        self.t_dim = 0
        self.loss_weights = loss_weights
        self.num_samples_y = 1
        self.num_samples_z = 1


    def build(self, num_samples_y=1, num_samples_z=1, max_seq_length=None):
        self.x_dim = self.qy_net.internal_input_shapes[0][-1]
        self.r_dim = self.qy_net.internal_input_shapes[1][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if self.pt_net is not None:
            self.t_dim = self.pt_net.internal_output_shapes[-1]
        if max_seq_length is None:
            self.max_seq_length = self.qy_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        assert(self.r_dim==self.decoder_net.internal_input_shapes[2][-1])
        self.num_samples_y = num_samples_y
        self.num_samples_z = num_samples_z
        self.num_samples = num_samples_y*num_samples_z
        self._build_model()
        self._build_loss()
        self.is_compiled = False

        
    def _build_model(self):
        x = Input(shape=(self.max_seq_length, self.x_dim,))
        r = Input(shape=(self.max_seq_length, self.r_dim,))
        
        self.qy_param = self.qy_net([x, r])
        if self.qy_form == 'diag_normal':
            y = DiagNormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                              num_samples=self.num_samples_y,
                                              one_sample_per_seq=False)(self.qy_param)
            yt = DiagNormalSampler(num_samples=1)(self.qy_param)
        else:
            y = NormalSamplerFromSeqLevel(seq_length=self.max_seq_length,
                                          num_samples=self.num_samples_y,
                                          one_sample_per_seq=False)(self.qy_param)
        

        if self.num_samples_y > 1:
            x_rep = Repeat(self.num_samples_y, axis=0)(x)
            r_rep = Repeat(self.num_samples_y, axis=0)(r)
        else:
            x_rep = x
            r_rep = r

        self.qz_param = self.qz_net([x_rep, y, r_rep])
        
        self.encoder_net = Model([x, r], self.qy_param+self.qz_param)

        
        if self.qz_form == 'diag_normal':
            z = DiagNormalSampler(num_samples = self.num_samples_z)(self.qz_param)
        else:
            z = NormalSampler(num_samples = self.num_samples_z)(self.qz_param)


        if self.num_samples_z > 1:
            y_rep = Repeat(self.num_samples_z, axis=0)(y)
            r_rep = Repeat(self.num_samples, axis=0)(r)
        else:
            y_rep = y

        x_dec_param=self.decoder_net([y_rep, z, r_rep])
        if self.px_cond_form != 'bernoulli':
            if self.px_cond_form == 'normal':
                x_chol = Reshape((self.max_seq_length, self.x_dim**2))(x_dec_param[2])
                x_dec_param = [x_dec_param[0], x_dec_param[1], x_chol]
            x_dec_param=Concatenate(axis=-1)(x_dec_param)

        if self.pt_net is None:
            self.model=Model([x, r], x_dec_param)
        else:
            t_param = self.pt_net(yt)
            self.model=Model([x, r], [x_dec_param, t_param])


    def _build_loss(self):
        
        seq_length = lambda x: K.sum(
            K.cast(K.any(K.not_equal(x, 0), axis=-1), K.floatx()), axis=-1)
            
        if self.px_cond_form == 'bernoulli':
            logPx_f = self._get_loss_bernoulli
        elif self.px_cond_form == 'diag_normal':
            logPx_f = self._get_loss_diag_normal
        elif self.px_cond_form == 'normal':
            logPx_f = self._get_loss_normal
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
        kl_y = lambda x : K.expand_dims(K.clip(kl_y_f(self.qy_param), self.min_kl, None)
                             /seq_length(x), axis=1)
        
        self.loss=(lambda x, y: logPx_f(x, y, self.num_samples) +
                   kl_z + kl_y(x))
            
        if self.pt_net is not None:
            self.loss = [self.loss, objectives.categorical_crossentropy]


    def compile(self, optimizer=None):
        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)

        if len(self.loss) == 1:
            swm = 'temporal'
        else:
            swm = ['temporal', None]
                
        self.model.compile(optimizer=optimizer, loss=self.loss, loss_weights=self.loss_weights,
                           sample_weight_mode=swm)
        self.is_compiled=True

        
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
        if self.pt_net is not None:
            file_model = '%s.pt.h5' % (file_path)
            self.pt_net.save(file_model)

    def get_config(self):
        config = { 'loss_weights': self.loss_weights}
        base_config = super(TiedCVAE_qYqZgY2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=TiedCVAE_qYqZgY2.load_config_from_json(f.read())

        file_model = '%s.qy.h5' % (file_path)
        qy_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.qz.h5' % (file_path)
        qz_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.dec.h5' % (file_path)
        decoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.pt.h5' % (file_path)
        if os.path.isfile(file_model):
            pt_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        else:
            pt_net = None
        
        k_args = ('px_cond_form', 'qz_form', 'pz_prior_form',
                  'qy_form', 'py_prior_form', 'kl_min' , 'loss_weights')
        args = {k: config[k] for k in k_args if k in config }
        return cls(qy_net, qz_net, decoder_net, pt_net=pt_net, **args)
