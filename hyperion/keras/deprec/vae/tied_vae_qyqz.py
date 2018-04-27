""""
Tied Variational Autoencoder with 2 latent variables:
 - y: variable tied across all the samples in the segment
 - z: untied variable,  it has a different value for each frame

Factorization of the posterior:
   q(y_i,Z_i)=q(y_i) \prod_j q(z_{ij})

The parameters \phi of all the  variational distributions are given by 
a unique NN.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import time

import numpy as np
import scipy.linalg as sla

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Lambda, Concatenate, Reshape
from keras.models import Model, load_model, model_from_json

from ...hyp_defs import float_keras
from ...utils.math import invert_pdmat
from ...pdfs.plda import PLDABase
from ...transforms import LNorm
from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *


from .vae import VAE

class TiedVAE_qYqZ(VAE):

    def __init__(self, encoder_net, decoder_net,
                 px_cond_form='diag_normal',
                 qy_form='diag_normal',
                 qz_form='diag_normal',
                 py_prior_form='std_normal',
                 pz_prior_form='std_normal',
                 min_kl=0.2):

        super(TiedVAE_qYqZ, self).__init__(
            encoder_net, decoder_net, px_cond_form=px_cond_form,
            qz_form=qz_form, pz_prior_form=pz_prior_form,
            min_kl=min_kl)
        
        self.qy_form = qy_form
        self.py_prior_form = py_prior_form
        self.y_dim = 0
        self.qy_param = None
        self.max_seq_length = 0

        
    def build(self, num_samples=1, max_seq_length=None):
        self.x_dim = self.encoder_net.internal_input_shapes[0][-1]
        self.y_dim = self.decoder_net.internal_input_shapes[0][-1]
        self.z_dim = self.decoder_net.internal_input_shapes[1][-1]
        if max_seq_length is None:
            self.max_seq_length = self.encoder_net.internal_input_shapes[0][-2]
        else:
            self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self._build_model()
        self._build_loss()
        self.is_compiled = False
        self.elbo_function = None

        
    def _build_model(self):
        x=Input(shape=(self.max_seq_length, self.x_dim,))
        qyz_param=self.encoder_net(x)

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

        x_dec_param=self.decoder_net([y, z])
        # hack for keras to work
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
            logPx_f = lambda x, y, z: self._get_loss_normal_1chol(x, y, self.x_chol, z)
        else:
            raise ValueError('Invalid Px cond %s' % self.px_cond_form)

        if self.pz_prior_form == 'std_normal':
            kl_z_f = hyp_obj.kl_normal_vs_std_normal
        else:
            kl_z_f = hyp_obj.kl_normal_vs_diag_normal
        kl_z = K.clip(kl_z_f(self.qz_param), self.min_kl, None)

        if self.py_prior_form == 'std_normal':
            kl_y_f = hyp_obj.kl_normal_vs_std_normal
        else:
            kl_y_f = hyp_obj.kl_normal_vs_diag_normal
        kl_y = lambda x: K.expand_dims(
            K.clip(kl_y_f(self.qy_param), self.min_kl, None)/seq_length(x), axis=1)

        self.loss=(lambda x, y: logPx_f(x, y, self.num_samples) +
                   kl_z + kl_y(x))
                   

            
    def compile(self, optimizer=None):
        if optimizer is None:
            optimizer=optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss=self.loss,
                           sample_weight_mode='temporal')
        self.is_compiled=True


    def compute_qyz_x(self, x, batch_size):
        return self.encoder_net.predict(x, batch_size=batch_size)
        
    def compute_qy_x(self, x, batch_size):
        return self.encoder_net.predict(x, batch_size=batch_size)[:2]

    
    def compute_px_yz(self, y, z, batch_size):
        return self.decoder_net.predict([y, z], batch_size=batch_size)

    
    def decode_yz(self, y, z, batch_size, sample_x=True):
        if y.ndim == 2:
            y=np.expand_dims(y, axis=1)
        if y.shape[1]==1:
            y=np.tile(y, (1, z.shape[1], 1))

        if not sample_x:
            x_param = self.decoder_net.predict([y, z], batch_size=batch_size)
            if self.px_cond_form=='bernoulli':
                return x_param
            return x_param[:,:,:self.x_dim]

        y_input = Input(shape=(self.max_seq_length, self.y_dim,))
        z_input = Input(shape=(self.max_seq_length, self.z_dim,))
        x_param = self.decoder_net([y_input, z_input])

        if self.px_cond_form == 'bernoulli' :
            x_sampled = BernoulliSampler()(x_param)
        elif self.px_cond_form == 'diag_normal' :
            x_sampled = DiagNormalSampler()(x_param)
        elif self.px_cond_form == 'normal' :
            x_sampled = NormalSampler()(x_param)
        elif self.px_cond_form == 'normal_1chol' :
            pass
        else:
            raise ValueError()

        generator = Model([y_input, z_input], x_sampled)
        return generator.predict([y, z], batch_size=batch_size)


    def generate(self, num_seqs, num_samples, batch_size,sample_x=True):
        y=np.random.normal(loc=0., scale=1., size=(num_seqs, 1, self.y_dim))
        z=np.random.normal(loc=0., scale=1., size=(num_seqs, num_samples,self.z_dim))
        return self.decode_yz(y, z, batch_size, sample_x)


    def generate_x_g_y(self, y, num_samples, batch_size, sample_x=True):
        num_seqs=y.shape[0]
        z=np.random.normal(loc=0., scale=1., size=(num_seqs, num_samples, self.z_dim))
        return self.decode_yz(y, z, batch_size, sample_x)

            
    def elbo(self, x, num_samples=1, batch_size=None, mask_value=0):
        if not self.is_compiled:
            self.compile()

        if self.elbo_function is None:
            self.elbo_function = make_eval_function(self.model, self.loss)

        if batch_size is None:
            batch_size = x.shape[0]

        sw = np.any(np.not_equal(x, mask_value),
                    axis=-1, keepdims=False).astype(float_keras())

        elbo = - eval_loss(self.model, self.elbo_function, x, x,
                           batch_size=batch_size, sample_weight=sw)
        for i in xrange(1, num_samples):
            elbo -= eval_loss(self.model, self.elbo_function, x, x,
                              batch_size=batch_size, sample_weight=sw)

        return elbo/num_samples


    
    def eval_llr_1vs1(self, x1, x2, score_mask=None, method='elbo', num_samples=1):
        if method == 'elbo':
            return self.eval_llr_1vs1_elbo(x1, x2, score_mask, num_samples)
        if method == 'cand':
            return self.eval_llr_1vs1_cand(x1, x2, score_mask)
        if method == 'qscr':
            return self.eval_llr_1vs1_qscr(x1, x2, score_mask)


        
    def eval_llr_1vs1_elbo(self, x1, x2, score_mask=None, num_samples=1):
        
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x1
        elbo_1 = self.elbo(xx, num_samples)

        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x2
        elbo_2 = self.elbo(xx, num_samples)
        p
        scores = - (np.expand_dims(elbo_1, axis=-1) +
                    np.expand_dims(elbo_2, axis=-1).T)

        for i in xrange(x1.shape[0]):
            xx[:,1,:] = x1[i,:]
            elbo_3 = self.elbo(xx, num_samples)
            scores[i,:] += elbo_3
        return scores


    
    def eval_llr_1vs1_cand(self, x1, x2, score_mask=None):
        if self.qy_form == 'diag_normal':
            logqy = lambda x, y, z: self._eval_logqy_eq_0_diagcov(x, y)
        else:
            logqy = lambda x, y, z: self._eval_logqy_eq_0_fullcov(x, y, z)
        
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x1
        y_mean, y_logvar, y_chol = self.compute_qyz_x(xx, batch_size=x1.shape[0])[:3]
        logq_1 = logqy(y_mean, y_logvar, y_chol)
        
        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x2
        y_mean, y_logvar, y_chol = self.compute_qyz_x(xx, batch_size=x2.shape[0])[:3]
        logq_2 = logqy(y_mean, y_logvar, y_chol)

        scores = np.expand_dims(logq_1, axis=-1) + np.expand_dims(logq_2, axis=-1).T

        for i in xrange(x1.shape[0]):
            xx[:,1,:] = x1[i,:]
            y_mean, y_logvar, y_chol= self.compute_qyz_x(xx, batch_size=x2.shape[0])[:3]
            scores[i,:] -= logqy(y_mean, y_logvar, y_chol)
        return scores


    
    @staticmethod
    def _eval_logqy_eq_0_diagcov(mu, logvar):
        var = np.exp(logvar)
        return -0.5*np.sum(logvar + mu**2/var, axis=-1)


    
    @staticmethod
    def _eval_logqy_eq_0_fullcov(mu, logvar, choly):
        #assume all have the same cov
        choly = choly[0,:,:]
        ichol = sla.inv(choly)
        mu = np.dot(mu, ichol)
        var = np.exp(logvar)
        return -0.5*np.sum(logvar + mu**2/var, axis=-1)

    

    def _eval_llr_1vs1_qscr_diagcov(self, x1, x2, score_mask=None):
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x1
        y1_mean, y1_logvar = self.compute_qyz_x(xx, batch_size=x1.shape[0])[:2]
        y1_p = np.exp(-y1_logvar)
        r1 = y1_p*y1_mean
        logq_1 = -0.5*np.sum(y1_logvar + r1**2/y1_p, axis=-1)
        
        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x2
        y2_mean, y2_logvar = self.compute_qyz_x(xx, batch_size=x2.shape[0])[:2]
        y2_p = np.exp(-y2_logvar)
        r2 = y2_p*y2_mean
        logq_2 = -0.5*np.sum(y2_logvar + r2**2/y2_p, axis=-1)

        scores = np.expand_dims(logq_1, axis=-1) + np.expand_dims(logq_2, axis=-1).T

        for i in xrange(x1.shape[0]):
            p_3 = y1_p[i,:] + y2_p - np.ones_like(y2_p)
            r3 = r1[i,:] + r2
            scores[i,:] += 0.5*np.sum(-np.log(p_3) + r3**2/p_3, axis=-1)
        return scores


    
    def _eval_llr_1vs1_qscr_fullcov(self, x1, x2, score_mask=None):
        xx_shape = (x1.shape[0], self.max_seq_length, x1.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x1
        y1_mean, y1_logvar, y1_chol= self.compute_qyz_x(xx, batch_size=x1.shape[0])[:3]
        #assume all have the same chol
        iy1_chol = sla.inv(y1_chol[0])
        y1_p = np.exp(-y1_logvar)
        P1=np.dot(iy1_chol*y1_p[0], iy1_chol.T)
        r1 = np.dot(y1_mean, P1)
        logq_1 = -0.5*np.sum(y1_logvar + r1*y1_mean, axis=-1)
        
        xx_shape = (x2.shape[0], self.max_seq_length, x2.shape[1])
        xx = np.zeros(xx_shape, float_keras())
        xx[:,0,:] = x2
        y2_mean, y2_logvar, y2_chol= self.compute_qyz_x(xx, batch_size=x2.shape[0])[:3]
        #assume all have the same chol
        iy2_chol = sla.inv(y2_chol[0])
        y2_p = np.exp(-y2_logvar)
        P2=np.dot(iy2_chol*y2_p[0], iy2_chol.T)
        r2 = np.dot(y2_mean, P2)
        logq_2 = -0.5*np.sum(y2_logvar + r2*y2_mean, axis=-1)

        scores = np.expand_dims(logq_1, axis=-1) + np.expand_dims(logq_2, axis=-1).T

        P3 = P1+P2-np.eye(P1.shape[0])
        iP3, _, ldP3 = invert_pdmat(P3, right_inv=True, return_logdet=True)
        for i in xrange(x1.shape[0]):
            r3 = r1[i,:] + r2
            r3iP3 = iP3(r3)
            scores[i,:] += -0.5*ldP3 + 0.5*np.sum(r3iP3*r3, axis=-1)
        return scores


    
    def eval_llr_1vs1_qscr(self, x1, x2, score_mask=None):
        if self.qy_form == 'diag_normal':
            return self._eval_llr_1vs1_qscr_diagcov(x1, x2)
        return self._eval_llr_1vs1_qscr_fullcov(x1, x2)
    


    def eval_llr_Nvs1(self, x1, ids1, x2, score_mask=None,
                      pool_method='vavg-lnorm', eval_method='qscr',
                      num_samples=1):
        
        if  pool_method == 'savg':
            return self.eval_llr_Nvs1_savg(x1, ids1, x2, score_mask=score_mask,
                                           method=eval_method,
                                           num_samples=num_samples)
        if pool_method == 'vavg':
            return self.eval_llr_Nvs1_vavg(x1, ids1, x2, score_mask=score_mask,
                                           do_lnorm=False, method=eval_method,
                                           num_samples=num_samples)
        if pool_method == 'vavg-lnorm':
            return self.eval_llr_Nvs1_vavg(x1, ids1, x2, score_mask=score_mask,
                                           do_lnorm=True, method=eval_method,
                                           num_samples=num_samples)
        if pool_method == 'book':
            return self.eval_llr_Nvs1_book(x1, ids1, x2, score_mask=score_mask,
                                           method=eval_method,
                                           num_samples=num_samples)

        

    def eval_llr_Nvs1_savg(self, x1, ids1, x2, score_mask=None, method='qscr',
                           num_samples=1):
        scores_1vs1 = self.eval_llr_1vs1(x1, x2, score_mask=score_mask,
                                         method=method, num_samples=num_samples)
        N, F, _ = self.compute_stats(scores_1vs1, ids1)
        scores = F/N[:, None]
        return scores

            

    def eval_llr_Nvs1_vavg(self, x1, ids1, x2, score_mask=None, do_lnorm=True, method='qscr',
                           num_samples=1):
        N, F, _ = PLDABase.compute_stats_hard(x1, class_ids=ids1)
        x1=F/np.expand_dims(N, axis=-1)
        if do_lnorm:
            lnorm=LNorm()
            x1=lnorm.predict(x1)
            x2=lnorm.predict(x2)
            
        return self.eval_llr_1vs1(x1, x2, score_mask=score_mask, method=method,
                                  num_samples=num_samples)

    
        
    def eval_llr_Nvs1_book(self, x1, ids1, x2, score_mask=None, method='qscr',
                           num_samples=1):
        pass
        

        
    def get_config(self):
        config = {
            'qy_form': self.qy_form,
            'py_prior_form': self.py_prior_form }
        base_config = super(TiedVAE_qYqZ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.qy.h5' % (file_path)
        self.encoder_net.save(file_model)
        file_model = '%s.dec.h5' % (file_path)
        self.decoder_net.save(file_model)


        
    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=VAE.load_config_from_json(f.read())

        file_model = '%s.qy.h5' % (file_path)
        qy_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.dec.h5' % (file_path)
        decoder_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        k_args = ('px_cond_form', 'qy_form', 'py_prior_form', 'kl_min')
        args = {k: config[k] for k in k_args if k in config }
        return cls(qy_net, decoder_net, **args)


    
    @staticmethod
    def _get_loss_bernoulli(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
            
        logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
        if num_samples > 1:
            max_seq_length = K.cast(K.shape(x)[1], 'int32')
            r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z


    
    @staticmethod
    def _get_loss_diag_normal(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
            
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:]]
        logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
        if num_samples > 1:
            max_seq_length = K.cast(K.shape(x)[1], 'int32')
            r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z

    
    @staticmethod
    def _get_loss_normal(x, x_dec_param, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        seq_length=K.cast(K.shape(x)[-2], 'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:2*x_dim],
                       K.reshape(x_dec_param[:,:,2*x_dim:], (-1, seq_length, x_dim, x_dim))]
        logPx_g_z = hyp_obj.normal_3d(x, x_dec_param)
        if num_samples > 1:
            max_seq_length = K.cast(K.shape(x)[1], 'int32')
            r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z

    
    @staticmethod
    def _get_loss_normal_1chol(x, x_dec_param, x_chol, num_samples=1):
        if num_samples > 1:
            x = K.repeat_elements(x, num_samples, axis=0)
        x_dim=K.cast(K.shape(x)[-1], 'int32')
        x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:], x_chol]
        logPx_g_z = hyp_obj.normal_1chol_3d(x, x_dec_param)
        if num_samples > 1:
            max_seq_length = K.cast(K.shape(x)[1], 'int32')
            r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
            logPx_g_z = K.mean(r, axis=1)
            
        return logPx_g_z


        
    # @staticmethod
    # def _get_loss_bernoulli(x, x_dec_param, y_param, z_param, num_samples):

    #     seq_length=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
    #                             K.floatx()), axis=-1)
    #     if num_samples > 1:
    #         x = K.repeat_elements(x, num_samples, axis=0)

    #     logPx_g_z = hyp_obj.bernoulli(x, x_dec_param)
    #     if num_samples > 1:
    #         max_seq_length = K.cast(K.shape(x)[1], 'int32')
    #         r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
    #         logPx_g_z = K.mean(r, axis=1)

    #     kl_y = K.expand_dims(
    #         hyp_obj.kl_normal_vs_std_normal(y_param)/seq_length, dim=1)
    #     kl_z = hyp_obj.kl_normal_vs_std_normal(z_param)
        
    #     return logPx_g_z + kl_y + kl_z

    

    # @staticmethod
    # def _get_loss_normal(x, x_dec_param, y_param, z_param, num_samples):
    #     num_seq_samples=K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1),
    #                                K.floatx()), axis=-1)
    #     x_dim=K.cast(K.shape(x)[-1], 'int32')

    #     if num_samples > 1:
    #         x = K.repeat_elements(x, num_samples, axis=0)

    #     x_dec_param = [x_dec_param[:,:,:x_dim], x_dec_param[:,:,x_dim:]]
    #     logPx_g_z = hyp_obj.diag_normal(x, x_dec_param)
    #     if num_samples > 1:
    #         max_seq_length = K.cast(K.shape(x)[1], 'int32')
    #         r = K.reshape(logPx_g_z, (-1, num_samples, max_seq_length))
    #         logPx_g_z = K.mean(r, axis=1)

    #     kl_y = K.expand_dims(
    #         hyp_obj.kl_normal_vs_std_normal(y_param)/num_seq_samples, dim=1)
    #     kl_z = hyp_obj.kl_normal_vs_std_normal(z_param)
        
    #     return logPx_g_z + kl_y + kl_z
