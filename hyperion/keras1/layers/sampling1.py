
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer, Merge


class Sampler(Layer):
    
    def __init__(self, nb_samples=1, **kwargs):
        self.nb_samples = nb_samples
        super(Sampler, self).__init__(**kwargs)

        
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        #output_shape[-2] *= self.nb_samples
        return tuple(output_shape)

    
    def get_config(self):
        config = {'nb_samples': self.nb_samples}
        base_config = super(Sampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class BernoulliSampler(Sampler):

    def call(self, p, mask=None):
        return K.random.binomial(p=p,shape=K.shape(p))


class DiagNormalSampler(Sampler):

    def __init__(self, var_spec = 'logvar', **kwargs):
        self.var_spec = var_spec
        self._g = None
        print('kk')
        super(DiagNormalSampler, self).__init__(**kwargs)

    def build(self, input_shape):
        g_dict = {
            'logvar': self._g_logvar,
            'var': self._g_var,
            'std': self._g_std }
        self._g=g_dict[self.var_spec]
        super(DiagNormalSampler, self).build(input_shape)
        
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape[0])
        #output_shape[-2] *= self.nb_samples
        return tuple(output_shape)

    def call(self, p, mask=None):
        epsilon = K.random_normal(shape=K.shape(p[0]), mean=0., std=1.)
        return self._g(p, epsilon)

    def get_config(self):
        config = {'var_spec': self.var_spec}
        base_config = super(DiagNormalSampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def _g_logvar(p, epsilon):
        mu, logvar = p
        return mu + K.exp(logvar/2) * epsilon

    @staticmethod
    def _g_var(p, epsilon):
        mu, var = p
        return mu + K.sqrt(K.abs(var)) * epsilon

    @staticmethod
    def _g_std(p, epsilon):
        mu, s = p
        return mu + K.abs(s) * epsilon


    
class DiagNormalSamplerFromSeqLevel(DiagNormalSampler):

    def __init__(self, seq_length, one_sample_per_seq=True, **kwargs):
        super(DiagNormalSamplerFromSeqLevel, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.one_sample = one_sample_per_seq


    def build(self, input_shape):
        super(DiagNormalSamplerFromSeqLevel, self).build(input_shape)
        
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape.insert(1, self.seq_length)
        return tuple(output_shape)

    def call(self, p, mask=None):
        # mu, logvar = p
        # mu=K.expand_dims(mu,dim=1)
        # logvar=K.expand_dims(logvar,dim=1)
        # epsilon = K.random_normal(shape=K.shape(mu), mean=0., std=1.)
        # y=mu + K.exp(logvar / 2) * epsilon
        # return K.tile(y, (1, self.seq_length, 1))

        p_exp=[K.expand_dims(p[0], dim=1), K.expand_dims(p[1], dim=1)]
        shape = K.shape(p_exp[0])
        if not self.one_sample:
            shape=(shape[0], self.seq_length, shape[2])
        epsilon = K.random_normal(shape=shape, mean=0., std=1.)
        if self.one_sample:
            y = self._g(p_exp, epsilon)
            return K.tile(y, (1, self.seq_length, 1))
        else:
            return self._g(p_exp, epsilon)

    def get_config(self):
        config = {'seq_length': self.seq_length,
                  'one_sample': self.one_sample}
        base_config = super(DiagNormalSamplerFromSeqLevel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
