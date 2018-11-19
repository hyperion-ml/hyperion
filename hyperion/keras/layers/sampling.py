
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer


class ConvertNormalDiagCovPostStdPriorFmt(Layer):
    def __init__(self, in_fmt, out_fmt, min_var=0.0001, **kwargs):
        super(ConvertNormalDiagCovPostStdPriorFmt, self).__init__(**kwargs)
        self.in_fmt = in_fmt.split(sep='+', maxsplit=1)
        self.out_fmt = out_fmt.split(sep='+', maxsplit=1)
        self.min_var = min_var

        
    def compute_output_shape(self, input_shape):
        return input_shape


    def _logitvar_to_prec(self, logitvar):
        # p = (1+exp(-y))/(1+b exp(-y))
        logitvar = K.clip(logitvar, -7, 7)
        e = K.exp(-logitvar)
        return (1+e)/(1+self.min_var*e)

    
    def _logprec_minus_1_to_prec(self, logprec):
        # p = (1+exp(y))/(1+b exp(y))
        logprec = K.clip(logprec, -7, 7)
        e = K.exp(logprec)
        return (1+e)/(1+self.min_var*e)


    def _logvar_to_prec(self, logvar):
        return K.clip(K.exp(-logvar), 1, 1/self.min_var)


    def _logprec_to_prec(self, logprec):
        return K.clip(K.exp(logprec), 1, 1/self.min_var)


    def _var_to_prec(self, var):
        return K.clip(1/var, 1, 1/self.min_var)

    
    def _prec_to_prec(self, prec):
        return K.clip(prec, 1, 1/self.min_var)

    
    def _prec_minus_1_to_prec(self, prec):
        return K.clip(prec+1, 1, 1/self.min_var)


    def _compute_input_prec(self, p):
        if self.in_fmt[1] == 'logitvar':
            return self._logitvar_to_prec(p)
        if self.in_fmt[1] == 'logprec-1':
            return self._logprec_minus_1_to_prec(p)
        if self.in_fmt[1] == 'logvar':
            return self._logvar_to_prec(p)
        if self.in_fmt[1] == 'logprec':
            return self._logprec_to_prec(p)
        if self.in_fmt[1] == 'var':
            return self._var_to_prec(p)
        if self.in_fmt[1] == 'prec':
            return self._prec_to_prec(p)
        if self.in_fmt[1] == 'prec-1':
            return self._prec_minus_1_to_prec(p)
        
    
    def call(self, x, mask=None):
        p1, p2 = x

        prec = self._compute_input_prec(p2)
        if self.in_fmt[0] != self.out_fmt[0]:
            if self.in_fmt[0] == 'mean' :
                r1 = p1*prec
            else:
                r1 = p1/prec
        else:
            r1 = p1
            
        if self.out_fmt[1] == 'logvar':
            r2 = - K.log(prec)
        elif self.out_fmt[1] == 'var':
            r2 = 1/prec
        if self.out_fmt[1] == 'logprec':
            r2 = K.log(prec)
        elif self.out_fmt[1] == 'prec':
            r2 = prec

        return [r1, r2]


        
    def get_config(self):
        config = {'in_fmt': '+'.join(self.in_fmt),
                  'out_fmt':'+'.join(self.out_fmt),
                  'min_var': self.min_var}
        base_config = super(ConvertNormalDiagCovPostStdPriorFmt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

        

class Sampler(Layer):
    
    def __init__(self, num_samples=1, **kwargs):
        self.num_samples = num_samples
        super(Sampler, self).__init__(**kwargs)

        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[0] is not None:
            output_shape[0] *= self.num_samples
        return tuple(output_shape)

    
    def get_config(self):
        config = {'num_samples': self.num_samples}
        base_config = super(Sampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
class BernoulliSampler(Sampler):

    def call(self, p, mask=None):
        if self.num_samples > 0:
            p = K.repeat_elements(p, self.num_samples, axis=0)
        r = K.random_uniform(shape=K.shape(p))
        return r < p



class NormalDiagCovSampler(Sampler):

    def __init__(self, in_fmt = 'mean+logvar', num_samples=1, **kwargs):
        self.in_fmt = in_fmt.split(sep='+', maxsplit=1)
        super(NormalDiagCovSampler, self).__init__(num_samples, **kwargs)



    def _compute_stddev(self, p):
        if self.in_fmt[1] == 'stddev':
            return K.abs(p)
        if self.in_fmt[1] == 'var':
            return K.sqrt(K.abs(p))
        if self.in_fmt[1] == 'logvar':
            return K.exp(p/2)

        raise NotImplementedError('Format %s not implemented' % self.in_fmt[1])


            
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        if output_shape[0] is not None:
            output_shape[0] *= self.num_samples
        return tuple(output_shape)


            
    def call(self, p, mask=None):
        if self.num_samples > 1:
            p1_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            p2_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            p = [p1_rep, p2_rep]
            
        mu, p2 = p
        s = self._compute_stddev(p2)
        if self.in_fmt[0] == 'nat':
            mu = s*s*mu
            
        epsilon = K.random_normal(shape=K.shape(p[0]), mean=0., stddev=1.)        
        return mu + s * epsilon


    
    def get_config(self):
        config = {'in_fmt': '+'.join(self.in_fmt) }
        base_config = super(NormalDiagCovSampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    
class TiedNormalDiagCovSampler(NormalDiagCovSampler):

    def __init__(self, seq_length, in_fmt = 'mean+logvar', num_samples=1,
                 one_per_seq=False, **kwargs):
        super(TiedNormalDiagCovSampler, self).__init__(
            in_fmt, num_samples, **kwargs)
        self.seq_length = seq_length
        self.one_per_seq = one_per_seq


        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape.insert(1, self.seq_length)
        return tuple(output_shape)

    
    def call(self, p, mask=None):
        if self.num_samples > 1:
            p1_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            p2_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            p = [p1_rep, p2_rep]
        
        p_exp=[K.expand_dims(p[0], axis=1), K.expand_dims(p[1], axis=1)]

        mu, p2 = p_exp
        s = self._compute_stddev(p2)
        if self.in_fmt[0] == 'nat':
            mu = s*s*mu
        
        shape = K.shape(mu)
        if not self.one_per_seq:
            shape=(shape[0], self.seq_length, shape[2])
            
        epsilon = K.random_normal(shape=shape, mean=0., stddev=1.)
        x = mu + s * epsilon

        if self.one_per_seq:
            x = K.tile(x, (1, self.seq_length, 1))

        return x

        
    def get_config(self):
        config = {'seq_length': self.seq_length,
                  'one_per_seq': self.one_per_seq}
        base_config = super(TiedNormalDiagCovSampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

#######################################################
####### DEPRECATED FROM HERE ##########################
#######################################################
    
    
class DiagNormalSampler(Sampler):

    def __init__(self, var_spec = 'logvar', num_samples=1, **kwargs):
        self.var_spec = var_spec
        self._g = None
        super(DiagNormalSampler, self).__init__(num_samples, **kwargs)

    def build(self, input_shape):
        g_dict = {
            'logvar': self._g_logvar,
            'var': self._g_var,
            'std': self._g_std }
        self._g=g_dict[self.var_spec]
        super(DiagNormalSampler, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        if output_shape[0] is not None:
            output_shape[0] *= self.num_samples
        return tuple(output_shape)

    def call(self, p, mask=None):
        if self.num_samples > 1:
            mu_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            v_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            p = [mu_rep, v_rep]
        epsilon = K.random_normal(shape=K.shape(p[0]), mean=0., stddev=1.)
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

    def __init__(self, seq_length, var_spec = 'logvar', num_samples=1,
                 one_sample_per_seq=True, **kwargs):
        super(DiagNormalSamplerFromSeqLevel, self).__init__(
            var_spec, num_samples, **kwargs)
        self.seq_length = seq_length
        self.one_sample = one_sample_per_seq


    def build(self, input_shape):
        super(DiagNormalSamplerFromSeqLevel, self).build(input_shape)

        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape.insert(1, self.seq_length)
        return tuple(output_shape)

    
    def call(self, p, mask=None):
        if self.num_samples > 1:
            mu_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            v_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            p = [mu_rep, v_rep]
        
        p_exp=[K.expand_dims(p[0], axis=1), K.expand_dims(p[1], axis=1)]
        shape = K.shape(p_exp[0])
        if not self.one_sample:
            shape=(shape[0], self.seq_length, shape[2])
        epsilon = K.random_normal(shape=shape, mean=0., stddev=1.)
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

    

class NormalSampler(Sampler):

    def __init__(self, var_spec = 'logvar+chol', num_samples=1, **kwargs):
        self.var_spec = var_spec
        self._g = None
        super(NormalSampler, self).__init__(num_samples, **kwargs)

    def build(self, input_shape):
        g_dict = {
            'logvar+chol': self._g_logvar_chol,
            'var+chol': self._g_var_chol,
            'std+chol': self._g_std_chol }
        self._g=g_dict[self.var_spec]
        super(NormalSampler, self).build(input_shape)

        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        if output_shape[0] is not None:
            output_shape[0] *= self.num_samples
        return tuple(output_shape)

    
    def call(self, p, mask=None):
        if self.num_samples > 1:
            mu_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            v_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            chol_rep = K.repeat_elements(p[2], self.num_samples, axis=0)
            p = [mu_rep, v_rep, chol_rep]
        epsilon = K.random_normal(shape=K.shape(p[0]), mean=0., stddev=1.)
        return self._g(p, epsilon)

    
    def get_config(self):
        config = {'var_spec': self.var_spec}
        base_config = super(NormalSampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    @staticmethod
    def _g_logvar_chol1(p, epsilon):
        mu, logvar, chol = p
        if mu.ndim == 3:
            x_dim = K.cast(K.shape(epsilon)[-1], 'int32')
            seq_length = K.cast(K.shape(epsilon)[-2], 'int32')
            epsilon = K.reshape(epsilon, (-1, x_dim))
            chol = K.reshape(chol, (-1, x_dim, x_dim))
        epsilon = K.batch_dot(epsilon, chol, axes=(2, 1))
        if mu.ndim == 3:
            epsilon = K.reshape(epsilon, (-1, seq_length, x_dim))
        return mu + K.exp(logvar/2) * epsilon

    
    @staticmethod
    def _g_logvar_chol(p, epsilon):
        mu, logvar, chol = p
        epsilon = epsilon * K.exp(logvar/2)
        if mu.ndim == 3:
            x_dim = K.cast(K.shape(epsilon)[-1], 'int32')
            seq_length = K.cast(K.shape(epsilon)[-2], 'int32')
            epsilon = K.reshape(epsilon, (-1, x_dim))
            chol = K.reshape(chol, (-1, x_dim, x_dim))
        epsilon = K.batch_dot(epsilon, chol, axes=(1, 1))
        if mu.ndim == 3:
            epsilon = K.reshape(epsilon, (-1, seq_length, x_dim))
        return mu + epsilon

    
    @staticmethod
    def _g_var_chol(p, epsilon):
        mu, var, chol = p
        epsilon = K.batch_dot(epsilon, chol, axes=(1, 2))
        return mu + K.sqrt(K.abs(var)) * epsilon

    @staticmethod
    def _g_std_chol(p, epsilon):
        mu, s, chol = p
        epsilon = K.batch_dot(epsilon, chol, axes=(1, 2))
        return mu + K.abs(s) * epsilon



    
class NormalSamplerFromSeqLevel(NormalSampler):

    def __init__(self, seq_length, var_spec = 'logvar+chol', num_samples=1,
                 one_sample_per_seq=True, **kwargs):
        super(NormalSamplerFromSeqLevel, self).__init__(
            var_spec, num_samples, **kwargs)
        self.seq_length = seq_length
        self.one_sample = one_sample_per_seq


    def build(self, input_shape):
        super(NormalSamplerFromSeqLevel, self).build(input_shape)

        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape.insert(1, self.seq_length)
        return tuple(output_shape)

    
    def call(self, p, mask=None):
        if self.num_samples > 1:
            mu_rep = K.repeat_elements(p[0], self.num_samples, axis=0)
            v_rep = K.repeat_elements(p[1], self.num_samples, axis=0)
            chol_rep = K.repeat_elements(p[2], self.num_samples, axis=0)
            p = [mu_rep, v_rep, chol_rep]

        shape = K.shape(p[0])
        if self.one_sample:
            epsilon = K.random_normal(shape=shape, mean=0., stddev=1.)
            y = self._g_logvar_chol_2D(p, epsilon)
            y = K.expand_dims(y, axis=1)
            return K.tile(y, (1, self.seq_length, 1))

        p_exp=[K.expand_dims(p[0], axis=1), K.expand_dims(p[1], axis=1), p[2]]
        shape=(shape[0], self.seq_length, shape[1])
        epsilon = K.random_normal(shape=shape, mean=0., stddev=1.)
        return self._g_logvar_chol_3D(p_exp, epsilon)

    
    def get_config(self):
        config = {'seq_length': self.seq_length,
                  'one_sample': self.one_sample}
        base_config = super(NormalSamplerFromSeqLevel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @staticmethod
    def _g_logvar_chol_2D1(p, epsilon):
        mu, logvar, chol = p
        epsilon = K.batch_dot(epsilon, chol, axes=(1, 1))
        return mu + K.exp(logvar/2) * epsilon

    
    @staticmethod
    def _g_logvar_chol_3D1(p, epsilon):
        mu, logvar, chol = p
        epsilon = K.batch_dot(epsilon, chol, axes=(2, 1))
        return mu + K.exp(logvar/2) * epsilon

    @staticmethod
    def _g_logvar_chol_2D(p, epsilon):
        mu, logvar, chol = p
        epsilon = K.batch_dot(epsilon * K.exp(logvar/2), chol, axes=(1, 1))
        return mu + epsilon

    
    @staticmethod
    def _g_logvar_chol_3D(p, epsilon):
        mu, logvar, chol = p
        epsilon = K.batch_dot(epsilon * K.exp(logvar/2), chol, axes=(2, 1))
        return mu + epsilon
