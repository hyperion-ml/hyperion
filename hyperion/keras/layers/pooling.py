
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import logging

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras.layers.pooling import _GlobalPooling1D
from keras import activations, initializers, regularizers, constraints

from ...hyp_defs import float_keras
from .. import backend_addons as K2
from .. import constraints as hyp_constraints



class GlobalMaskedMaxPooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedMaxPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    
    def call(self, xw, mask=None):
        x, weights = xw
        return K2.max_with_mask(x, weights, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None


    

class GlobalMaskedAveragePooling1D(_GlobalPooling1D):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

        
    def call(self, x, mask=None):
        return K.mean(x[mask.nonzeros(),:], axis=1)


    
class GlobalWeightedAveragePooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedAveragePooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    
    def call(self, xw, mask=None):
        x, weights = xw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        return K.mean(x*weights, axis=1)/K.mean(weights, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None



    
class GlobalWeightedSumPooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    
    def call(self, xw, mask=None):
        x, weights = xw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        return K.sum(x*weights, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None


    

class GlobalWeightedMeanStdPooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedMeanStdPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xw, mask=None):
        x, weights = xw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)
        N = K.mean(weights, axis=1, keepdims=True)
        mu1 = K.mean(x*weights, axis=1, keepdims=True)/N
        s1 = K.sqrt(K.mean(((x-mu1)**2)*weights, axis=1, keepdims=True)/N)

        mu = mu1[:,0,:]
        s = s1[:,0,:]

        return [mu, s]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]




class GlobalWeightedMeanLogVarPooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedMeanLogVarPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xw, mask=None):
        x, weights = xw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        N = K.mean(weights, axis=1, keepdims=True)
        mu1 = K.mean(x*weights, axis=1, keepdims=True)/N
        logvar1 = K.log(K.mean(((x-mu1)**2)*weights, axis=1, keepdims=True)/N+1e-10)
        
        mu = mu1[:,0,:]
        logvar = logvar1[:,0,:]
        
        return [mu, logvar]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



    
class GlobalSumPooling1D(_GlobalPooling1D):

    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    
    def call(self, x, mask=None):
        return K.sum(x, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None



    
class GlobalSumWeights(_GlobalPooling1D):

    def __init__(self, **kwargs):
        super(GlobalSumWeights, self).__init__(**kwargs)
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    
    def call(self, x, mask=None):
        return K.sum(x, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None


    
class GlobalNormalDiagCovPostStdPriorPooling1D(Layer):

    def __init__(self, in_fmt='nat+logitvar', out_fmt='nat+var',
                 min_var=0.001, frame_corr_penalty=1, **kwargs):
        
        super(GlobalNormalDiagCovPostStdPriorPooling1D, self).__init__(**kwargs)
        self.in_fmt=in_fmt.split(sep='+', maxsplit=1)
        self.out_fmt=out_fmt.split(sep='+', maxsplit=1)
        self.min_var = min_var
        self.frame_corr_penalty = frame_corr_penalty
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]


    def _logitvar_to_prec_minus_1(self, logitvar):
        # p = (1+exp(-y))/(a+b exp(-y))
        # p-1 = (1-b)exp(-y)/(1+b exp(-y)) = (1-b)/(b+exp(y))
        
        return (1-self.min_var)/(self.min_var+K.exp(logitvar))


    def _logprec_minus_1_to_prec_minus_1(self, logprec):
        # p = (1+exp(y))/(1+b exp(y))
        # p-1 = (1-b)exp(-y)/(1+b exp(-y)) = (1-b)/(b+exp(-y))         
        
        return (1-self.min_var)/(self.min_var+K.exp(-logprec))


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
        p1, p2, weights = x
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        input_prec = self._compute_input_prec(p2)
        if self.in_fmt[0] == 'mean' :
            p1 = p1*input_prec
            
        eta = self.frame_corr_penalty * K.sum(p1*weights, axis=1)
        prec = 1 + self.frame_corr_penalty * K.sum((input_prec-1)*weights, axis=1)
        logging.debug(prec)
        logging.debug(self.out_fmt)
        prec = K.clip(prec, 1, 1e5)

        if self.out_fmt[0] == 'mean':
            r1 = eta/prec
        else:
            r1 = eta

        if self.out_fmt[1] == 'logvar':
            r2 = - K.log(prec)
        elif self.out_fmt[1] == 'var':
            r2 = 1/prec
        if self.out_fmt[1] == 'logprec':
            r2 = K.log(prec)
        elif self.out_fmt[1] == 'prec':
            r2 = prec

        return [r1, r2]
            

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    
    def get_config(self):
        config = { 'in_fmt': '+'.join(self.in_fmt),
                   'out_fmt': '+'.join(self.out_fmt),
                   'min_var': self.min_var,
                   'frame_corr_penalty': self.frame_corr_penalty}
        base_config = super(GlobalNormalDiagCovPostStdPriorPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



GlobalDiagNormalPostStdPriorPooling1D = GlobalNormalDiagCovPostStdPriorPooling1D


class LDE1D(Layer):

    def __init__(self, num_comp=64, order=2, **kwargs):
        super(LDE1D, self).__init__(**kwargs)
        self.num_comp = num_comp
        self.order = order
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True
        self.s = None
        self.mu = None

        
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], self.num_comp*input_shape[0][2])
        return output_shape


    def build(self, input_shape):
        assert len(input_shape[0]) >= 2
        input_dim = input_shape[0][-1]

        self.mu = self.add_weight(shape=(self.num_comp, input_dim),
                                  initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                  name='mu')
        
        self.s = self.add_weight(shape=(self.num_comp,),
                                 initializer='ones',
                                 name='s',
                                 constraint=constraints.non_neg())

    
    def call(self, xw, mask=None):
        x, weights = xw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        total_weights = []
        for k in xrange(self.num_comp):
            delta = K.bias_add(x, -self.mu[k])
            if self.order == 1:
                delta = K.sum(K.abs(delta), axis=-1, keepdims=True)
            else:
                delta = K.sum(delta*delta, axis=-1, keepdims=True)
            total_weights.append(-self.s[k] * delta)
        total_weights = K.concatenate(total_weights, axis=-1)
        total_weights = K.softmax(total_weights, axis=-1) * weights

        e = []
        for k in xrange(self.num_comp):
            w_k = K.expand_dims(total_weights[:,:,k], axis=-1)
            N_k = K.mean(w_k, axis=1, keepdims=True)
            e_k = K.mean(x*w_k, axis=1, keepdims=True)/N_k
            e_k = K.squeeze(e_k, axis=1)
            e_k = K.bias_add(e_k, -self.mu[k])
            e.append(e_k)
        e = K.concatenate(e, axis=-1)
        return e


    def get_config(self):
        config = {
            'num_comp': self.num_comp,
            'order': self.order,
        }
        base_config = super(LDE1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def compute_mask(self, inputs, mask=None):
        return None

####################################################################################
######### DEPRECATED FROM HERE #####################################################
####################################################################################

class GlobalProdRenormDiagNormalStdPrior(Layer):

    def __init__(self, **kwargs):
        super(GlobalProdRenormDiagNormalStdPrior, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xvw, mask=None):
        # input: mu_i/sigma2_i, log sigma2_i
        x, logvar_i, weights = xvw
        if K.ndim(weights) == 2:
            weights = K.expand_dims(weights, axis=-1)

        gamma = K.sum(x*weights, axis=1) 
        # N = K.sum(weights, axis=1)
        sum_prec_i = K.sum(K.exp(-logvar_i)*weights, axis=1)
        #prec = 1 + K.relu(sum_prec_i - N)
        prec = 1 + K.relu(sum_prec_i)
        mu  = gamma/prec
        logvar = - K.log(prec)
        return [mu, logvar]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



class GlobalProdRenormDiagNormalStdPrior2(Layer):

    def __init__(self, min_var=0.95, **kwargs):
        super(GlobalProdRenormDiagNormalStdPrior2, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True
        self.min_var = min_var

        
    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xvw, mask=None):
        # input: mu_i/sigma2_i, log sigma2_i
        x, y, weights = xvw
        gamma = K.sum(x*weights, axis=1)
        # p = (1+exp(y))/(1+b exp(y))                                                                                                                                                                 
        # p-1 = (1-b)exp(y)/(1+b exp(y)) = (1-b)/(b+exp(-y))
        
        pm1 = (1-self.min_var)/(self.min_var+K.exp(-y))
        sum_pm1 = K.sum(pm1*weights, axis=1)
        p_total = 1 + sum_pm1
        mu  = gamma/p_total
        logvar = - K.log(p_total)
        return [mu, logvar]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]

    
    def get_config(self):
        config = {'min_var': self.min_var }
        base_config = super(GlobalProdRenormDiagNormalStdPrior2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class GlobalProdRenormDiagNormalStdPrior3(Layer):

    def __init__(self, min_var=0.95, **kwargs):
        super(GlobalProdRenormDiagNormalStdPrior3, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True
        self.min_var = min_var

        
    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xvw, mask=None):
        # input: mu_i/sigma2_i, log sigma2_i
        x, y, weights = xvw
        mu = K.mean(x*weights, axis=1)
        # p = (1+exp(y))/(1+b exp(y))
        # p-1 = (1-b)exp(y)/(1+b exp(y)) = (1-b)/(b+exp(-y))
        
        pm1 = (1-self.min_var)/(self.min_var+K.exp(-y))
        sum_pm1 = K.sum(pm1*weights, axis=1)
        p_total = 1 + sum_pm1
        logvar = - K.log(p_total)
        return [mu, logvar]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]

    
    def get_config(self):
        config = {'min_var': self.min_var }
        base_config = super(GlobalProdRenormDiagNormalStdPrior3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    
class GlobalProdRenormDiagNormalCommonCovStdPrior(Layer):

    def __init__(self, **kwargs):
        super(GlobalProdRenormDiagNormalCommonCovStdPrior, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def call(self, xvw, mask=None):
        # input: mu_i/sigma2_i, log sigma2_i
        x, logvar_i, weights = xvw
        gamma = K.sum(x*weights,axis=1) 
        N = K.sum(weights, axis=1)
        prec_i = K.exp(-logvar_i)
        #var_i = K.exp(logvar_i[:,0,:])
        prec = 1 + N * (prec_i - 1)
        mu  = gamma/prec
        logvar = - K.log(prec)
        return [mu, logvar]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]


    
    
# class MultConstDiagCov(Layer):
    
#     def __init__(self, output_dim,
#                  weights=None,
#                  regularizer=None, 
#                  constraint=None, **kwargs):

#         self.output_dim = output_dim
#         self.input_dim = None

#         self.regularizer = regularizers.get(regularizer)
#         self.constraint = constraints.get(constraint)

#         self.initial_weights = weights
#         self.input_spec = [InputSpec(ndim='2+')]

#         if self.input_dim:
#             kwargs['input_shape'] = (self.input_dim,)
#         super(MultConstDiagCov, self).__init__(**kwargs)
#         self.supports_masking = True

        
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#         self.input_dim = input_dim
#         self.input_spec = [InputSpec(dtype=K.floatx(),
#                                      ndim='2+')]

#         self.logvar = self.add_weight((self.output_dim,),
#                                       initializer='zero',
#                                       name='{}_logvar'.format(self.name),
#                                       regularizer=self.regularizer,
#                                       constraint=self.constraint)

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

        
#     def call(self, x, mask=None):
#         var = K.exp(self.logvar)
#         mu  = x*var
#         tile_shape = list(K.shape(mu))
#         tile_shape[-1] = 1
#         logvar = K.tile(self.logvar, tuple(tile_shape))
#         return [mu, logvar]


#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1] and input_shape[-1] == self.input_dim
#         return [input_shape, input_shape]

    
#     def get_config(self):
#         config = {'output_dim': self.output_dim,
#                   'regularizer': self.regularizer.get_config() if self.regularizer else None,
#                   'constraint': self.constraint.get_config() if self.constraint else None}
#         base_config = super(MultConstDiagCov, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def compute_mask(self, inputs, mask=None):
#         return [None, None]



    
# class MultConstDiagCovStdPrior(Layer):
    
#     def __init__(self, output_dim,
#                  weights=None,
#                  regularizer=None, **kwargs):

#         self.output_dim = output_dim
#         self.input_dim = None

#         self.regularizer = regularizers.get(regularizer)
#         self.constraint = None #constraints.get('nonneg')

#         self.initial_weights = weights
#         self.input_spec = [InputSpec(ndim='2+')]

#         if self.input_dim:
#             kwargs['input_shape'] = (self.input_dim,)
#         super(MultConstDiagCovStdPrior, self).__init__(**kwargs)
#         self.supports_masking = True

        
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#         self.input_dim = input_dim
#         self.input_spec = [InputSpec(dtype=K.floatx(),
#                                      ndim='2+')]

#         self.A = self.add_weight((self.output_dim,),
#                                    initializer='zero',
#                                    name='{}_A'.format(self.name),
#                                    regularizer=self.regularizer,
#                                    constraint=self.constraint)

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

        
#     def call(self, x, mask=None):
#         logvar = - K.log(1+K.exp(self.A))
#         #logvar = - K.log(1+self.A)
#         var = K.exp(logvar)
#         mu  = x*var
#         tile_shape = list(K.shape(mu))
#         tile_shape[-1] = 1
#         logvar = K.tile(logvar, tuple(tile_shape))
#         return [mu, logvar]


#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1] and input_shape[-1] == self.input_dim
#         return [input_shape, input_shape]

    
#     def get_config(self):
#         config = {'output_dim': self.output_dim,
#                   'regularizer': self.regularizer.get_config() if self.regularizer else None}
#         base_config = super(MultConstDiagCovStdPrior, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

    
#     def compute_mask(self, inputs, mask=None):
#         return [None, None]


    
    
class GlobalProdRenormDiagNormalConstCovStdPrior(Layer):
    
    def __init__(self, units,
                 logvar_initializer='zeros',
                 logvar_regularizer=None, 
                 logvar_constraint=None, **kwargs):
        
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalProdRenormDiagNormalConstCovStdPrior, self).__init__(**kwargs)
        
        self.units = units
        self.logvar_initializer = initializers.get(logvar_initializer)
        self.logvar_regularizer = regularizers.get(logvar_regularizer)
        self.logvar_constraint = constraints.get(logvar_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True


        
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        input_dim = input_shape[0][-1]

        self.logvar = self.add_weight((self.units,),
                                      initializer=self.logvar_initializer,
                                      name='logvar',
                                      regularizer=self.logvar_regularizer,
                                      constraint=self.logvar_constraint)
        self.input_spec = [InputSpec(shape=(None, None, input_dim)), InputSpec(min_ndim=2)]
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        prec_1 = K.exp(-self.logvar)
        prec = 1 + N * (prec_1 - 1)
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'logvar_initializer': initializers.serialize(self.logvar_initializer),
                  'logvar_regularizer': regularizers.serialize(self.logvar_regularizer),
                  'logvar_constraint': constraints.serialize(self.logvar_constraint) }
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]




class GlobalProdRenormDiagNormalConstCovStdPrior2(Layer):
    
    def __init__(self, units,
                 prec_initializer='zeros',
                 prec_regularizer=None, 
                 prec_constraint=None, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalProdRenormDiagNormalConstCovStdPrior2, self).__init__(**kwargs)

        self.units = units
        self.prec_initializer = initializers.get(prec_initializer)
        self.prec_regularizer = regularizers.get(prec_regularizer)
        self.prec_constraint = constraints.get(prec_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True


        
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        input_dim = input_shape[0][-1]

        self.prec_1 = self.add_weight((self.units,),
                                      initializer=self.prec_initializer,
                                      name='prec',
                                      regularizer=self.prec_regularizer,
                                      constraint=self.prec_constraint)
        self.input_spec = [InputSpec(shape=(None, None, input_dim)), InputSpec(min_ndim=2)]
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        prec_1 = K.relu(self.prec_1)
        prec = 1 + N * prec_1
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'prec_initializer': initializers.serialize(self.prec_initializer),
                  'prec_regularizer': regularizers.serialize(self.prec_regularizer),
                  'prec_constraint': constraints.serialize(self.prec_constraint) }
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



    
class GlobalProdRenormDiagNormalConstCovStdPrior3(Layer):
    
    def __init__(self, units,
                 prec_initializer='normal',
                 bias_initializer='zeros',
                 prec_regularizer=None,
                 bias_regularizer=None, 
                 prec_constraint=None,
                 bias_constraint=None, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalProdRenormDiagNormalConstCovStdPrior3, self).__init__(**kwargs)
        
        self.units = units
        self.prec_initializer = initializers.get(prec_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.prec_regularizer = regularizers.get(prec_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.prec_constraint = constraints.get(prec_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True


        
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        input_dim = input_shape[0][-1]

        self.prec_1 = self.add_weight((self.units,),
                                      initializer=self.prec_initializer,
                                      name='prec',
                                      regularizer=self.prec_regularizer,
                                      constraint=self.prec_constraint)

        self.bias = self.add_weight((self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.input_spec = [InputSpec(shape=(None, None, input_dim)), InputSpec(min_ndim=2)]
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        prec = 1 + K.relu(N * self.prec_1 + self.bias)
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'prec_initializer': initializers.serialize(self.prec_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'prec_regularizer': regularizers.serialize(self.prec_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'prec_constraint': constraints.serialize(self.prec_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



class GlobalProdRenormDiagNormalConstCovStdPrior4(Layer):
    
    def __init__(self, units,
                 prec_initializer='zeros',
                 bias_initializer='zeros',
                 prec_regularizer=None,
                 bias_regularizer=None, 
                 prec_constraint=None,
                 bias_constraint=None, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalProdRenormDiagNormalConstCovStdPrior4, self).__init__(**kwargs)

        self.units = units
        self.prec_initializer = initializers.get(prec_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.prec_regularizer = regularizers.get(prec_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.prec_constraint = constraints.get(prec_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        input_dim = input_shape[0][-1]

        self.prec_1 = self.add_weight((self.units,),
                                      initializer=self.prec_initializer,
                                      name='prec',
                                      regularizer=self.prec_regularizer,
                                      constraint=self.prec_constraint)

        self.bias = self.add_weight((self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.input_spec = [InputSpec(shape=(None, None, input_dim)), InputSpec(min_ndim=2)]
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        #prec = 1 + K.relu(N * self.prec_1)
        #prec = 1 + N * K.exp(self.prec_1)
        prec = K.exp(self.bias) + N * K.exp(self.prec_1)
        #prec = 1 + N * self.prec_1 * self.prec_1
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'prec_initializer': initializers.serialize(self.prec_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'prec_regularizer': regularizers.serialize(self.prec_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'prec_constraint': constraints.serialize(self.prec_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]




class GlobalProdRenormNormalCommonCovStdPrior(Layer):

    def __init__(self,**kwargs):
        super(GlobalProdRenormDiagNormalCommonCovStdPrior, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=4), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape=(input_shape[0][0], input_shape[0][2])
        output_shape_chol=(input_shape[0][0], input_shape[0][2], input_shape[0][2])
        return [output_shape, output_shape, output_shape_chol]

    
    def call(self, xvw, mask=None):
        # input: mu_i/sigma2_i, log sigma2_i
        x, logvar_i, chol_i, weights = xvw
        gamma = K.sum(x*weights,axis=1) 
        N = K.sum(weights, axis=1)
        var_i = K.exp(logvar_i)
        cov_i = K.dot(chol_i.T,var_i*cholT)
        prec_i = K2.matrix_inverse(cov_i)
        #var_i = K.exp(logvar_i[:,0,:])
        I = K.eye(K.shape(x)[-1])
        prec = I + N * (prec_i - I)
        cov = K2.matrix_inverse(prec)

        mu  = K.dot(gamma,cov)

        var = K.diag(cov)
        chol = K2.cholesqy(cov/var)
        logvar = K.log(var)
        
        return [mu, logvar, chol]

    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None]



    
class GlobalProdRenormNormalConstCovStdPrior(Layer):

    def __init__(self, units,
                 D_initializer='zeros', chol_initializer='identity',
                 D_regularizer=None, chol_regularizer=None,
                 D_constraint=None, chol_constraint=None, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GlobalProdRenormNormalConstCovStdPrior, self).__init__(**kwargs)

        self.units = units
        self.D_initializer = initializers.get(D_initializer)
        self.chol_initializer = initializers.get(chol_initializer)
        self.D_regularizer = regularizers.get(D_regularizer)
        self.chol_regularizer = regularizers.get(chol_regularizer)
        self.D_constraint = constraints.get(D_constraint)
        self.chol_constraint = hyp_constraints.Triu(units, 1)
        # self.chol_constraint = constraints.get(chol_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(min_ndim=2)]
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        input_dim = input_shape[0][-1]

        self.D = self.add_weight((self.units,),
                                 initializer=self.D_initializer,
                                 name='D',
                                 regularizer=self.D_regularizer,
                                 constraint=self.D_constraint)

        self.chol = self.add_weight((self.units, self.units),
                                    initializer=self.chol_initializer,
                                    name='chol',
                                    regularizer=self.chol_regularizer,
                                    constraint=self.chol_constraint)

        self.input_spec = [InputSpec(shape=(None, None, input_dim)), InputSpec(min_ndim=2)]
        self.built = True


        
    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) == 3
        assert input_shape[0][-1]
        output_shape=(input_shape[0][0], input_shape[0][2])
        output_shape_chol=(input_shape[0][0], input_shape[0][2], input_shape[0][2])
        return [output_shape, output_shape, output_shape_chol]
    

    def call(self, xw, mask=None):
        # input: mu_i/sigma2_i
        x, weights = xw
        gamma = K.sum(x*weights,axis=1) 
        N = K.sum(weights, axis=1, keepdims=True)
        var_i = K.exp(self.D)
        cov_i = K.dot(self.chol.T*var_i,self.chol)
        #var_i = K.expand_dims(K.exp(self.D), axis=-1)
        #cov_i = K.dot(self.chol, var_i*self.chol.T)
        prec_i = K.expand_dims(K2.matrix_inverse(cov_i), axis=0)

        I = K.expand_dims(K.eye(self.units, dtype=float_keras()), axis=0)
        #prec = I + N * (prec_i - I)
        prec = I + N * prec_i

        fcov = lambda x: K2.matrix_inverse(x)
        cov = K.map_fn(fcov, prec)
        cov = 0.5*(cov + K.permute_dimensions(cov, [0, 2, 1]))

        mu  = K.batch_dot(gamma, cov)

        fchol = lambda x: K2.cholesky(x, lower=False)
        chol = K.map_fn(fchol, cov)

        fdiag = lambda x: K2.diag(x)
        sigma = K.map_fn(fdiag, chol)

        chol = chol/K.expand_dims(sigma, axis=-1)
        logvar = 2*K.log(sigma)
        
        return [mu, logvar, chol]


    def get_config(self):
        config = {'units': self.units,
                  'D_initializer': initializers.serialize(self.D_initializer),
                  'chol_initializer': initializers.serialize(self.chol_initializer),
                  'D_regularizer': regularizers.serialize(self.D_regularizer),
                  'chol_regularizer': regularizers.serialize(self.chol_regularizer),
                  'D_constraint': constraints.serialize(self.D_constraint)}
        #          'chol_constraint': constraints.serialize(self.chol_constraint)}
        
        base_config = super(GlobalProdRenormNormalConstCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None]
