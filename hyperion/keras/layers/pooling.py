
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras.layers.pooling import _GlobalPooling1D
from keras import activations, initializers, regularizers, constraints

from ...hyp_defs import float_keras
from .. import backend_addons as K2
from .. import constraints as hyp_constraints


class GlobalMaskedAveragePooling1D(_GlobalPooling1D):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

        
    def call(self, x, mask=None):
        return K.mean(x[mask.nonzeros(),:], axis=1)


    
class GlobalWeightedAveragePooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedAveragePooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    
    def call(self, xw, mask=None):
        x, weights = xw
        return K.mean(x*weights, axis=1)/K.mean(weights, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None



    
class GlobalWeightedSumPooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalWeightedSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    
    def call(self, xw, mask=None):
        x, weights = xw
        return K.sum(x*weights, axis=1)

    
    def compute_mask(self, inputs, mask=None):
        return None



    
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



    
class GlobalProdRenormDiagNormalCommonCovStdPrior(Layer):

    def __init__(self, **kwargs):
        super(GlobalProdRenormDiagNormalCommonCovStdPrior, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=2)]
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
