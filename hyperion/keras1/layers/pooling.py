
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer, Merge
from keras.layers.pooling import _GlobalPooling1D
from keras import activations, initializations, regularizers, constraints

from ...hyp_defs import float_keras
from .. import backend_addons as K2
from .. import constraints as hyp_constraints


class GlobalMaskedAveragePooling1D(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalMaskedAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        return K.mean(x[mask.nonzeros(),:],axis=1)

class GlobalWeightedAveragePooling1D(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalWeightedAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def call(self, xw, mask=None):
        x, weights = xw
        return K.mean(x*weights,axis=1)/K.mean(weights,axis=1)
    
    def compute_mask(self, inputs, mask=None):
        return None


class GlobalWeightedSumPooling1D(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalWeightedSumPooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def call(self, xw, mask=None):
        x, weights = xw
        return K.sum(x*weights,axis=1)
    
    def compute_mask(self, inputs, mask=None):
        return None


class GlobalSumPooling1D(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        return K.sum(x, axis=1)
    
    def compute_mask(self, inputs, mask=None):
        return None



class GlobalSumWeights(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalSumWeights, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

    def call(self, x, mask=None):
        return K.sum(x,axis=1)
    
    def compute_mask(self, inputs, mask=None):
        return None


class GlobalProdRenormDiagNormalCommonCovStdPrior(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalProdRenormDiagNormalCommonCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
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


#     def get_output_shape_for(self, input_shape):
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


#     def get_output_shape_for(self, input_shape):
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
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalProdRenormDiagNormalConstCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) >= 2
        input_dim = input_shape[0][-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.logvar = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_logvar'.format(self.name),
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
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


    def get_output_shape_for(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) >= 2
        assert input_shape[0][-1] and input_shape[0][-1] == self.input_dim
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]




class GlobalProdRenormDiagNormalConstCovStdPrior2(Layer):
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalProdRenormDiagNormalConstCovStdPrior2, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) >= 2
        input_dim = input_shape[0][-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.prec_1 = self.add_weight((self.output_dim,),
                                    initializer='zero',
                                    name='{}_logvar'.format(self.name),
                                    regularizer=self.regularizer,
                                    constraint=self.constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
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


    def get_output_shape_for(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) >= 2
        assert input_shape[0][-1] and input_shape[0][-1] == self.input_dim
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



    

class GlobalProdRenormDiagNormalConstCovStdPrior3(Layer):
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalProdRenormDiagNormalConstCovStdPrior3, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) >= 2
        input_dim = input_shape[0][-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.prec_1 = self.add_weight((self.output_dim,),
                                    initializer='normal',
                                    name='{}_prec'.format(self.name),
                                    regularizer=self.regularizer,
                                    constraint=self.constraint)

        self.b = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        prec = 1 + K.relu(N * self.prec_1 + self.b)
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def get_output_shape_for(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) >= 2
        assert input_shape[0][-1] and input_shape[0][-1] == self.input_dim
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



class GlobalProdRenormDiagNormalConstCovStdPrior4(Layer):
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        #self.constraint = constraints.get('nonneg')

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalProdRenormDiagNormalConstCovStdPrior4, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) >= 2
        input_dim = input_shape[0][-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.prec_1 = self.add_weight((self.output_dim,),
                                    initializer='zero',
                                    name='{}_prec'.format(self.name),
                                    regularizer=self.regularizer,
                                    constraint=self.constraint)

        self.b = self.add_weight((self.output_dim,),
                                  initializer='zero',
                                  name='{}_b'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        
    def call(self, xw, mask=None):
        x, weights = xw
        gamma = K.sum(x*weights, axis=1) 
        N = K.sum(weights, axis=1)
        #prec = 1 + K.relu(N * self.prec_1)
        #prec = 1 + N * K.exp(self.prec_1)
        prec = K.exp(self.b) + N * K.exp(self.prec_1)
        #prec = 1 + N * self.prec_1 * self.prec_1
        logvar = - K.log(prec)
        mu  = gamma*K.exp(logvar)
        return [mu, logvar]


    def get_output_shape_for(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) >= 2
        assert input_shape[0][-1] and input_shape[0][-1] == self.input_dim
        output_shape = (input_shape[0][0], input_shape[0][2])
        return [output_shape, output_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(GlobalProdRenormDiagNormalConstCovStdPrior4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]




class GlobalProdRenormNormalCommonCovStdPrior(_GlobalPooling1D):

    def __init__(self,**kwargs):
        super(GlobalProdRenormDiagNormalCommonCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
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

    def __init__(self, output_dim, weights=None,
                 D_regularizer=None, chol_regularizer=None,
                 D_constraint=None, chol_constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.D_regularizer = regularizers.get(D_regularizer)
        self.chol_regularizer = regularizers.get(chol_regularizer)
        
        self.D_constraint = None #constraints.get('nonneg')
        self.chol_constraint = hyp_constraints.Triu(output_dim, 1)
        
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalProdRenormNormalConstCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True

        
    def get_output_shape_for(self, input_shape):
        output_shape=(input_shape[0][0], input_shape[0][2])
        output_shape_chol=(input_shape[0][0], input_shape[0][2], input_shape[0][2])
        return [output_shape, output_shape, output_shape_chol]

    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.D = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_D'.format(self.name),
                                 regularizer=self.D_regularizer,
                                 constraint=self.D_constraint)

        self.chol = self.add_weight((self.output_dim, self.output_dim),
                                    initializer='identity',
                                    name='{}_chol'.format(self.name),
                                    regularizer=self.chol_regularizer,
                                    constraint=self.chol_constraint)

        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True


    def call(self, xw, mask=None):
        # input: mu_i/sigma2_i
        x, weights = xw
        gamma = K.sum(x*weights,axis=1) 
        N = K.sum(weights, axis=1, keepdims=True)
        var_i = K.exp(self.D)
        cov_i = K.dot(self.chol.T*var_i,self.chol)
        #var_i = K.expand_dims(K.exp(self.D), dim=-1)
        #cov_i = K.dot(self.chol, var_i*self.chol.T)
        prec_i = K.expand_dims(K2.matrix_inverse(cov_i), dim=0)

        I = K.expand_dims(K.eye(self.output_dim, dtype=float_keras()), dim=0)
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

        chol = chol/K.expand_dims(sigma, dim=-1)
        logvar = 2*K.log(sigma)
        
        return [mu, logvar, chol]


    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'D_regularizer': self.D_regularizer.get_config()
                  if self.D_regularizer else None,
                  'D_constraint': self.D_constraint.get_config()
                  if self.D_constraint else None,
                  'chol_regularizer': self.chol_regularizer.get_config()
                  if self.chol_regularizer else None }
        
        base_config = super(GlobalProdRenormNormalConstCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None]
