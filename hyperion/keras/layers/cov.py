
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


class MultConstDiagCov(Layer):
    
    def __init__(self, units,
                 logvar_initializer='zeros',
                 logvar_regularizer=None, 
                 logvar_constraint=None, **kwargs):

        super(MultConstDiagCov, self).__init__(**kwargs)
        self.units = units
        self.logvar_initializer = initializers.get(logvar_initializer)
        self.logvar_regularizer = regularizers.get(logvar_regularizer)
        self.logvar_constraint = constraints.get(logvar_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.logvar = self.add_weight((self.units,),
                                      initializer=self.logvar_initializer,
                                      name='logvar',
                                      regularizer=self.logvar_regularizer,
                                      constraint=self.logvar_constraint)

        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, x, mask=None):
        var = K.exp(self.logvar)
        mu  = x*var
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K2.tile(self.logvar, tuple(tile_shape))
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return [input_shape, input_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'logvar_initializer': initializers.serialize(self.logvar_initializer),
                  'logvar_regularizer': regularizers.serialize(self.logvar_regularizer),
                  'logvar_constraint': constraints.serialize(self.logvar_constraint) }
        
        base_config = super(MultConstDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]



    
class MultConstDiagCovStdPrior(Layer):
    
    def __init__(self, units,
                 prec_initializer='zeros',
                 prec_regularizer=None,
                 prec_constraint=None, **kwargs):

        super(MultConstDiagCovStdPrior, self).__init__(**kwargs)
        self.units = units
        self.prec_initializer = initializers.get(prec_initializer)
        self.prec_regularizer = regularizers.get(prec_regularizer)
        self.prec_constraint = constraints.get(prec_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.prec_1 = self.add_weight((self.units,),
                                      initializer=self.prec_initializer,
                                      name='prec',
                                      regularizer=self.prec_regularizer,
                                      constraint=self.prec_constraint)
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, x, mask=None):
        logvar = - K.log(1+K.exp(self.prec_1))
        #logvar = - K.log(1+self.prec_1)
        var = K.exp(logvar)
        mu  = x*var
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K2.tile(logvar, tuple(tile_shape))
        return [mu, logvar]


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return [input_shape, input_shape]

    
    def get_config(self):
        config = {'units': self.units,
                  'prec_initializer': initializers.serialize(self.prec_initializer),
                  'prec_regularizer': regularizers.serialize(self.prec_regularizer),
                  'prec_constraint': constraints.serialize(self.prec_constraint) }
        
        base_config = super(MultConstDiagCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]


    
class MultConstCovStdPrior(Layer):
    
    def __init__(self, units,
                 D_initializer='zeros', chol_initializer='identity',
                 D_regularizer=None, chol_regularizer=None,
                 D_constraint=None, chol_constraint=None, **kwargs):

        super(MultConstCovStdPrior, self).__init__(**kwargs)
        self.units = units
        self.D_initializer = initializers.get(D_initializer)
        self.chol_initializer = initializers.get(chol_initializer)
        self.D_regularizer = regularizers.get(D_regularizer)
        self.chol_regularizer = regularizers.get(chol_regularizer)
        self.D_constraint = constraints.get(D_constraint)
        self.chol_constraint = hyp_constraints.Triu(units, 1)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

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
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, x, mask=None):
        logvar = - K.log(1+K.exp(self.D))
        var = K.exp(logvar)
        cov = K.dot(self.chol.T, var*self.chol)
        mu  = K.dot(x, cov)
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K2.tile(logvar, tuple(tile_shape))
        tile_shape = tile_shape + [1]
        cholvar = K2.tile(self.chol, tuple(tile_shape))
        return [mu, logvar, cholvar]


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape_chol = tuple([self.input_dim]*(len(input_shape)+1))
        return [input_shape, input_shape, output_shape_chol]

    
    def get_config(self):
        config = {'units': self.units,
                  'D_initializer': initializers.serialize(self.D_initializer),
                  'chol_initializer': initializers.serialize(self.chol_initializer),
                  'D_regularizer': regularizers.serialize(self.D_regularizer),
                  'chol_regularizer': regularizers.serialize(self.chol_regularizer),
                  'D_constraint': constraints.serialize(self.D_constraint)}
        #          'chol_constraint': constraints.serialize(self.chol_constraint)}
        
        base_config = super(MultConstVarStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None]



#class DenseUppChol(Dense):

#     def __init__(self, cov_dim, diag_val=1, **kwargs):
#         units = cov_dim**2
#         super(DenseUppChol, self).__init__(units, **kwargs)
#         self.cov_dim = units
#         self.diag_val = diag_val
#         self.mask = None
#         self.diag = None

#     def build(self):
#         super(Dense, self).build()
#         mask = np.zeros((self.cov_dim, self.cov_dim), dtype=float_keras())
        

#     def get_config(self):
#         config = {'cov_dim': self.cov_dim,
#                   'diag_val': self.diag_val}
        
#         base_config = super(Dense, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
