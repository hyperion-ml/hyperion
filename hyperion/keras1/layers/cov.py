
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer, Merge
from keras.layers.pooling import _GlobalPooling1D
from keras import activations, initializations, regularizers, constraints

from ...hyp_defs import float_keras
#from .. import backend_addons as K2
from .. import constraints as hyp_constraints

class MultConstDiagCov(Layer):
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MultConstDiagCov, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
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

        
    def call(self, x, mask=None):
        var = K.exp(self.logvar)
        mu  = x*var
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K.tile(self.logvar, tuple(tile_shape))
        return [mu, logvar]


    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        return [input_shape, input_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config()
                  if self.regularizer else None,
                  'constraint': self.constraint.get_config()
                  if self.constraint else None}
        
        base_config = super(MultConstDiagCov, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return [None, None]



    
class MultConstDiagCovStdPrior(Layer):
    
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = None #constraints.get('nonneg')

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MultConstDiagCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.A = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_A'.format(self.name),
                                   regularizer=self.regularizer,
                                   constraint=self.constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        
    def call(self, x, mask=None):
        logvar = - K.log(1+K.exp(self.A))
        #logvar = - K.log(1+self.A)
        var = K.exp(logvar)
        mu  = x*var
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K.tile(logvar, tuple(tile_shape))
        return [mu, logvar]


    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        return [input_shape, input_shape]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config()
                  if self.regularizer else None}
        
        base_config = super(MultConstDiagCovStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None]


    
class MultConstCovStdPrior(Layer):
    
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
        super(MultConstCovStdPrior, self).__init__(**kwargs)
        self.supports_masking = True


        
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

        
    def call(self, x, mask=None):
        logvar = - K.log(1+K.exp(self.D))
        #logvar = - K.log(1+self.A)
        var = K.exp(logvar)
        cov = K.dot(self.chol.T, var*self.chol)
        mu  = K.dot(x, cov)
        tile_shape = list(K.shape(mu))
        tile_shape[-1] = 1
        logvar = K.tile(logvar, tuple(tile_shape))
        tile_shape = tile_shape + [1]
        cholvar = K.tile(self.chol, tuple(tile_shape))
        return [mu, logvar, cholvar]


    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape_chol = tuple([self.input_dim]*(len(input_shape)+1))
        return [input_shape, input_shape, output_shape_chol]

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'D_regularizer': self.D_regularizer.get_config()
                  if self.D_regularizer else None,
                  'chol_regularizer': self.chol_regularizer.get_config()
                  if self.chol_regularizer else None}
        
        base_config = super(MultConstVarStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None]



#class DenseUppChol(Dense):

#     def __init__(self, cov_dim, diag_val=1, **kwargs):
#         output_dim = cov_dim**2
#         super(DenseUppChol, self).__init__(output_dim, **kwargs)
#         self.cov_dim = output_dim
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
