
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints

from ...hyp_defs import float_keras
from ..constraints import Clip

class CatQScoringDiagNormalPostStdPrior(Layer):

    def __init__(self, units, input_format='nat+prec',
                 q_class_format = 'mean+prec',
                 p1_initializer='glorot_uniform',
                 p2_initializer='ones', **kwargs):
        super(CatQScoringDiagNormalPostStdPrior, self).__init__(**kwargs)
        self.units = units
        self.input_format=input_format.split(sep='+', maxsplit=1)
        self.q_class_format=q_class_format.split(sep='+', maxsplit=1)
        self.p1_initializer = initializers.get(p1_initializer)
        self.p2_initializer = initializers.get(p2_initializer)
        
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape[0]) == 2
        input_dim = input_shape[0][-1]

        p2_constraint=Clip(min_val=1, max_val=1e5)

        self.p1 = self.add_weight((self.units, input_dim),
                                  initializer=self.p1_initializer,
                                  name='param-1')
        self.p2 = self.add_weight((self.units, input_dim),
                                  initializer=self.p2_initializer,
                                  constraint=p2_constraint,
                                  name='param-2')
        self.input_spec = [InputSpec(shape=(None, input_dim)), InputSpec(shape=(None, input_dim))] 
        self.built = True

        
    def call(self, inputs):
        p1_t, p2_t = inputs

        assert self.input_format == ['nat', 'prec']
        assert self.q_class_format == ['mean', 'prec']

        eta_e = self.p2 * self.p1 # (num_classes x dim)
        L_e = self.p2  # (num_classes x dim)
        eta_t = p1_t  # (batch x dim)
        L_t = p2_t  # (batch x dim)

        #### TODO
        L_et = L_t + Le - 1 # (batch x dim)

        C_et = 1/L_et        # (batch x dim)
        C_e = C_et - 1/L_e # (batch x dim)
        C_t = C_et - 1/L_t # (batch x dim)

        r_e = K.sum(K.log(L_e), axis=1, keepdims=True) + K.dot(C_e, K.transpose(eta_e*eta_e))          # (batch x num_classes)
        r_t = K.sum(K.log(L_t), axis=1, keepdims=True) + K.sum(C_t*eta_t**2, axis=1, keepdims=True)    # (batch x 1)
        r_et = -K.sum(K.log(L_et), axis=1, keepdims=True) + 2 * np.dot(eta_t*C_et, K.transpose(eta_e)) # (batch x num_classes)          
        logR = 0.5*(r_et + r_e + r_t)
        return logR


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)


    def compute_mask(self, inputs, mask=None):
        return None
    

    def get_config(self):
        config = {'units': self.units,
                  'input_format': '+'.join(self.input_format),
                  'q_class_format': '+'.join(self.q_class_format),
                  'p1_initializer': initializers.serialize(self.p1_initializer),
                  'p2_initializer': initializers.serialize(self.p2_initializer)}
        base_config =super(CatQScoringDiagNormalPostStdPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CatQScoringDiagNormalHomoPostStdPrior(CatQScoringDiagNormalPostStdPrior):

    def __init__(self, units, **kwargs):
        super(CatQScoringDiagNormalHomoPostStdPrior, self).__init__(units, **kwargs)

        
    def build(self, input_shape):
        assert len(input_shape[0]) == 2
        input_dim = input_shape[0][-1]
        p2_constraint=Clip(min_val=1, max_val=1e5)
        
        self.p1 = self.add_weight((self.units, input_dim),
                                   initializer=self.p1_initializer,
                                   name='param-1')
        self.p2 = self.add_weight((input_dim,),
                                  initializer=self.p2_initializer,
                                  constraint=p2_constraint,
                                  name='param-2')
        self.input_spec = [InputSpec(shape=(None, input_dim)), InputSpec(shape=(None, input_dim))] 
        self.built = True

        
    def call(self, inputs):
        p1_t, p2_t = inputs

        assert self.input_format == ['nat', 'prec']
        assert self.q_class_format == ['mean', 'prec']

        eta_e = self.p2 * self.p1 # (num_classes x dim)
        L_e = self.p2  # (1 x dim)
        eta_t = p1_t  # (batch x dim)
        L_t = p2_t  # (batch x dim)

        L_et = L_t + L_e - 1 # (batch x dim)

        C_et = 1/L_et      # (batch x dim)
        C_e = C_et - 1/L_e # (batch x dim)
        C_t = C_et - 1/L_t # (batch x dim)

        r_e = K.sum(K.log(L_e), axis=0, keepdims=True) + K.dot(eta_e*eta_e, K.transpose(C_e))    # (num_classes x batch)
        r_t = K.sum(K.log(L_t), axis=1, keepdims=True) + K.sum(C_t*eta_t**2, axis=1, keepdims=True)    # (batch x 1)
        r_et = -K.sum(K.log(L_et), axis=1, keepdims=True) + 2 * K.dot(eta_t*C_et, K.transpose(eta_e)) # (batch x num_classes)          
        logR = 0.5*(r_et + K.transpose(r_e) + r_t)
        return logR 
