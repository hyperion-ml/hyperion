
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import activations, initializers, regularizers, constraints

from .. import backend_addons as K2
from .. import constraints as hyp_constraints


class Bias(Layer):
    
    def __init__(self, units,
                 activation=None,
                 bias_initializer='zeros',
                 bias_regularizer=None, activity_regularizer=None,
                 bias_constraint=None, **kwargs):
        
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Bias, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.bias = self.add_weight((self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, inputs, mask=None):
        output = inputs + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    
    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
              }
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class Constant(Layer):
    def __init__(self, units, initializer='zeros', regularizer=None, 
                 constraint=None, **kwargs):
        
        super(Constant, self).__init__(**kwargs)
        self.units = units
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.const = self.add_weight((self.units,),
                                 initializer=self.initializer,
                                 name='const',
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)
        
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, inputs, mask=None):
        return self.const

    
    def compute_output_shape(self, input_shape):
        return (self.units,)

    
    def get_config(self):
        config = {'units': self.units,
                  'initializer': initializers.serialize(self.initializer),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'constraint': constraints.serialize(self.constraint)
              }
        base_config = super(Constant, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    
class TiledConstant(Constant):
        
    def build(self, input_shape):
        super(TiledConstant, self).build(input_shape)
    
    
    def call(self, inputs, mask=None):
        shape=list(K.shape(inputs))
        shape[-1] = 1
        return K2.tile(self.const, tuple(shape))

    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


    
class ConstTriu(Layer):
    def __init__(self, units, diag_val = None,
                 initializer='identity', regularizer=None, 
                 constraint=None, **kwargs):

        super(ConstTriu, self).__init__(**kwargs)
        self.units = units
        self.diag_val = diag_val
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = hyp_constraints.Triu(units, diag_val)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.const = self.add_weight((self.units, self.units),
                                 initializer=self.initializer,
                                 name='const',
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)
        
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, inputs, mask=None):
        return self.const

    
    def compute_output_shape(self, input_shape):
        return (self.units, self.units)

    
    def get_config(self):
        config = {'units': self.units,
                  'diag_val': self.diag_val,
                  'initializer': initializers.serialize(self.initializer),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'constraint': constraints.serialize(self.constraint)}
        base_config = super(ConstTriu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    
class TiledConstTriu(ConstTriu):
        
    def build(self, input_shape):
        super(TiledConstTriu, self).build(input_shape)
        
    
    def call(self, inputs, mask=None):
        shape = list(K.shape(inputs))[:-1] + [1, 1]
        return K2.tile(self.const, tuple(shape))

    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)[:-1] + [self.units]*2
        return tuple(output_shape)


    
class Invert(Layer):
    def call(self, inputs, mask=None):
        return 1/inputs

    
class Exp(Layer):
    def call(self, inputs, mask=None):
        return K.exp(inputs)


class ExpTaylor(Layer):
    def __init__(self, order=1):
        self.order = order

        
    def call(self, inputs, mask=None):
        y = inputs + 1
        f = 1
        xx = inputs
        for i in xrange(1, self.order):
            f = i*f
            xx = xx*inputs/f
            y = y + xx
        return y
    

    def get_config(self):
        config = {'order': self.order }
        base_config = super(ExpTaylor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Log(Layer):
    def call(self, inputs, mask=None):
        return K.log(inputs)

    
class Log1(Layer):
    def call(self, inputs, mask=None):
        return K.log(inputs+1)

class NegLog(Layer):
    def call(self, inputs, mask=None):
        return -K.log(inputs)
    
class NegLog1(Layer):
    def call(self, inputs, mask=None):
        return -K.log(inputs+1)
    

class NegSoftplus(Layer):
    def call(self, inputs, mask=None):
        return -K.log(1+K.exp(-inputs))

    
class Add1(Layer):
    def call(self, inputs, mask=None):
        return inputs+1


class Add01(Layer):
    def call(self, inputs, mask=None):
        return inputs+0.1

    
