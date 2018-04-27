
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer, Merge
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer

from .. import backend_addons as K2
from .. import constraints as hyp_constraints


class Bias(Layer):
    def __init__(self, output_dim,
                 activation=None, weights=None,
                 b_regularizer=None, activity_regularizer=None,
                 b_constraint=None, **kwargs):

        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = output_dim

        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Bias, self).__init__(**kwargs)

        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.b = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        
    def call(self, x, mask=None):
        output += self.b
        return self.activation(output)

    
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'activation': self.activation.__name__,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# class Constant(Layer):
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
#         super(Constant, self).__init__(**kwargs)

        
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#         self.input_dim = input_dim
#         self.input_spec = [InputSpec(dtype=K.floatx(),
#                                      ndim='2+')]

#         self.b = self.add_weight((self.output_dim,),
#                                  initializer='zero',
#                                  name='{}_b'.format(self.name),
#                                  regularizer=self.regularizer,
#                                  constraint=self.constraint)

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

#     def call(self, x, mask=None):
#         shape = list(K.shape(x))
#         shape[-1] = 1
#         return K.tile(self.b, tuple(shape))


#     def get_output_shape_for(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1] and input_shape[-1] == self.input_dim
#         output_shape = list(input_shape)
#         output_shape[-1] = self.output_dim
#         return tuple(output_shape)

#     def get_config(self):
#         config = {'output_dim': self.output_dim,
#                   'regularizer': self.regularizer.get_config() if self.regularizer else None,
#                   'constraint': self.constraint.get_config() if self.constraint else None}
#         base_config = super(Constant, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# class Constant2(Layer):
#     def __init__(self, output_dim,
#                  ndim=None,
#                  weights=None,
#                  regularizer=None, 
#                  constraint=None, **kwargs):

#         self.output_dim = output_dim
#         self.ndim = None
#         self.input_dim = None

#         self.regularizer = regularizers.get(regularizer)
#         self.constraint = constraints.get(constraint)

#         self.initial_weights = weights
#         self.input_spec = [InputSpec(ndim='2+')]

#         if self.input_dim:
#             kwargs['input_shape'] = (self.input_dim,)
#         super(Constant2, self).__init__(**kwargs)

        
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#         self.input_dim = input_dim
#         self.input_spec = [InputSpec(dtype=K.floatx(),
#                                      ndim='2+')]

#         self.b = self.add_weight((self.output_dim,),
#                                  initializer='zero',
#                                  name='{}_b'.format(self.name),
#                                  regularizer=self.regularizer,
#                                  constraint=self.constraint)

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

#     def call(self, x, mask=None):
#         shape = list(K.shape(x))
#         if self.ndim is not None:
#             if self.ndim == 1:
#                 return self.b
#             shape=shape[:self.ndim]
#         shape[-1] = 1
#         return K.tile(self.b, tuple(shape))


#     def get_output_shape_for(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1] and input_shape[-1] == self.input_dim
#         output_shape = list(input_shape)
#         if self.ndim is not None:
#             if self.ndim == 1:
#                 return (self.output_dim,)
#             output_shape = output_shape[:self.ndim]
#         output_shape[-1] = self.output_dim
#         return tuple(output_shape)

#     def get_config(self):
#         config = {'output_dim': self.output_dim,
#                   'regularizer': self.regularizer.get_config() if self.regularizer else None,
#                   'constraint': self.constraint.get_config() if self.constraint else None}
#         base_config = super(Constant2, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))



class Constant(Layer):
    def __init__(self, output_dim,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None

        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.initial_weights = weights
        self.input_spec = None

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Constant, self).__init__(**kwargs)

        
    def build(self, input_shape):
        # self.input_dim = 0
        # self.input_spec = [InputSpec(dtype=K.floatx(),
        #                              ndim=0)]
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.b = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        
    def call(self, x, mask=None):
        return self.b

    
    def get_output_shape_for(self, input_shape):
        return (self.output_dim,)

    
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(Constant, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TiledConstant(Constant):
        
    def build(self, input_shape):
        
        super(TiledConstant, self).build(input_shape)
        self.built = False
        
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.built = True

    
    def call(self, x, mask=None):
        shape = list(K.shape(x))
        shape[-1] = 1
        return K.tile(self.b, tuple(shape))

    
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)


class ConstTriu(Layer):
    def __init__(self, output_dim,
                 diag_val = None,
                 weights=None,
                 regularizer=None, 
                 constraint=None, **kwargs):

        self.output_dim = output_dim
        self.input_dim = None
        self.diag_val = diag_val

        self.regularizer = regularizers.get(regularizer)
        self.constraint = hyp_constraints.Triu(output_dim, diag_val)

        self.initial_weights = weights
        self.input_spec = None

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ConstTriu, self).__init__(**kwargs)

        
    def build(self, input_shape):
        # self.input_dim = 0
        # self.input_spec = [InputSpec(dtype=K.floatx(),
        #                              ndim=0)]
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.b = self.add_weight((self.output_dim, self.output_dim),
                                 initializer='identity',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.regularizer,
                                 constraint=self.constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        return self.b

    def get_output_shape_for(self, input_shape):
        return (self.output_dim, self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'diag_val': self.diag_val,
                  'regularizer': self.regularizer.get_config() if self.regularizer else None,
                  'constraint': self.constraint.get_config() if self.constraint else None}
        base_config = super(ConstTriu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TiledConstTriu(ConstTriu):
        
    def build(self, input_shape):
        
        super(TiledConstTriu, self).build(input_shape)
        self.built = False
        
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]
        self.built = True

    
    def call(self, x, mask=None):
        shape = list(K.shape(x))[:-1] + [1, 1]
        return K.tile(self.b, tuple(shape))

    
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)[:-1] + [self.output_dim]*2
        return tuple(output_shape)


class Invert(Layer):
    def call(self, x, mask=None):
        return 1/x

class Exp(Layer):
    def call(self, x, mask=None):
        return K.exp(x)


class ExpTaylor(Layer):
    def __init__(self, order=1):
        self.order = order
        
    def call(self, x, mask=None):
        y = x + 1
        f = 1
        xx = x
        for i in xrange(1, self.order):
            f = i*f
            xx = xx*x/f
            y = y + xx
        return y
    

    def get_config(self):
        config = {'order': self.order }
        base_config = super(ExpTaylor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Log1(Layer):
    def call(self, x, mask=None):
        return K.log(x+1)

class NegLog1(Layer):
    def call(self, x, mask=None):
        return -K.log(x+1)


class Repeat(Layer):
    
    def __init__(self, n, axis, **kwargs):
        self.n = n
        self.axis = axis
        super(Repeat, self).__init__(**kwargs)

        
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[0] is not None:
            output_shape[0] *= self.n
        return tuple(output_shape)


    def call(self, x, mask=None):
        return K.repeat_elements(x, self.n, axis=self.axis)

    
    def get_config(self):
        config = {'n': self.n,
                  'axis': self.axis }
        base_config = super(Repeat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
