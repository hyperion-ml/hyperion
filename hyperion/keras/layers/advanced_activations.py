from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer


class Swish(Layer):
    """ Self-gated activation function.

        f(x) = x \sigma(\beta x)

         # References
          Swish: a Self-Gated Activation Function https://arxiv.org/abs/1710.05941v1
    """

    def __init__(self, beta=1, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs * K.sigmoid(self.beta*inputs)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
class SwishBeta(Layer):
    """ Self-gated activation function with trainable beta

        f(x) = x \sigma(\beta x)

         # References
          Swish: a Self-Gated Activation Function https://arxiv.org/abs/1710.05941v1
    """

    def __init__(self, beta=1, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta_initializer = K.cast_to_floatx(beta)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.beta = self.add_weight((1,),
                                    initializer=self.beta_initializer,
                                    name='beta')
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.build = True

        
    def call(self, inputs):
        return inputs * K.sigmoid(self.beta*inputs)



class NIN(Layer):
    """ Network in network non-linearity as defined in 
    Acoustic modelling from the signal domain using CNNs, Interspeech 2016
    http://www.danielpovey.com/files/2016_interspeech_raw.pdf

    """

    def __init__(self, output_units, hidden_units, input_units,
                 activation='relu',
                 U1_initializer='glorot_uniform',
                 U2_initializer='glorot_uniform',
                 U1_regularizer=None,
                 U2_regularizer=None,
                 U1_constraint=None,
                 U2_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super(NIN, self).__init__(**kwargs)
            self.output_units = output_units
            self.hidden_units = hidden_units
            self.input_units = input_units
            self.activation = activations.get(activation)
            self.U1_initializer = initializers.get(U1_initializer)
            self.U2_initializer = initializers.get(U2_initializer)
            self.U1_regularizer = regularizers.get(U1_regularizer)
            self.U2_regularizer = regularizers.get(U2_regularizer)
            self.U1_constraint = constraints.get(U1_constraint)
            self.U2_constraint = constraints.get(U2_constraint)
            self.input_spec = InputSpec(min_ndim=2)
            self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        assert input_dim % self.input_units ==0, (
            'input_dim (%d) must be multiple of input_units (%d)' %
            (input_dim, self.input_units))
        
        self.num_micronets=int(input_dim/self.input_units)

        self.U1 = self.add_weight(shape=(self.input_units, self.hidden_units),
                                  initializer=self.U1_initializer,
                                  name='u1',
                                  regularizer=self.U1_regularizer,
                                  constraint=self.U1_constraint)
        self.U2 = self.add_weight(shape=(self.hidden_units, self.output_units),
                                  initializer=self.U2_initializer,
                                  name='u2',
                                  regularizer=self.U2_regularizer,
                                  constraint=self.U2_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, inputs):

        inputs= self.activation(inputs)
        output=[]
        for i in xrange(self.num_micronets):
            first = i*self.input_dim
            output_i = K.dot(self.activation(
                K.dot(self.inputs[:, first:first+self.input_units], self.U1)),
                             self.U2)
            output.append(output_i)
        output=K.cocatenate(tuple(output), axis=-1)
        return self.activation(output)

    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        
        assert input_shape[-1] % self.input_units ==0, (
            'input_dim (%d) must be multiple of input_units (%d)' %
            (input_shape[-1], self.input_units))
        
        num_micronets=int(input_shape[-1]/self.input_units)
        
        output_shape = list(input_shape)
        output_shape[-1] = self.output_units*num_micronets
        return tuple(output_shape)

    
    def get_config(self):
        config = {
            'input_units': self.input_units,
            'hidden_units': self.hidden_units,
            'output_units': self.output_units,
            'activation': activations.serialize(self.activation),
            'U1_initializer': initializers.serialize(self.U1_initializer),
            'U2_initializer': initializers.serialize(self.U2_initializer),
            'U1_regularizer': regularizers.serialize(self.U1_regularizer),
            'U2_regularizer': regularizers.serialize(self.U2_regularizer),
            'U1_constraint': constraints.serialize(self.U1_constraint),
            'U2_constraint': constraints.serialize(self.U2_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
