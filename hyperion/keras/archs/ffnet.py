from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras.layers import Conv1D, Activation, Input, Dense, Dropout, BatchNormalization
from keras.models import Model

#from ..layers.advanced_activations import *



def FFNetV1(num_layers, 
            output_units, hidden_units, input_units,
            output_activation=None, hidden_activation='relu',
            use_batchnorm=True, dropout_rate=0,
            name='ffnn-v1',
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            kernel_constraint=None, bias_constraint=None):

    assert num_layers >= 1, 'num_layers (%d < 1)' % num_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if isinstance(hidden_units, list):
        assert num_layers-1 == len(hidden_units)
    else:
        hidden_units = [hidden_units for i in xrange(num_layers-1)]


    x = Input(shape=(input_units,))

    h_i = x
    for i in xrange(num_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = Dense(hidden_units[i], name=('fc-%d' % i), 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if dropout_rate > 0:
            h_i = Dropout(dropout_rate)(h_i)


    if use_batchnorm:
        h_i = BatchNormalization()(h_i)

    h_i = Dense(output_units, name=('fc-%d' % (num_layers-1)),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint)(h_i)

    if output_activation is not None:
        h_i = output_activation(h_i)

            
    return Model(x, h_i, name=name)
    
    
