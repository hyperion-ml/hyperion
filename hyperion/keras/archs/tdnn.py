from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras.layers import Conv1D, Activation, Input, Dense, Dropout, BatchNormalization, TimeDistributed
from keras.models import Model

#from ..layers.advanced_activations import *



def TDNNV1(num_td_layers, num_fc_layers,
           output_units, hidden_td_units, hidden_fc_units, input_units, kernel_size,
           dilation_rate=1, dilation_rate_factor=1, padding='valid',
           output_activation=None, hidden_activation='relu',
           use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
           name='tdnn-v1',
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None,
           kernel_constraint=None, bias_constraint=None):
    
    assert num_td_layers >= 1, 'num_td_layers (%d < 1)' % num_td_layers
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    

    hidden_activation=Activation(hidden_activation)

    if isinstance(kernel_size, list):
        assert num_td_layers == len(kernel_size)
    else:
        kernel_size = [kernel_size for i in xrange(num_td_layers)]

    if isinstance(dilation_rate, list):
         assert num_td_layers == len(dilation_rate)
    else:
        dilation_rate = [dilation_rate_factor*i+dilation_rate for i in xrange(num_td_layers)]

    if isinstance(hidden_fc_units, list):
        assert num_fc_layers == len(hidden_fc_units)
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]

    
    x = Input(shape=(None, input_units,))

    h_i = x
    for i in xrange(num_td_layers):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = Conv1D(hidden_td_units, kernel_size=kernel_size[i],
                     dilation_rate=dilation_rate[i],
                     padding=padding, name=('td-%d' % i), 
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if td_dropout_rate > 0:
            h_i = Dropout(td_dropout_rate)(h_i)


    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = TimeDistributed(Dense(hidden_fc_units[i], name=('fc-%d' % i), 
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint))(h_i)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if fc_dropout_rate > 0:
            h_i = Dropout(fc_dropout_rate)(h_i)


    if use_batchnorm:
        h_i = BatchNormalization()(h_i)

    if isinstance(output_units, list):
        if output_activation is None:
            output_activation = len(output_units)*[None]
        elif isinstance(output_activation, list):
            assert len(output_units) == len(output_activation)
        else:
            output_activation = len(output_units)*[Activation(output_activation)]
            
        y = []
        for i in xrange(len(output_units)):
             y_i = TimeDistributed(Dense(output_units[i], name=('fc-%d' % (num_fc_layers-1)),
                                         activation = output_activation[i],
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         bias_constraint=bias_constraint))(h_i)
             y.append(y_i)

    else:
        if output_activation is not None:
            output_activation=Activation(output_activation)
        y = TimeDistributed(Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
                                  activation = output_activation,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint))(h_i)
        
            
            
    return Model(x, y, name=name)
    
    




