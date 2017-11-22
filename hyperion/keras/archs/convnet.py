from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras.layers import Conv2D, Activation, Input, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, TimeDistributed, Reshape
from keras.models import Model

def ConvNetZhang17V1(output_units, input_shape,
                     num_conv_layers=10, num_fc_layers=3, 
                     conv_filters=[128]*4+[256]*6, hidden_fc_units=1024,
                     kernel_size=(5, 3), padding='same',
                     pool_size=(1, 3), 
                     output_activation=None, hidden_activation='relu',
                     use_batchnorm=True, conv_dropout_rate=0.3, fc_dropout_rate=0.3,
                     name='zhang17-v1',
                     kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None,
                     kernel_constraint=None, bias_constraint=None):
    """ Convnet used in
    Towards End-to-End Speech Recognition with Deep Convolutional Neural
    Networks
    https://arxiv.org/pdf/1701.02720.pdf
    """

    assert num_conv_layers >= 1, 'num_conv_layers (%d < 1)' % num_conv_layers
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size for i in xrange(num_conv_layers)]

    if not isinstance(conv_filters, list):
        conv_filters = [conv_filters for i in xrange(num_conv_layers)]

    input_dim=input_shape[-1]
    seq_length=input_shape[-2]
    x = Input(shape=input_shape)

    x2d = Reshape((-1, input_dim, 1), name='reshape-2d')(x)

    h_i = x2d
    for i in xrange(num_conv_layers):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)
        h_i = Conv2D(conv_filters[i], kernel_size=kernel_size[i],
                     padding=padding, name=('conv-%d' % i), 
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(h_i)
        
        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if i==0:
            h_i = MaxPooling2D(pool_size=pool_size, padding=padding)(h_i)
        
        if conv_dropout_rate > 0:
            h_i = Dropout(conv_dropout_rate)(h_i)

    h_i = Reshape((seq_length, -1), 'reshape-1d')(h_i)

    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = TimeDistributed(Dense(hidden_fc_units, name=('fc-%d' % i), 
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

    h_i = TimeDistributed(Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint))(h_i)

    if output_activation is not None:
        h_i = output_activation(h_i)

            
    return Model(x, h_i, name=name)




def TDConv2DNetV1(output_units, input_shape,
                  num_conv_layers=3, num_fc_layers=2, 
                  conv_filters=256, hidden_fc_units=512,
                  kernel_size=(5, 3), padding='same',
                  pool_size=(1, 3), num_poolings=3,
                  dilation_rate=1, dilation_rate_factor=1,
                  output_activation=None, hidden_activation='relu',
                  use_batchnorm=True, conv_dropout_rate=0.3, fc_dropout_rate=0.3,
                  name='tdconv2dnet-v1',
                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  kernel_regularizer=None, bias_regularizer=None,
                  kernel_constraint=None, bias_constraint=None):
    """ TDNN with 2D convolutions
    """

    assert num_conv_layers >= 1, 'num_conv_layers (%d < 1)' % num_conv_layers
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if isinstance(kernel_size, list):
        assert len(kernel_size) == num_conv_layers
    else:
        kernel_size = [kernel_size for i in xrange(num_conv_layers)]
    

    if isinstance(conv_filters, list):
        assert len(conv_filters) == num_conv_layers
    else:
        conv_filters = [conv_filters for i in xrange(num_conv_layers)]

    if isinstance(dilation_rate, list):
        assert len(dilation_rate) == num_conv_layers
        dilation_rate = [rate if isinstance(rate, tuple) else (rate, 1) for rate in dilation_rate]
    else:
        dilation_rate = [(dilation_rate_factor*i+dilation_rate, 1) for i in xrange(num_conv_layers)]


    input_dim=input_shape[-1]
    seq_length=input_shape[-2]
    x = Input(shape=input_shape)

    x2d = Reshape((-1, input_dim, 1), name='reshape-2d')(x)

    h_i = x2d
    for i in xrange(num_conv_layers):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)
        h_i = Conv2D(conv_filters[i], kernel_size=kernel_size[i],
                     dilation_rate=dilation_rate[i],
                     padding=padding, name=('conv-%d' % i), 
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(h_i)
        
        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if i < num_poolings:
            h_i = MaxPooling2D(pool_size=pool_size, padding=padding)(h_i)
        
        if conv_dropout_rate > 0:
            h_i = Dropout(conv_dropout_rate)(h_i)

    h_i = Reshape((seq_length, -1), name='reshape-1d')(h_i)

    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = TimeDistributed(Dense(hidden_fc_units, name=('fc-%d' % i), 
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

    h_i = TimeDistributed(Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                kernel_constraint=kernel_constraint,
                                bias_constraint=bias_constraint))(h_i)

    if output_activation is not None:
        h_i = output_activation(h_i)

            
    return Model(x, h_i, name=name)

