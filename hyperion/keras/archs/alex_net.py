from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras.layers import Conv2D, Activation, Input, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Model

#from ..layers.advanced_activations import *


def AlexNetV1(output_units, input_shape=(224,224,3),
              num_conv_layers=5, num_fc_layers=3, num_colums=2,
              conv_filters=[48, 128, 192, 192, 128], hidden_fc_units=2048,
              kernel_size=[11, 5, 3, 3, 3], input_strides=4, padding='valid',
              pool_size=3, pool_strides=2,
              output_activation=None, hidden_activation='relu',
              use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
              name='alexnet-v1',
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None,
              kernel_constraint=None, bias_constraint=None):
    """ AlexNet as defined in 
    ImageNet Classification with Deep Convolutional Neural Networks
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """

    assert num_conv_layers >= 3, 'num_conv_layers (%d < 3)' % num_conv_layers
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size for i in xrange(num_conv_layers)]

    if not isinstance(conv_filters, list):
        conv_filters = [conv_filters for i in xrange(num_conv_layers)]


    x = Input(shape=input_shape)
    if use_batchnorm:
        x = BatchNormalization()(x)


    for j in xrange(num_columns):
        h_i = []
        for i in xrange(2):
            if i==0:
                h_ij = x
            if i>0 and use_batchnorm:
                h_ij = BatchNormalization()(h_ij)
            h_ij = Conv2D(conv_filters[i], kernel_size=kernel_size[i],
                          strides=input_strides,
                          padding=padding, name=('conv-%d' % i), 
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)(h_ij)
        
            if hidden_activation is not None:
                h_ij = hidden_activation(h_ij)

            h_ij = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=padding)(h_ij)
        
            if conv_dropout_rate > 0:
                h_ij = Dropout(conv_dropout_rate)(h_ij)
        
            h_i.append(h_ij)

    h = Concatenate(axis=-1)(h_i)
    if use_batchnorm:
        h = BatchNormalization()(h)


    for j in xrange(num_columns):
        h_i = []
        for i in xrange(2, num_conv_layers):
            if i==0:
                h_ij = h
            if i>0 and use_batchnorm:
                h_ij = BatchNormalization()(h_ij)
            h_ij = Conv2D(conv_filters[i], kernel_size=kernel_size[i],
                          strides=input_strides,
                          padding=padding, name=('conv-%d' % i), 
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)(h_ij)
        
            if hidden_activation is not None:
                h_ij = hidden_activation(h_ij)

            if i == num_conv_layers - 1:
                h_ij = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=padding)(h_ij)
        
            if conv_dropout_rate > 0:
                h_ij = Dropout(conv_dropout_rate)(h_ij)
        
            h_i.append(h_ij)

    h = Concatenate(axis=-1)(h_i)
    h = Flatten()(h)
    
    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h = BatchNormalization()(h)
        h_i = []
        for j in xrange(num_columns):
            h_ij = Dense(hidden_fc_units, name=('fc-%d' % i), 
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint)(h)

            if hidden_activation is not None:
                h_ij = hidden_activation(h_ij)

            if fc_dropout_rate > 0:
                h_ij = Dropout(fc_dropout_rate)(h_ij)

            h_i.append(h_ij)
        h = Concatenate(axis=-1)(h_i)

    if use_batchnorm:
        h = BatchNormalization()(h)

    y = Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer,
              kernel_constraint=kernel_constraint,
              bias_constraint=bias_constraint)(h)

    if output_activation is not None:
        y = output_activation(y)

            
    return Model(x, y, name=name)
    



