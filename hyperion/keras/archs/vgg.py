from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras.layers import Conv2D, Activation, Input, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, Reshape, TimeDistributed
from keras.models import Model

#from ..layers.advanced_activations import *


def VGGNetV1(output_units, input_shape=(224,224,3),
             num_conv_blocks=5, num_block_layers=2*[2]+3*[3], num_fc_layers=3,
             conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
             hidden_fc_units=4096,
             kernel_size=3, padding='same',
             pool_size=2, pool_strides=2,
             output_activation=None, hidden_activation='relu',
             use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
             name='vgg-v1',
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             kernel_regularizer=None, bias_regularizer=None,
             kernel_constraint=None, bias_constraint=None):
    """ VGG net

    Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/abs/1409.1556)
    """
    assert num_conv_blocks >= 1, 'num_conv_blocks (%d < 5)' % num_conv_blocks
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if isinstance(conv_filters, list):
        assert len(conv_filters) == num_conv_blocks
    else:
        conv_filters = [min(max_conv_filters, conv_filters*conv_filters_factor**i) for i in xrange(num_conv_blocks)]

    if isinstance(num_block_layers, list):
        assert len(num_block_layers) == num_conv_blocks
    else:
        num_block_layers = [num_block_layers for i in xrange(num_conv_blocks)]
        
    if isinstance(hidden_fc_units, list):
        assert len(hidden_fc_units) == num_fc_layers-1
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers-1)]


    x = Input(shape=input_shape)

    for block in xrange(num_conv_blocks):
        for i in xrange(num_block_layers[block]):
            if use_batchnorm:
                h_i = BatchNormalization()(h_i)
            h_i = Conv2D(conv_filters[block], kernel_size=kernel_size,
                         padding=padding, name=('conv-%d-%d' % (block,i)), 
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)(h_i)
        
            if hidden_activation is not None:
                h_i = hidden_activation(h_i)

            if conv_dropout_rate > 0:
                h_i = Dropout(conv_dropout_rate)(h_i)

        h_i = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=padding)(h_i)
        

    h_i = Flatten()(h_i)

    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)
        h_i = Dense(hidden_fc_units[i], name=('fc-%d' % i), 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if fc_dropout_rate > 0:
            h_i = Dropout(fc_dropout_rate)(h_i)


    if use_batchnorm:
        h_i= BatchNormalization()(h_i)

    y = Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer,
              kernel_constraint=kernel_constraint,
              bias_constraint=bias_constraint)(h_i)

    if output_activation is not None:
        y = output_activation(y)

    return Model(x, y, name=name)

    
    
def VGG16(output_units, input_shape=(224,224,3),
          padding='same',
          output_activation='softmax', hidden_activation='relu',
          use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
          name='vgg16',
          kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None,
          kernel_constraint=None, bias_constraint=None):

    return VGGNetV1(output_units, input_shape,
                    num_conv_blocks=5, num_block_layers=2*[2]+3*[3], num_fc_layers=3,
                    conv_filters=64, conv_filters_factor=2,
                    hidden_fc_units=4096,
                    kernel_size=3, padding=padding,
                    output_activation=output_activation, hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm,
                    conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        


def VGG19(output_units, input_shape=(224,224,3),
          padding='same',
          output_activation='softmax', hidden_activation='relu',
          use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
          name='vgg19',
          kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None,
          kernel_constraint=None, bias_constraint=None):

    return VGGNetV1(output_units, input_shape,
                    num_conv_blocks=5, num_block_layers=2*[2]+3*[4], num_fc_layers=3,
                    conv_filters=64, conv_filters_factor=2,
                    hidden_fc_units=4096,
                    kernel_size=3, padding=padding,
                    output_activation=output_activation, hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm,
                    conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        



def TDVGGNetV1(output_units, input_shape,
               num_conv_blocks=5, num_block_layers=2*[2]+3*[3], num_fc_layers=3,
               conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
               dilation_rate=1, dilation_rate_factor=0,
               hidden_fc_units=4096,
               kernel_size=3, padding='same',
               pool_size=(1, 2), pool_strides=(1,2),
               output_activation=None, hidden_activation='relu',
               use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
               name='tdvgg-v1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               kernel_constraint=None, bias_constraint=None):
    """ Time delay version of VGG net 

    Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/abs/1409.1556)
    """
    assert num_conv_blocks >= 1, 'num_conv_blocks (%d < 5)' % num_conv_blocks
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if isinstance(conv_filters, list):
        assert len(conv_filters) == num_conv_blocks
    else:
        conv_filters = [min(max_conv_filters, conv_filters*conv_filters_factor**i) for i in xrange(num_conv_blocks)]

    if isinstance(num_block_layers, list):
        assert len(num_block_layers) == num_conv_blocks
    else:
        num_block_layers = [num_block_layers for i in xrange(num_block_layers)]
        
    if isinstance(hidden_fc_units, list):
        assert len(hidden_fc_units) == num_fc_layers-1
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers-1)]

    if isinstance(dilation_rate, list):
        assert len(dilation_rate) == num_conv_blocks
        dilation_rate = [rate if isinstance(rate, tuple) else (rate, 1) for rate in dilation_rate]
    else:
        dilation_rate = [(dilation_rate_factor*i+dilation_rate, 1) for i in xrange(num_conv_blocks)]

        
    input_dim=input_shape[-1]
    seq_length=input_shape[-2]
    x = Input(shape=input_shape)

    x2d = Reshape((-1, input_dim, 1), name='reshape-2d')(x)
    h_i = x2d

    for block in xrange(num_conv_blocks):
        for i in xrange(num_block_layers[block]):
            if use_batchnorm:
                h_i = BatchNormalization()(h_i)
            h_i = Conv2D(conv_filters[block], kernel_size=kernel_size,
                         dilation_rate=dilation_rate[block],
                         padding=padding, name=('conv-%d-%d' % (block,i)), 
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)(h_i)
        
            if hidden_activation is not None:
                h_i = hidden_activation(h_i)

            if conv_dropout_rate > 0:
                h_i = Dropout(conv_dropout_rate)(h_i)

        h_i = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=padding)(h_i)
        
    h_i = Reshape((seq_length, -1), name='reshape-1d')(h_i)

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
        h_i= BatchNormalization()(h_i)

    y = TimeDistributed(Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint))(h_i)

    if output_activation is not None:
        y = output_activation(y)

    return Model(x, y, name=name)

    
    
def TDVGG16(output_units, input_shape, conv_filters=64,
            dilation_rate=1, dilation_rate_factor=0,
            padding='same',
            output_activation='softmax', hidden_activation='relu',
            use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
            name='tdvgg16',
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            kernel_constraint=None, bias_constraint=None):
    
    return TDVGGNetV1(output_units, input_shape,
                      num_conv_blocks=5, num_block_layers=2*[2]+3*[3], num_fc_layers=3,
                      conv_filters=conv_filters, conv_filters_factor=2,
                      dilation_rate=dilation_rate, dilation_rate_factor=dilation_rate_factor,
                      hidden_fc_units=4096,
                      kernel_size=3, padding=padding,
                      output_activation=output_activation, hidden_activation=hidden_activation,
                      use_batchnorm=use_batchnorm,
                      conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate,
                      name=name,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        


def TDVGG19(output_units, input_shape, conv_filters=64,
            dilation_rate=1, dilation_rate_factor=0,
            padding='same',
            output_activation='softmax', hidden_activation='relu',
            use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
            name='tdvgg19',
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            kernel_constraint=None, bias_constraint=None):

    return TDVGGNetV1(output_units, input_shape,
                      num_conv_blocks=5, num_block_layers=2*[2]+3*[4], num_fc_layers=3,
                      conv_filters=conv_filters, conv_filters_factor=2,
                      dilation_rate=dilation_rate, dilation_rate_factor=dilation_rate_factor,
                      hidden_fc_units=4096,
                      kernel_size=3, padding=padding,
                      output_activation=output_activation, hidden_activation=hidden_activation,
                      use_batchnorm=use_batchnorm,
                      conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate,
                      name=name,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        

