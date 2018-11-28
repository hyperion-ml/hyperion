from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import logging
import numpy as np

from keras.layers import Conv2D, Activation, Input, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, GlobalAveragePooling2D, Reshape, Add, TimeDistributed
from keras.models import Model


def conv2d_res_block(x, filters, kernel_size, strides=1,
                     proj_shortcut=False,
                     activation=None,
                     use_batchnorm=True, dropout_rate=0,
                     padding='same', name=None,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None,
                     kernel_constraint=None, bias_constraint=None):

    if use_batchnorm:
        x = BatchNormalization()(x)

    h_i = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                 padding=padding, name=('%s-1' % (name)), 
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint)(x)
        
    if activation is not None:
        h_i = activation(h_i)

    if dropout_rate > 0:
        h_i = Dropout(dropout_rate)(h_i)

    if use_batchnorm:
        h_i = BatchNormalization()(h_i)
        
    h_i = Conv2D(filters, kernel_size=kernel_size,
                 padding=padding, name=('%s-2' % (name)), 
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint)(h_i)
        

    if proj_shortcut:
        x = Conv2D(filters, kernel_size=1, strides=strides,
                   padding=padding, name=('%s-s' % (name)), 
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint,
                   bias_constraint=bias_constraint)(x)
    
    h_i = Add()([h_i, x])

    if activation is not None:
        h_i = activation(h_i)

    if dropout_rate > 0:
        h_i = Dropout(dropout_rate)(h_i)

    return h_i



def conv2d_bn_res_block(x, filters1, filters2,
                        kernel_size, strides=1,
                        proj_shortcut=False,
                        activation=None,
                        use_batchnorm=True, dropout_rate=0,
                        padding='same', name=None,
                        kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=None, bias_regularizer=None,
                        kernel_constraint=None, bias_constraint=None):

    if use_batchnorm:
        x = BatchNormalization()(x)

    h_i = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                 padding=padding, name=('%s-1' % (name)), 
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint)(x)
        
    if activation is not None:
        h_i = activation(h_i)

    if dropout_rate > 0:
        h_i = Dropout(dropout_rate)(h_i)

    if use_batchnorm:
        h_i = BatchNormalization()(h_i)
        
    h_i = Conv2D(filters1, kernel_size=kernel_size,
                 padding=padding, name=('%s-2' % (name)), 
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint)(h_i)

    if activation is not None:
        h_i = activation(h_i)

    if dropout_rate > 0:
        h_i = Dropout(dropout_rate)(h_i)

    if use_batchnorm:
        h_i = BatchNormalization()(h_i)
        
    h_i = Conv2D(filters2, kernel_size=(1, 1),
                 padding=padding, name=('%s-3' % (name)), 
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint)(h_i)
    
    
    if proj_shortcut:
        x = Conv2D(filters2, kernel_size=(1,1), strides=strides,
                   padding=padding, name=('%s-s' % (name)), 
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint,
                   bias_constraint=bias_constraint)(x)
    
    h_i = Add()([h_i, x])
    
    if activation is not None:
        h_i = activation(h_i)

    if dropout_rate > 0:
        h_i = Dropout(dropout_rate)(h_i)

    return h_i


def ResNetV1(output_units, input_shape=(224,224,3),
             num_blocks=4, num_subblocks=[3, 4, 6, 3], num_fc_layers=1,
             conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
             hidden_fc_units=2048,
             kernel_size=3, padding='same',
             interblock_strides=2,
             input_kernel_size=7, input_strides=2,
             input_pool_size=(3, 3), input_pool_strides=2, 
             bottleneck_blocks=True,
             bottleneck_filter_factor=4, max_bottleneck_filters=2048,
             output_activation=None, hidden_activation='relu',
             use_batchnorm=True, conv_dropout_rate=0, fc_dropout_rate=0.5,
             is_sequence=False,
             name='resnet-v1',
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             kernel_regularizer=None, bias_regularizer=None,
             kernel_constraint=None, bias_constraint=None, return_context=False):
    
    """ ResNet
    Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    """
    assert num_blocks >= 2, 'num_blocks (%d < 2)' % num_blocks
    assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
    output_activation=Activation(output_activation)
    hidden_activation=Activation(hidden_activation)

    if isinstance(conv_filters, list):
        assert len(conv_filters) == num_blocks
    else:
        conv_filters = [min(max_conv_filters, conv_filters*conv_filters_factor**i) for i in xrange(num_blocks)]

    conv_filters_bn = [min(max_bottleneck_filters, bottleneck_filter_factor*f) for f in conv_filters]
        
    if isinstance(num_subblocks, list):
        assert len(num_subblocks) == num_blocks
    else:
        num_subblocks = [num_subblocks for i in xrange(num_blocks)]
        
    if isinstance(hidden_fc_units, list):
        assert len(hidden_fc_units) == num_fc_layers-1
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers-1)]

    context = int((input_kernel_size - 1)/2)

    logging.debug('Making ResNetV1 %s' % name)
    x = Input(shape=input_shape)
    if len(input_shape) == 2 and is_sequence:
        new_shape = tuple(list(input_shape)+[1])
        h_i = Reshape(new_shape)(x)
    else:
        h_i = x

    logging.debug('ResNetV1 %s input_shape =' % (name), h_i._keras_shape)
    
    if use_batchnorm:
        h_i = BatchNormalization()(h_i)
    h_i = Conv2D(conv_filters[0], kernel_size=input_kernel_size,
                 padding=padding, name='conv-0',
                 strides = input_strides,
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

    logging.debug('ResNetV1 %s conv2d shape =' % (name), h_i._keras_shape,
          'kernel =', input_kernel_size, 'stride =', input_strides,
          'context =', context)

    dec_ratio = x._keras_shape[1]/h_i._keras_shape[1]
    context += int(dec_ratio*(input_pool_size[0]-1)/2)
    
    h_i = MaxPooling2D(pool_size=input_pool_size, strides=input_pool_strides, padding=padding)(h_i)

    logging.debug('ResNetV1 %s maxpool2d shape =' % (name), h_i._keras_shape,
          'kernel =', input_pool_size, 'stride =', input_pool_strides,
          'context =', context)



    strides=1
    for block in xrange(num_blocks):
        if block > 0:
            strides = interblock_strides

        for i in xrange(num_subblocks[block]):
            proj_shortcut=False 

            dec_ratio = x._keras_shape[1]/h_i._keras_shape[1]
            context += int(dec_ratio*(kernel_size-1)/2)
            
            if bottleneck_blocks:
                if block>0 and i==0 and conv_filters_bn[block] != conv_filters_bn[block-1]:
                    proj_shortcut=True

                h_i = conv2d_bn_res_block(
                    h_i, conv_filters[block], conv_filters_bn[block],
                    kernel_size=kernel_size, strides=strides,
                    proj_shortcut=proj_shortcut,
                    activation=hidden_activation,
                    use_batchnorm=use_batchnorm, dropout_rate=conv_dropout_rate,
                    padding=padding, name=('conv-%d-%d' % (block+1,i)), 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)
            else:
                if block>0 and i==0 and conv_filters[block] != conv_filters[block-1]:
                    proj_shortcut=True
                h_i = conv2d_res_block(
                    h_i, conv_filters[block], 
                    kernel_size=kernel_size, strides=strides,
                    proj_shortcut=proj_shortcut,
                    activation=hidden_activation,
                    use_batchnorm=use_batchnorm, dropout_rate=conv_dropout_rate,
                    padding=padding, name=('conv-%d-%d' % (block+1,i)), 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)

            logging.debug('ResNetV1 %s conv_block %d-%d shape =' % (name,block,i), h_i._keras_shape,
                  'kernel =', kernel_size, 'stride =', strides,
                  'context =', context)

            strides = 1
        

    #h_i = GlobalAveragePooling2D()(h_i)
    if is_sequence:
        new_shape = (h_i._keras_shape[1], h_i._keras_shape[2]*h_i._keras_shape[3])
        h_i = Reshape(new_shape)(h_i)
    else:
        h_i = Flatten()(h_i)

    logging.debug('ResNetV1 %s reshape shape =' % (name), h_i._keras_shape,
          'context =', context)
        
    for i in xrange(num_fc_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)
            
        layer_i = Dense(hidden_fc_units[i], name=('fc-%d' % i), 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)

        if is_sequence:
            h_i = TimeDistributed(layer_i)(h_i)
        else:
            h_i = layer_i(h_i)
            

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if fc_dropout_rate > 0:
            h_i = Dropout(fc_dropout_rate)(h_i)
        logging.debug('ResNetV1 %s fc shape =' % (name), h_i._keras_shape,
              'context =', context)


    if use_batchnorm:
        h_i= BatchNormalization()(h_i)

    output_layer = Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)

    if is_sequence:
        y = TimeDistributed(output_layer)(h_i)
    else:
        y = output_layer(h_i)

    
    if output_activation is not None:
        y = output_activation(y)

    logging.debug('ResNetV1 %s output shape =' % (name), y._keras_shape,
          'context =', context)

    model = Model(x, y, name=name)
    if return_context:
        return model, context
    return model



def ResNet18V1(output_units, input_shape=(224,224,3),
               conv_filters=64, max_conv_filters=512,
               padding='same',
               output_activation=None, hidden_activation='relu',
               use_batchnorm=True, dropout_rate=0,
               is_sequence=False,
               name='resnet18-v1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               kernel_constraint=None, bias_constraint=None, return_context=False):

    return ResNetV1(output_units, input_shape=input_shape,
                    num_blocks=4, num_subblocks=2, num_fc_layers=1,
                    conv_filters=conv_filters, conv_filters_factor=2, max_conv_filters=max_conv_filters,
                    kernel_size=3, padding=padding,
                    interblock_strides=2,
                    input_kernel_size=7, input_strides=2,
                    input_pool_size=(3, 3), input_pool_strides=2, 
                    bottleneck_blocks=False,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm, conv_dropout_rate=dropout_rate,
                    is_sequence=is_sequence,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    return_context=return_context)



def ResNet34V1(output_units, input_shape=(224,224,3),
               conv_filters=64, max_conv_filters=512,
               padding='same',
               output_activation=None, hidden_activation='relu',
               use_batchnorm=True, dropout_rate=0, 
               name='resnet34-v1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               kernel_constraint=None, bias_constraint=None):

    return ResNetV1(output_units, input_shape=(224,224,3),
                    num_blocks=4, num_subblocks=[3,4,6,3], num_fc_layers=1,
                    conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
                    kernel_size=3, padding=padding,
                    interblock_strides=2,
                    input_kernel_size=7, input_strides=2,
                    input_pool_size=(3, 3), input_pool_strides=2, 
                    bottleneck_blocks=False,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm, conv_dropout_rate=dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)



def ResNet50V1(output_units, input_shape=(224,224,3),
               conv_filters=64, max_conv_filters=512,
               max_bottleneck_filters=2048,
               padding='same',
               output_activation=None, hidden_activation='relu',
               use_batchnorm=True, dropout_rate=0, 
               name='resnet50-v1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               kernel_constraint=None, bias_constraint=None):

    return ResNetV1(output_units, input_shape=(224,224,3),
                    num_blocks=4, num_subblocks=[3,4,6,3], num_fc_layers=1,
                    conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
                    kernel_size=3, padding=padding,
                    interblock_strides=2,
                    input_kernel_size=7, input_strides=2,
                    input_pool_size=(3, 3), input_pool_strides=2, 
                    bottleneck_blocks=True,
                    bottleneck_filter_factor=4,
                    max_bottleneck_filters=max_bottleneck_filters,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm, conv_dropout_rate=dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)



def ResNet101V1(output_units, input_shape=(224,224,3),
                conv_filters=64, max_conv_filters=512,
                max_bottleneck_filters=2048,
                padding='same',
                output_activation=None, hidden_activation='relu',
                use_batchnorm=True, dropout_rate=0, 
                name='resnet101-v1',
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                kernel_constraint=None, bias_constraint=None):

    return ResNetV1(output_units, input_shape=(224,224,3),
                    num_blocks=4, num_subblocks=[3,4,23,3], num_fc_layers=1,
                    conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
                    kernel_size=3, padding=padding,
                    interblock_strides=2,
                    input_kernel_size=7, input_strides=2,
                    input_pool_size=(3, 3), input_pool_strides=2, 
                    bottleneck_blocks=True,
                    bottleneck_filter_factor=4,
                    max_bottleneck_filters=max_bottleneck_filters,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm, conv_dropout_rate=dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)


def ResNet152V1(output_units, input_shape=(224,224,3),
                conv_filters=64, max_conv_filters=512,
                max_bottleneck_filters=2048,
                padding='same',
                output_activation=None, hidden_activation='relu',
                use_batchnorm=True, dropout_rate=0, 
                name='resnet152-v1',
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None,
                kernel_constraint=None, bias_constraint=None):

    return ResNetV1(output_units, input_shape=(224,224,3),
                    num_blocks=4, num_subblocks=[3,8,36,3], num_fc_layers=1,
                    conv_filters=64, conv_filters_factor=2, max_conv_filters=512,
                    kernel_size=3, padding=padding,
                    interblock_strides=2,
                    input_kernel_size=7, input_strides=2,
                    input_pool_size=(3, 3), input_pool_strides=2, 
                    bottleneck_blocks=True,
                    bottleneck_filter_factor=4,
                    max_bottleneck_filters=max_bottleneck_filters,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    use_batchnorm=use_batchnorm, conv_dropout_rate=dropout_rate,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)
