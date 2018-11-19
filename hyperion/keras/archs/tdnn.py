from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import keras.backend as K
from keras.layers import Conv1D, Activation, Input, Dense, Dropout, BatchNormalization, TimeDistributed, Add, Concatenate, SpatialDropout1D, Lambda, Conv2DTranspose
from keras.models import Model

#from ..layers.advanced_activations import *

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1,
                    padding='same', dilation_rate=1,
                    name=None, activation=None,
                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    kernel_constraint=None, bias_constraint=None):
    
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                            strides=(strides, 1), padding=padding,
                            dilation_rate=(dilation_rate, 1),
                            name=name, activation=activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x


# def TDNNV1(num_td_layers, num_fc_layers,
#            output_units, hidden_td_units, hidden_fc_units, input_units, kernel_size,
#            dilation_rate=1, dilation_rate_factor=1, padding='same',
#            output_activation=None, hidden_activation='relu',
#            use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
#            name='tdnn-v1',
#            kernel_initializer='glorot_uniform', bias_initializer='zeros',
#            kernel_regularizer=None, bias_regularizer=None,
#            kernel_constraint=None, bias_constraint=None, return_context=False):
    
#     assert num_td_layers >= 1, 'num_td_layers (%d < 1)' % num_td_layers
#     assert num_fc_layers >= 1, 'num_fc_layers (%d < 1)' % num_fc_layers
    
#     hidden_activation=Activation(hidden_activation)

#     if isinstance(kernel_size, list):
#         assert num_td_layers == len(kernel_size)
#     else:
#         kernel_size = [kernel_size for i in xrange(num_td_layers)]

#     if isinstance(dilation_rate, list):
#          assert num_td_layers == len(dilation_rate)
#     else:
#         dilation_rate = [dilation_rate_factor*i+dilation_rate for i in xrange(num_td_layers)]

#     if isinstance(hidden_fc_units, list):
#         assert num_fc_layers == len(hidden_fc_units) + 1
#     else:
#         hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers-1)]


#     x = Input(shape=(None, input_units,))

#     h_i = x
#     for i in xrange(num_td_layers):
#         if use_batchnorm:
#             h_i = BatchNormalization()(h_i)

#         h_i = Conv1D(hidden_td_units, kernel_size=kernel_size[i],
#                      dilation_rate=dilation_rate[i],
#                      padding=padding, name=('td-%d' % i), 
#                      kernel_initializer=kernel_initializer,
#                      bias_initializer=bias_initializer,
#                      kernel_regularizer=kernel_regularizer,
#                      bias_regularizer=bias_regularizer,
#                      kernel_constraint=kernel_constraint,
#                      bias_constraint=bias_constraint)(h_i)

#         if hidden_activation is not None:
#             h_i = hidden_activation(h_i)

#         if td_dropout_rate > 0:
#             h_i = Dropout(td_dropout_rate)(h_i)


#     for i in xrange(num_fc_layers-1):
#         if use_batchnorm:
#             h_i = BatchNormalization()(h_i)

#         h_i = TimeDistributed(Dense(hidden_fc_units[i], name=('fc-%d' % i), 
#                                     kernel_initializer=kernel_initializer,
#                                     bias_initializer=bias_initializer,
#                                     kernel_regularizer=kernel_regularizer,
#                                     bias_regularizer=bias_regularizer,
#                                     kernel_constraint=kernel_constraint,
#                                     bias_constraint=bias_constraint))(h_i)

#         if hidden_activation is not None:
#             h_i = hidden_activation(h_i)

#         if fc_dropout_rate > 0:
#             h_i = Dropout(fc_dropout_rate)(h_i)


#     if use_batchnorm:
#         h_i = BatchNormalization()(h_i)

#     if isinstance(output_units, list):
#         if output_activation is None:
#             output_activation = len(output_units)*[None]
#         elif isinstance(output_activation, list):
#             assert len(output_units) == len(output_activation)
#         else:
#             output_activation = len(output_units)*[Activation(output_activation)]
            
#         y = []
#         for i in xrange(len(output_units)):
#              y_i = TimeDistributed(Dense(output_units[i], name=('fc-%d-%d' % (num_fc_layers-1,i)),
#                                          activation = output_activation[i],
#                                          kernel_initializer=kernel_initializer,
#                                          bias_initializer=bias_initializer,
#                                          kernel_regularizer=kernel_regularizer,
#                                          bias_regularizer=bias_regularizer,
#                                          kernel_constraint=kernel_constraint,
#                                          bias_constraint=bias_constraint))(h_i)
#              y.append(y_i)

#     else:
#         if output_activation is not None:
#             output_activation=Activation(output_activation)
#         y = TimeDistributed(Dense(output_units, name=('fc-%d' % (num_fc_layers-1)),
#                                   activation = output_activation,
#                                   kernel_initializer=kernel_initializer,
#                                   bias_initializer=bias_initializer,
#                                   kernel_regularizer=kernel_regularizer,
#                                   bias_regularizer=bias_regularizer,
#                                   kernel_constraint=kernel_constraint,
#                                   bias_constraint=bias_constraint))(h_i)
        
#     model = Model(x, y, name=name)
#     if return_context:
#         context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
#         return model, context
#     return model


def TDNNV1(num_td_layers, num_fc_layers,
           output_units, hidden_td_units, hidden_fc_units, input_units, kernel_size,
           dilation_rate=1, dilation_rate_factor=1, padding='same',
           output_activation=None, hidden_activation='relu',
           use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
           spatial_dropout=True,
           name='tdnn-v1',
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None,
           kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
        assert num_fc_layers == len(hidden_fc_units) + 1
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers-1)]


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
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
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
             y_i = TimeDistributed(Dense(output_units[i], name=('fc-%d-%d' % (num_fc_layers-1,i)),
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
        
    model = Model(x, y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model



TDNNV2 = TDNNV1    



def ResTDNNV1(num_td_blocks, num_fc_blocks,
              output_units, hidden_td_units, hidden_fc_units, input_units, kernel_size,
              dilation_rate=1, dilation_rate_factor=1, padding='same',
              output_activation=None, hidden_activation='relu',
              use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
              spatial_dropout = True,
              name='res-tdnn-v1',
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None,
              kernel_constraint=None, bias_constraint=None, return_context=False):
    
    assert num_td_blocks >= 1, 'num_td_layers (%d < 1)' % num_td_blocks
    num_td_layers = num_td_blocks*2
    num_fc_layers = num_fc_blocks*2+1

    hidden_activation=Activation(hidden_activation)

    if isinstance(kernel_size, list):
        assert num_td_layers == len(kernel_size)
    else:
        kernel_size = [kernel_size for i in xrange(num_td_layers)]

    if isinstance(dilation_rate, list):
         assert num_td_layers == len(dilation_rate)
    else:
        dilation_rate = [dilation_rate_factor*i+dilation_rate for i in xrange(num_td_layers)]

    # if isinstance(hidden_fc_units, list):
    #     assert num_fc_layers == len(hidden_fc_units)
    # else:
    #     hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]


    x = Input(shape=(None, input_units,))

    h_i = x
    for i in xrange(num_td_blocks):
        j = 2*i
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        if i==0: 
            h_res_i = Conv1D(hidden_td_units, kernel_size=1, name=('res-td-%d' % i),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             bias_constraint=bias_constraint)(h_i)
        else:
            h_res_i = h_i

        h_i_1 = Conv1D(hidden_td_units, kernel_size=kernel_size[j],
                       dilation_rate=dilation_rate[j],
                       padding=padding, name=('td-%d-1' % i), 
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i_1 = hidden_activation(h_i_1)

        if td_dropout_rate > 0:
            if spatial_dropout:
                h_i_1 = SpatialDropout1D(td_dropout_rate)(h_i_1)
            else:
                h_i_1 = Dropout(td_dropout_rate)(h_i_1)

        h_i_2 = Conv1D(hidden_td_units, kernel_size=kernel_size[j+1],
                       dilation_rate=dilation_rate[j+1],
                       padding=padding, name=('td-%d-2' % i), 
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i_2 = hidden_activation(h_i_2)

        if td_dropout_rate > 0:
            if spatial_dropout:
                h_i_2 = SpatialDropout1D(td_dropout_rate)(h_i_2)
            else:
                h_i_2 = Dropout(td_dropout_rate)(h_i_2)

        h_i = Add()([h_i_2, h_res_i])



    for i in xrange(num_fc_blocks):
        j = 2*i
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        if i==0 and hidden_td_units != hidden_fc_units: 
            h_res_i = Conv1D(hidden_fc_units, kernel_size=1, name=('res-fc-%d' % i),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             bias_constraint=bias_constraint)(h_i)
        else:
            h_res_i = h_i

        h_i_1 = Conv1D(hidden_fc_units, kernel_size=1,
                       padding=padding, name=('fc-%d-1' % i), 
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i_1 = hidden_activation(h_i_1)

        if fc_dropout_rate > 0:
            if spatial_dropout:
                h_i_1 = SpatialDropout1D(fc_dropout_rate)(h_i_1)
            else:
                h_i_1 = Dropout(fc_dropout_rate)(h_i_1)

        h_i_2 = Conv1D(hidden_fc_units, kernel_size=1,
                       padding=padding, name=('fc-%d-2' % i), 
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint)(h_i)

        if hidden_activation is not None:
            h_i_2 = hidden_activation(h_i_2)

        if fc_dropout_rate > 0:
            if spatial_dropout:
                h_i_2 = SpatialDropout1D(fc_dropout_rate)(h_i_2)
            else:
                h_i_2 = Dropout(fc_dropout_rate)(h_i_2)

        h_i = Add()([h_i_2, h_res_i])


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
             y_i = TimeDistributed(Dense(output_units[i], name=('fc-%d-%d' % (num_fc_layers-1,i)),
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
        
    model = Model(x, y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model
    
    




def TDNNV1WithEmbedInputV1(num_td_layers, num_fc_layers,
                           output_units, hidden_td_units, hidden_fc_units, input_units,
                           embed_units, kernel_size,
                           dilation_rate=1, dilation_rate_factor=1, padding='same',
                           output_activation=None, hidden_activation='relu',
                           use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
                           spatial_dropout = True,
                           cat_embed_to_all_fc_layers=False,
                           name='tdnn-v1-e-v1',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros',
                           kernel_regularizer=None, bias_regularizer=None,
                           kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
    e = Input(shape=(None, embed_units,))

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
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
                h_i = Dropout(td_dropout_rate)(h_i)

            
    for i in xrange(num_fc_layers-1):
        if i==0 or cat_embed_to_all_fc_layers:
            h_i = Concatenate(axis=-1)([h_i, e])
        
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
                h_i = Dropout(fc_dropout_rate)(h_i)

                
    if num_fc_layers == 1:
        h_i = Concatenate(axis=-1)([h_i, e])
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
             y_i = TimeDistributed(Dense(output_units[i], name=('fc-%d-%d' % (num_fc_layers-1,i)),
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
        
    model = Model([x, e], y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model



def TDNNV1Transpose(num_fc_layers, num_td_layers,
                    output_units, hidden_fc_units, hidden_td_units, input_units, kernel_size,
                    dilation_rate=1, dilation_rate_factor=1, padding='same',
                    output_activation=None, hidden_activation='relu',
                    use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
                    spatial_dropout = True,
                    name='t-tdnn-v1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
        dilation_rate = [dilation_rate_factor*(num_td_layers - i)+dilation_rate for i in xrange(num_td_layers)]

    if isinstance(hidden_fc_units, list):
        assert num_fc_layers == len(hidden_fc_units)
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]


    x = Input(shape=(None, input_units,))

    h_i = x
    for i in xrange(num_fc_layers):
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
                h_i = Dropout(fc_dropout_rate)(h_i)

            
    for i in xrange(num_td_layers-1):
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
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
                h_i = Dropout(td_dropout_rate)(h_i)


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
             y_i = Conv1D(output_units[i], kernel_size=kernel_size[-1],
                          dilation_rate=dilation_rate[-1],
                          padding=padding,
                          name=('td-%d-%d' % (num_td_layers-1,i)),
                          activation = output_activation[i],
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)(h_i)
             y.append(y_i)

    else:
        if output_activation is not None:
            output_activation=Activation(output_activation)
        y = Conv1D(output_units, kernel_size=kernel_size[-1],
                   dilation_rate=dilation_rate[-1],
                   padding=padding,
                   name=('td-%d' % (num_td_layers-1)),
                   activation = output_activation,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint,
                   bias_constraint=bias_constraint)(h_i)
        
    model = Model(x, y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model



def TDNNV2Transpose(num_fc_layers, num_td_layers,
                    output_units, hidden_fc_units, hidden_td_units, input_units, kernel_size,
                    dilation_rate=1, dilation_rate_factor=1, padding='same',
                    output_activation=None, hidden_activation='relu',
                    use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
                    spatial_dropout = True,
                    name='t-tdnn-v2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
        dilation_rate = [dilation_rate_factor*(num_td_layers - i)+dilation_rate for i in xrange(num_td_layers)]

    if isinstance(hidden_fc_units, list):
        assert num_fc_layers == len(hidden_fc_units)
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]


    x = Input(shape=(None, input_units,))

    h_i = x
    for i in xrange(num_fc_layers):
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
                h_i = Dropout(fc_dropout_rate)(h_i)

            
    for i in xrange(num_td_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = Conv1DTranspose(h_i, hidden_td_units, kernel_size=kernel_size[i],
                              dilation_rate=dilation_rate[i],
                              padding=padding, name=('td-%d' % i), 
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if td_dropout_rate > 0:
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
                h_i = Dropout(td_dropout_rate)(h_i)


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
             y_i = Conv1DTranspose(h_i, output_units[i], kernel_size=kernel_size[-1],
                                   dilation_rate=dilation_rate[-1],
                                   padding=padding,
                                   name=('td-%d-%d' % (num_td_layers-1,i)),
                                   activation = output_activation[i],
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint)
             y.append(y_i)

    else:
        if output_activation is not None:
            output_activation=Activation(output_activation)
        y = Conv1DTranspose(h_i, output_units, kernel_size=kernel_size[-1],
                            dilation_rate=dilation_rate[-1],
                            padding=padding,
                            name=('td-%d' % (num_td_layers-1)),
                            activation = output_activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint)
        
    model = Model(x, y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model



def TDNNV1TransposeWithEmbedInputV1(num_fc_layers, num_td_layers,
                                    output_units, hidden_fc_units, hidden_td_units, input_units, embed_units,
                                    kernel_size,
                                    dilation_rate=1, dilation_rate_factor=1, padding='same',
                                    output_activation=None, hidden_activation='relu',
                                    use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
                                    spatial_dropout=True,
                                    cat_embed_to_all_fc_layers=False,
                                    name='t-tdnn-v1-e-v1',
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                    kernel_regularizer=None, bias_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
        dilation_rate = [dilation_rate_factor*(num_td_layers - i) + dilation_rate for i in xrange(num_td_layers)]

    if isinstance(hidden_fc_units, list):
        assert num_fc_layers == len(hidden_fc_units)
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]


    x = Input(shape=(None, input_units,))
    e = Input(shape=(None, embed_units,))

    h_i = x
    for i in xrange(num_fc_layers):
        if i==0 or cat_embed_to_all_fc_layers:
            h_i = Concatenate(axis=-1)([h_i, e])
            
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
                h_i = Dropout(fc_dropout_rate)(h_i)

            
    for i in xrange(num_td_layers-1):
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
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
                h_i = Dropout(td_dropout_rate)(h_i)

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
             y_i = Conv1D(output_units[i], kernel_size=kernel_size[-1],
                          dilation_rate=dilation_rate[-1],
                          padding=padding,
                          name=('td-%d-%d' % (num_td_layers-1,i)),
                          activation = output_activation[i],
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)(h_i)
             y.append(y_i)

    else:
        if output_activation is not None:
            output_activation=Activation(output_activation)
        y = Conv1D(output_units, kernel_size=kernel_size[-1],
                   dilation_rate=dilation_rate[-1],
                   padding=padding,
                   name=('td-%d' % (num_td_layers-1)),
                   activation = output_activation,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   kernel_constraint=kernel_constraint,
                   bias_constraint=bias_constraint)(h_i)
        
    model = Model([x, e], y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model



    
def TDNNV2TransposeWithEmbedInputV1(num_fc_layers, num_td_layers,
                                    output_units, hidden_fc_units, hidden_td_units, input_units, embed_units,
                                    kernel_size,
                                    dilation_rate=1, dilation_rate_factor=1, padding='same',
                                    output_activation=None, hidden_activation='relu',
                                    use_batchnorm=True, td_dropout_rate=0, fc_dropout_rate=0,
                                    spatial_dropout=True,
                                    cat_embed_to_all_fc_layers=False,
                                    name='t-tdnn-v1-e-v1',
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                    kernel_regularizer=None, bias_regularizer=None,
                                    kernel_constraint=None, bias_constraint=None, return_context=False):
    
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
        dilation_rate = [dilation_rate_factor*(num_td_layers - i) + dilation_rate for i in xrange(num_td_layers)]

    if isinstance(hidden_fc_units, list):
        assert num_fc_layers == len(hidden_fc_units)
    else:
        hidden_fc_units = [hidden_fc_units for i in xrange(num_fc_layers)]


    x = Input(shape=(None, input_units,))
    e = Input(shape=(None, embed_units,))

    h_i = x
    for i in xrange(num_fc_layers):
        if i==0 or cat_embed_to_all_fc_layers:
            h_i = Concatenate(axis=-1)([h_i, e])
            
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
            if spatial_dropout:
                h_i = SpatialDropout1D(fc_dropout_rate)(h_i)
            else:
                h_i = Dropout(fc_dropout_rate)(h_i)

            
    for i in xrange(num_td_layers-1):
        if use_batchnorm:
            h_i = BatchNormalization()(h_i)

        h_i = Conv1DTranspose(h_i, hidden_td_units, kernel_size=kernel_size[i],
                              dilation_rate=dilation_rate[i],
                              padding=padding, name=('td-%d' % i), 
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint)

        if hidden_activation is not None:
            h_i = hidden_activation(h_i)

        if td_dropout_rate > 0:
            if spatial_dropout:
                h_i = SpatialDropout1D(td_dropout_rate)(h_i)
            else:
                h_i = Dropout(td_dropout_rate)(h_i)

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
             y_i = Conv1DTranspose(h_i, output_units[i], kernel_size=kernel_size[-1],
                                   dilation_rate=dilation_rate[-1],
                                   padding=padding,
                                   name=('td-%d-%d' % (num_td_layers-1,i)),
                                   activation = output_activation[i],
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint)
             y.append(y_i)

    else:
        if output_activation is not None:
            output_activation=Activation(output_activation)
        y = Conv1DTranspose(h_i, output_units, kernel_size=kernel_size[-1],
                            dilation_rate=dilation_rate[-1],
                            padding=padding,
                            name=('td-%d' % (num_td_layers-1)),
                            activation = output_activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint)
        
    model = Model([x, e], y, name=name)
    if return_context:
        context = int(np.sum(np.array(dilation_rate)*(np.array(kernel_size)-1)/2))
        return model, context
    return model

