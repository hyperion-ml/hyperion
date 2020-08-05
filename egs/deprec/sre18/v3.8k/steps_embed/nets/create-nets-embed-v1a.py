#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse

import numpy as np
import scipy.sparse

from keras.layers import Input, Dense, Concatenate, Add, MaxoutDense, Dropout, BatchNormalization, Activation, Lambda, Multiply, Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import initializers
from keras import optimizers
from keras.regularizers import l2

from hyperion.keras.keras_utils import save_model_arch
from hyperion.keras.layers.masking import CreateMask
from hyperion.keras.layers.pooling import *
from hyperion.keras.layers.core import *
from hyperion.keras.layers.cov import *
from hyperion.keras.archs import TDNNV1, FFNetV1

l2_reg = 1e-5

def create_pt_net(t_dim, y_dim, h_dim, p, my_init, act,
                  num_layers, output_path, **kwargs):
    
    # define post pooling net
    net2 = FFNetV1(num_layers, t_dim, h_dim, 2*y_dim, 'softmax', act, use_batchnorm=True,
                   dropout_rate=p, name='pt-net',
                   kernel_initializer=my_init, kernel_regularizer=l2(l2_reg))
    
    save_model_arch(output_path+'/pt.json', net2)


    
def create_enc_net(x_dim, y_dim, h_dim, num_layers, p, my_init, act,
                   output_path, **kwargs):

    # define prepooling net
    net1, context = TDNNV1(3, num_layers-3, y_dim, h_dim, h_dim, x_dim, [5, 3, 3],
                           padding='same', hidden_activation=act,
                           use_batchnorm=True, td_dropout_rate=p, fc_dropout_rate=p, spatial_dropout=True, name='enc-net',
                           kernel_initializer=my_init, kernel_regularizer=l2(l2_reg), return_context=True)
    save_model_arch(output_path+'/enc.json', net1)
        
    with open(output_path+'/context','w') as f:
        f.write('%d\n' % context)


        
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Create net archs')

    parser.add_argument('--x-dim', dest='x_dim', required=True, type=int,
                        help='input dimension')
    parser.add_argument('--y-dim', dest='y_dim', required=True, type=int,
                        help='encoder network output dimension')
    parser.add_argument('--t-dim', dest='t_dim', default=0, type=int,
                        help='number of training classes')
    parser.add_argument('--h-y-dim', dest='h_y_dim', default=10, type=int,
                        help='hidden dimension of tdnn network')
    parser.add_argument('--h-t-dim', dest='h_t_dim', default=10, type=int,
                        help='hidden dimension of classification network')
    parser.add_argument('--num-layers-y', dest='num_layers_y', default=1, type=int,
                        help='number of encoder layers')
    parser.add_argument('--num-layers-t', dest='num_layers_t', default=1, type=int,
                        help='number of classification network layers')
    parser.add_argument('--dropout', dest='p', default=0.25, type=float,
                        help='dropout drop prob.')
    parser.add_argument('--act', dest='act', default='relu',
                        choices=['relu','sigmoid','softplus', 'tanh', 'elu', 'selu'],
                        help='activation function')
    parser.add_argument('--init-f', dest='init_f', default='normal',
                        choices=['normal','uniform'], help='weight init distribution')
    parser.add_argument('--init-mode', dest='init_mode', default='fan_in',
                        choices=['fan_in','fan_out', 'fan_avg'], help='weight init mode')
    parser.add_argument('--init-s', dest='init_s', default=1, type=float,
                        help='weight init scaling')

    parser.add_argument('--output-path', dest='output_path', required=True,
                        help='output directory')
    
    args=parser.parse_args()

    my_init=initializers.VarianceScaling(scale=args.init_s, mode=args.init_mode, distribution=args.init_f)

    create_enc_net(**vars(args), my_init=my_init, h_dim=args.h_y_dim, num_layers=args.num_layers_y)
    create_pt_net(**vars(args), my_init=my_init,  h_dim=args.h_t_dim,
                  num_layers=args.num_layers_t)



        
