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
from hyperion.keras.archs import FTDNNV1, FFNetV1



def create_net2(t_dim, y_dim, h_dim, p, my_init, act,
                num_layers, out_path, **kwargs):

    # define decoder architecture
    N_i = None
    l2_reg = 1e-5

    net2 = FFNetV1(num_layers, t_dim, h_dim, 2*y_dim, 'softmax', act, use_batchnorm=True,
                   dropout_rate=p, name='net-2',
                   kernel_initializer=my_init, kernel_regularizer=l2(l2_reg))
    
    save_model_arch(out_path+'/net2.json', net2)


    

def create_net1(x_dim, y_dim, h_dim, num_layers, p, my_init, act,
                out_path, **kwargs):

    # define q(y|x)
    N_i = None
    l2_reg = 1e-5

    kernel_sizes =  [5] + 7*[3] + 20*[1]
    kernel_sizes[2] = 1
    kernel_sizes[4] = 1
    skip_conn = {4: [2], 6: [1,3], 8:[3,5,7]}
    kernel_sizes = kernel_sizes[:num_layers-1]
    net1, context = FTDNNV1(1, num_layers-2, 1, y_dim, 512, h_dim, int(h_dim/4), h_dim, x_dim,
                            kernel_sizes,
                            dilation_rate = [1,2] + (num_layers-3)*[3], 
                            padding='same',
                            skip_conn=skip_conn,
                            hidden_activation=act,
                            use_batchnorm=True, td_dropout_rate=p, fc_dropout_rate=p, spatial_dropout=True, name='net-1',
                            kernel_initializer=my_init, kernel_regularizer=l2(l2_reg), return_context=True)
    save_model_arch(out_path+'/net1.json', net1)
        
    with open(out_path+'/context','w') as f:
        f.write('%d\n' % context)
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Create net archs')

    parser.add_argument('--x-dim', dest='x_dim', required=True, type=int)
    parser.add_argument('--y-dim', dest='y_dim', required=True, type=int)
    parser.add_argument('--t-dim', dest='t_dim', default=0, type=int)
    parser.add_argument('--h-y-dim', dest='h_y_dim', default=10, type=int)
    parser.add_argument('--h-t-dim', dest='h_t_dim', default=10, type=int)
    parser.add_argument('--num-layers-y', dest='num_layers_y', default=1, type=int)
    parser.add_argument('--num-layers-t', dest='num_layers_t', default=1, type=int)
    parser.add_argument('--dropout', dest='p', default=0.25, type=float)
    parser.add_argument('--act', dest='act', default='relu', choices=['relu','sigmoid','softplus', 'tanh', 'elu', 'selu'])
    parser.add_argument('--init-f', dest='init_f', default='normal', choices=['normal','uniform'])
    parser.add_argument('--init-mode', dest='init_mode', default='fan_in', choices=['fan_in','fan_out', 'fan_avg'])
    parser.add_argument('--init-s', dest='init_s', default=1, type=float)

    parser.add_argument('--out-path', dest='out_path', required=True)
    
    args=parser.parse_args()

    my_init=initializers.VarianceScaling(scale=args.init_s, mode=args.init_mode, distribution=args.init_f)

    create_net1(**vars(args), my_init=my_init, h_dim=args.h_y_dim, num_layers=args.num_layers_y)
    create_net2(**vars(args), my_init=my_init,  h_dim=args.h_t_dim,
                num_layers=args.num_layers_t)



        
