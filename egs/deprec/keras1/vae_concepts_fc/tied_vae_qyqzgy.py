#!/usr/bin/env python

'''
Run Tied VAE with Q(y,Z)=Q(y)\prod Q(z_i|y)
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Merge
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import initializations
from keras import optimizers
from keras.regularizers import l2

import h5py

from hyperion.keras1.layers.core import *
from hyperion.keras1.vae import TiedVAE_qYqZgY as TVAE

from utils import *


def vae(file_path):

    file_path2 = file_path + '/tied_vae_qyqzgy'
    
    # load data
    x_train, r_train, t_train, x_val, r_val, t_val, K, M = load_xrt(
        file_path + '/data.h5')

    x_val2d = x_val
    x_train = x_2dto3d(x_train, M)
    x_val = x_2dto3d(x_val, M)
    N_i=x_train.shape[1]
    elbo_samples = 2
    
    batch_size = 10
    x_dim = 2
    z_dim = 1
    y_dim = 1
    h_dim = 200
    nb_epoch = 1000
    l2_reg=0.0001

    # define encoder architecture
    # define q(y|x)
    x = Input(shape=(N_i, x_dim,))
    h1_y = TimeDistributed(Dense(h_dim, activation='relu',
                                 init=my_init, W_regularizer=l2(l2_reg)))(x)
    h2_y = TimeDistributed(Dense(int(h_dim/2), activation='relu',
                                 init=my_init, W_regularizer=l2(l2_reg)))(h1_y)
    h2_y_pool = GlobalAveragePooling1D()(h2_y)
    y_mean = Dense(y_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_y_pool)
    y_logvar = Dense(y_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_y_pool)
    qy=Model(x,[y_mean, y_logvar])
    
    # define q(z|x,y)
    y1=Input(shape=(N_i, y_dim,))
    xy = Merge(mode='concat', concat_axis=-1)([x, y1])
    h1_z = TimeDistributed(Dense(h_dim, activation='relu',
                                 init=my_init, W_regularizer=l2(l2_reg)))(xy)
    h2_z = TimeDistributed(Dense(int(h_dim/2), activation='relu',
                                 init=my_init, W_regularizer=l2(l2_reg)))(h1_z)
    z_mean = TimeDistributed(Dense(z_dim, init=my_init,
                                   W_regularizer=l2(l2_reg)))(h2_z)
    z_logvar = TimeDistributed(Dense(z_dim, init=my_init,
                                     W_regularizer=l2(l2_reg)))(h2_z)
    qz=Model([x, y1], [z_mean, z_logvar])



    # define decoder architecture
    y=Input(shape=(N_i, y_dim,))
    z=Input(shape=(N_i, z_dim,))
    yz = Merge(mode='concat', concat_axis=-1)([y, z])
    h1_dec = TimeDistributed(Dense(h_dim, activation='relu',
                                init=my_init, W_regularizer=l2(l2_reg)))(yz)
    h2_dec = TimeDistributed(Dense(h_dim, activation='relu',
                                   init=my_init, W_regularizer=l2(l2_reg)))(h1_dec)
    x_dec_mean = TimeDistributed(Dense(x_dim, init=my_init,
                                       W_regularizer=l2(l2_reg)))(h2_dec)
    x_dec_logvar = TimeDistributed(Dense(x_dim, init=my_init,
                                         W_regularizer=l2(l2_reg)))(h2_dec)
    x_dec_chol = TiledConstTriu(x_dim, diag_val=1)(h2_dec)
    
    decoder=Model([y, z], [x_dec_mean, x_dec_logvar, x_dec_chol])

    
    # train VAE
    vae=TVAE(qy, qz, decoder, px_cond_form='normal')
    vae.build(nb_samples_y=elbo_samples, nb_samples_z=elbo_samples)
    opt = optimizers.Adam(lr=0.001)
    h = vae.fit(x_train, x_val=x_val, optimizer=opt,
                shuffle=True, nb_epoch=nb_epoch,
                batch_size=batch_size, callbacks=my_callbacks())
    save_hist(file_path2 + '_hist.h5', h.history, 'TiedVAE Q(y)Q(Z|y)')

    # plot the latent space
    vae.build(nb_samples_y=1, nb_samples_z=1)
    y_val, _, z_val, _ = vae.compute_qyz_x(x_val, batch_size=batch_size)
    y_val2d = x_3dto2d(y_val)
    z_val2d = x_3dto2d(z_val)
    plot_xzrt(x_val2d, z_val2d, r_val, t_val, file_path2 + '_z_val.pdf')
    plot_xyt(x_val2d, y_val2d, t_val, file_path2 + '_y_val.pdf')

    # decode z_val
    x_val_dec = vae.decode_yz(y_val, z_val, batch_size=batch_size)
    x_val_dec = x_3dto2d(x_val_dec)
    plot_xdecrt(x_val2d, x_val_dec, r_val, t_val, file_path2 + '_x_val_dec.pdf')

    # Sample x from VAE
    x_sample = vae.generate(M, N_i, batch_size=batch_size)
    x_sample = x_3dto2d(x_sample)
    plot_xsample(x_val2d, x_sample, r_val, file_path2 + '_sample.pdf')

    idx_sort=np.argsort(y_val.ravel())
    idx_sort=[idx_sort[int(i*M/6)] for i in xrange(1,6,2)]
    x_val_sample=x_val[idx_sort,:,:]
    y_sample=y_val[idx_sort,:]
    x_sample = vae.generate_x_g_y(y_sample, N_i, batch_size=batch_size)
    plot_xsample_t(x_val_sample, x_sample, file_path2 + '_sample_t.pdf')


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Runs VAE')

    parser.add_argument('--exp', dest='exp', required=True)
    args=parser.parse_args()

    vae(args.exp)
    
