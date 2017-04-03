#!/usr/bin/env python

'''
Run Tied conditional VAE with Q(y,Z)=Q(y)\prod Q(z_i)
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

from hyperion.keras.vae import TiedCVAE_qYqZ as TVAE

from utils import *


def vae(file_path):

    file_path2 = file_path + '/tied_cvae_qyqz'
    
    # load data
    x_train, r_train, t_train, x_val, r_val, t_val, K, M = load_xrt(
        file_path + '/data.h5')

    x_val2d = x_val
    r_val2d = r_val
    x_train = x_2dto3d(x_train, M)
    x_val = x_2dto3d(x_val, M)
    r_train = x_2dto3d(to_onehot(r_train, K), M)
    r_val = x_2dto3d(to_onehot(r_val, K), M)

    N_i=x_train.shape[1]
    elbo_samples = 3

    batch_size = 10
    x_dim = 2
    z_dim = 1
    y_dim = 1
    h_dim = 200
    epochs = 1000
    l2_reg=0.0001


    # define encoder architecture
    x = Input(shape=(N_i,x_dim))
    r = Input(shape=(N_i, K))
    xr = Merge(mode='concat', concat_axis=-1)([x, r])
    h1 = TimeDistributed(Dense(h_dim, activation='relu',
                               init=my_init, W_regularizer=l2(l2_reg)))(xr)
    h2 = TimeDistributed(Dense(int(h_dim/2), activation='relu',
                               init=my_init, W_regularizer=l2(l2_reg)))(h1)
    z_mean = TimeDistributed(Dense(z_dim, init=my_init,
                                   W_regularizer=l2(l2_reg)))(h2)
    z_logvar = TimeDistributed(Dense(z_dim, init=my_init,
                                     W_regularizer=l2(l2_reg)))(h2)
    
    h3 = TimeDistributed(Dense(int(h_dim/2), activation='relu',
                               init=my_init, W_regularizer=l2(l2_reg)))(h1)
    
    h3pool = GlobalAveragePooling1D()(h3)
    y_mean = Dense(y_dim, activation='relu',
                   init=my_init, W_regularizer=l2(l2_reg))(h3pool)
    y_logvar = Dense(y_dim, activation='relu',
                     init=my_init, W_regularizer=l2(l2_reg))(h3pool)
    
    encoder=Model([x, r],[y_mean, y_logvar, z_mean, z_logvar])
    
    # define decoder architecture
    y=Input(shape=(N_i, y_dim))
    z=Input(shape=(N_i, z_dim))
    yzr = Merge(mode='concat', concat_axis=-1)([z, y, r])
    
    h1_dec = TimeDistributed(Dense(h_dim, activation='relu',
                                   init=my_init, W_regularizer=l2(l2_reg)))(yzr)
    h2_dec = TimeDistributed(Dense(h_dim, activation='relu',
                                   init=my_init, W_regularizer=l2(l2_reg)))(h1_dec)
    x_dec_mean = TimeDistributed(Dense(x_dim, init=my_init,
                                       W_regularizer=l2(l2_reg)))(h2_dec)
    x_dec_logvar = TimeDistributed(Dense(x_dim, init=my_init,
                                         W_regularizer=l2(l2_reg)))(h2_dec)

    decoder=Model([y, z, r],[x_dec_mean, x_dec_logvar])

    
    # train VAE
    vae=TVAE(encoder, decoder, 'diag_normal')
    vae.build(num_samples=elbo_samples)
    opt = optimizers.Adam(lr=0.001)
    h = vae.fit(x_train, r_train=r_train,
                x_val=x_val, r_val=r_val,
                optimizer=opt, shuffle=True, epochs=epochs,
                batch_size=batch_size, callbacks=my_callbacks())
    save_hist(file_path2 + '_hist.h5', h.history, 'TiedCVAE Q(y)Q(Z)')

    # plot the latent space
    y_val, _, z_val, _ = vae.compute_qyz_x(x_val, r_val, batch_size=batch_size)
    y_val2d = x_3dto2d(y_val)
    z_val2d = x_3dto2d(z_val)
    plot_xzrt(x_val2d, z_val2d, r_val2d, t_val, file_path2 + '_z_val.pdf')
    plot_xyt(x_val2d, y_val2d, t_val, file_path2 + '_y_val.pdf')

    # decode z_val
    x_val_dec = vae.decode_yz(y_val, z_val, r_val, batch_size=batch_size)
    x_val_dec = x_3dto2d(x_val_dec)
    plot_xdecrt(x_val2d, x_val_dec, r_val2d, t_val, file_path2 + '_x_val_dec.pdf')

    # Sample x from VAE
    x_sample = vae.generate(r_val, batch_size=batch_size)
    x_sample = x_3dto2d(x_sample)
    plot_xcsample(x_val2d, x_sample, r_val2d, file_path2 + '_sample.pdf')

    idx_sort=np.argsort(y_val.ravel())
    idx_sort=[idx_sort[int(i*M/6)] for i in xrange(1,6,2)]
    x_val_sample=x_val[idx_sort,:,:]
    y_sample=y_val[idx_sort,:]
    x_sample = vae.generate_x_g_y(y_sample, r_val, batch_size=batch_size)
    plot_xsample_t(x_val_sample, x_sample, file_path2 + '_sample_t.pdf')


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Runs VAE')

    parser.add_argument('--exp', dest='exp', required=True)
    args=parser.parse_args()

    vae(args.exp)
    
