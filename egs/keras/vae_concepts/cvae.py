#!/usr/bin/env python

'''
Run conditional VAE
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
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2

import h5py

from hyperion.keras.vae import CVAE

from utils import *


def vae(file_path):

    # load data
    # x_train, r_train, x_val, r_val, K = load_xr(
    #     file_path + '/data.h5')
    x_train, r_train, t_train, x_val, r_val, t_val, K, M = load_xrt(
        file_path + '/data.h5')

    r1h_train = to_onehot(r_train, K)
    r1h_val = to_onehot(r_val, K)
    
    num_samples=x_train.shape[0]
    elbo_samples = 3
    
    batch_size = 100
    x_dim = 2
    z_dim = 1
    h_dim = 200
    epochs = 1000
    l2_reg=0.0001
    
    # define encoder architecture
    x = Input(shape=(x_dim,))
    r = Input(shape=(K,))
    xr = Merge(mode='concat', concat_axis=-1)([x, r])
    h1 = Dense(h_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(xr)
    h2 = Dense(h_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(h1)
    z_mean = Dense(z_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)
    z_logvar = Dense(z_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)
    
    encoder=Model([x, r], [z_mean, z_logvar])
    
    # define decoder architecture
    z=Input(shape=(z_dim,))
    zr = Merge(mode='concat', concat_axis=-1)([z, r])
    h1_dec = Dense(h_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(zr)
    h2_dec = Dense(h_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(h1_dec)
    x_dec_mean = Dense(x_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_dec)
    x_dec_logvar = Dense(x_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_dec)

    decoder=Model([z, r], [x_dec_mean, x_dec_logvar])

    # train VAE
    vae=CVAE(encoder,decoder,'diag_normal')
    vae.build(num_samples=elbo_samples)
    opt = optimizers.Adam(lr=0.001)
    h = vae.fit(x_train, r_train=r1h_train, x_val=x_val, r_val=r1h_val,
                optimizer=opt, shuffle=True, epochs=epochs,
                batch_size=batch_size, callbacks=my_callbacks())
    save_hist(file_path + '/cvae_hist.h5', h.history, 'CVAE')


    # plot the latent space
    z_val = vae.compute_qz_x(x_val, r1h_val, batch_size=batch_size)[0]
    if M == 0:
        plot_xzr(x_val, z_val, r_val, file_path + '/cvae_z_val.pdf')
    else:
        plot_xzrt(x_val, z_val, r_val, t_val, file_path + '/cvae_z_val.pdf')

    
    # decode z_val
    x_val_dec = vae.decode_z(z_val, r1h_val, batch_size=batch_size)
    plot_xdecr(x_val, x_val_dec, r_val, file_path + '/cvae_x_val_dec.pdf')

    # Sample x from VAE
    x_sample = vae.generate(r1h_val, batch_size=batch_size)
    plot_xcsample(x_val, x_sample, r_val, file_path + '/cvae_sample.pdf')


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Runs VAE')

    parser.add_argument('--exp', dest='exp', required=True)
    args=parser.parse_args()

    vae(args.exp)
    
