#!/usr/bin/env python

"""
Trains PDDA
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse

import numpy as np

from hyperion.io import HypDataReader
from hyperion.transforms import LNorm
from hyperion.utils.scp_list import SCPList
from hyperion.utils.tensors import to3D_by_class
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedVAE_qYqZgY2 as TVAE


def load_data(hyp_reader, utt2spk_file, lnorm, max_length):

    utt2spk = SCPList.load(utt2spk_file, sep='=')
    x = hyp_reader.read(utt2spk.file_path, '.ivec')

    if lnorm is not None:
        x = lnorm.predict(x)
        
    _, _, class_ids=np.unique(utt2spk.key,
                              return_index=True, return_inverse=True)

    [x, sample_weights] = to3D_by_class(x, class_ids, max_length)
    return x, sample_weights


def train_pdda(iv_file, train_utt2spk, val_utt2spk,
               decoder_file, qy_file, qz_file,
               nb_epoch, batch_size,
               lnorm_file, out_path, **kwargs):

    cb_args = filter_callbacks_args(**kwargs)
    opt_args = filter_optimizer_args(**kwargs)
    
    if lnorm_file is not None:
        lnorm = LNorm.load(lnorm_file)
    else:
        lnorm = None
    hr = HypDataReader(iv_file)
    [x_train, sw_train] = load_data(hr, train_utt2spk, lnorm, max_length=51)
    [x_val, sw_val] = load_data(hr, val_utt2spk, lnorm, max_length=51)

    decoder = load_model_arch(decoder_file)
    qy = load_model_arch(qy_file)
    qz = load_model_arch(qz_file)

    
    vae=TVAE(qy, qz, decoder, 'normal')
    vae.build()
    
    opt = create_optimizer(**opt_args)
    cb = create_basic_callbacks(vae, out_path, **cb_args)
    h = vae.fit(x_train, x_val=x_val,
                sample_weight_train=sw_train, sample_weight_val=sw_val,
                optimizer=opt, shuffle=True, nb_epoch=100,
                batch_size=batch_size, callbacks=cb)

    opt = create_optimizer(**opt_args)
    cb = create_basic_callbacks(vae, out_path, **cb_args)
    h = vae.fit_mdy(x_train, x_val=x_val,
                    sample_weight_train=sw_train, sample_weight_val=sw_val,
                    optimizer=opt, shuffle=True, nb_epoch=200,
                    batch_size=batch_size, callbacks=cb)
    y_mean, y_logvar, z_mean, z_logvar = vae.compute_qyz_x(
        x_train, batch_size=batch_size)
    sw = np.expand_dims(sw_train, axis=-1)
    m_y = np.mean(np.mean(y_mean, axis=0))
    s2_y = np.sum(np.sum(np.exp(y_logvar)+y_mean**2, axis=0)/
                  y_logvar.shape[0]-m_y**2)
    m_z = np.mean(np.sum(np.sum(z_mean*sw, axis=1), axis=0)
                  /np.sum(sw))
    s2_z = np.sum(np.sum(np.sum((np.exp(z_logvar)+z_mean**2)*sw, axis=1), axis=0)
                  /np.sum(sw)-m_z**2)
    print('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
          (m_y, s2_y, m_z, s2_z))

    
    cb = create_basic_callbacks(vae, out_path, **cb_args)
    opt = create_optimizer(**opt_args)
    h = vae.fit(x_train, x_val=x_val,
                sample_weight_train=sw_train, sample_weight_val=sw_val,
                optimizer=opt, shuffle=True, nb_epoch=nb_epoch,
                batch_size=batch_size, callbacks=cb)

    vae.save(out_path + '/model')
    
    elbo = np.mean(vae.elbo(x_train, nb_samples=1, batch_size=batch_size))
    print('elbo: %.2f' % elbo)
    
    y_mean, y_logvar, z_mean, z_logvar = vae.compute_qyz_x(
        x_train, batch_size=batch_size)
    sw = np.expand_dims(sw_train, axis=-1)
    m_y = np.mean(np.mean(y_mean, axis=0))
    s2_y = np.sum(np.sum(np.exp(y_logvar)+y_mean**2, axis=0)/
                  y_logvar.shape[0]-m_y**2)
    m_z = np.mean(np.sum(np.sum(z_mean*sw, axis=1), axis=0)
                  /np.sum(sw))
    s2_z = np.sum(np.sum(np.sum((np.exp(z_logvar)+z_mean**2)*sw, axis=1), axis=0)
                  /np.sum(sw)-m_z**2)
    print('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
          (m_y, s2_y, m_z, s2_z))

    x1 = x_train[:,0,:]
    x2 = x_train[:,1,:]
    scores = vae.eval_llr_1vs1_elbo(x1, x2, nb_samples=10)
    tar = scores[np.eye(scores.shape[0], dtype=bool)]
    non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    print('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    print('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))

    scores = vae.eval_llr_1vs1_cand(x1, x2)
    tar = scores[np.eye(scores.shape[0], dtype=bool)]
    non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    print('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    print('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))

    scores = vae.eval_llr_1vs1_qscr(x1, x2)
    tar = scores[np.eye(scores.shape[0], dtype=bool)]
    non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    print('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    print('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train PDDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-utt2spk', dest='train_utt2spk', required=True)
    parser.add_argument('--val-utt2spk', dest='val_utt2spk', default=None)
    parser.add_argument('--decoder-file', dest='decoder_file', required=True)
    parser.add_argument('--qy-file', dest='qy_file', required=True)
    parser.add_argument('--qz-file', dest='qz_file', required=True)
    parser.add_argument('--lnorm-file', dest='lnorm_file', default=None)
    parser.add_argument('--out-path', dest='out_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
                        help=('Batch size (default: %(default)s)'))

    parser.add_argument('--optimizer', dest='opt_type', type=str.lower,
                        default='adam',
                        choices=['sgd','nsgd','rmsprop','adam','nadam','adamax'],
                        help=('Optimizers: SGD, '
                              'NSGD (SGD with Nesterov momentum), '
                              'RMSprop, Adam, Adamax, '
                              'Nadam (Adam with Nesterov momentum), '
                              '(default: %(default)s)'))

    parser.add_argument('--lr' , dest='lr',
                        default=0.002, type=float,
                        help=('Initial learning rate (default: %(default)s)'))
    parser.add_argument('--momentum', dest='momentum', default=0.6, type=float,
                        help=('Momentum (default: %(default)s)'))
    parser.add_argument('--lr-decay', dest='lr_decay', default=1e-6, type=float,
                        help=('Learning rate decay in SGD optimizer '
                              '(default: %(default)s)'))
    parser.add_argument('--rho', dest='rho', default=0.9, type=float,
                        help=('Rho in RMSprop optimizer (default: %(default)s)'))
    parser.add_argument('--epsilon', dest='epsilon', default=1e-8, type=float,
                        help=('Epsilon in RMSprop and Adam optimizers '
                              '(default: %(default)s)'))
    parser.add_argument('--beta1', dest='beta_1', default=0.9, type=float,
                        help=('Beta_1 in Adam optimizers (default: %(default)s)'))
    parser.add_argument('--beta2', dest='beta_2', default=0.999, type=float,
                        help=('Beta_2 in Adam optimizers (default: %(default)s)'))
    parser.add_argument('--schedule-decay', dest='schedule_decay',
                        default=0.004,type=float,
                        help=('Schedule decay in Nadam optimizer '
                              '(default: %(default)s)'))

    parser.add_argument('--nb-epoch', dest='nb_epoch', default=1000, type=int)

    parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
                        help=('Seed for the random number generator '
                              '(default: %(default)s)'))

    parser.add_argument('--patience', dest='patience', default=100, type=int,
                        help=('Training stops after PATIENCE epochs without '
                              'improvement of the validation loss '
                              '(default: %(default)s)'))
    parser.add_argument('--lr-patience', dest='lr_patience', default=10, type=int,
                        help=('Multiply the learning rate by LR_FACTOR '
                              'after LR_PATIENCE epochs without '
                              'improvement of the validation loss '
                              '(default: %(default)s)'))
    parser.add_argument('--lr-factor', dest='lr_factor', default=0.1, type=float,
                        help=('Learning rate scaling factor '
                              '(default: %(default)s)'))
    parser.add_argument('--min-delta', dest='min_delta', default=1e-4, type=float,
                        help=('Minimum improvement'
                              '(default: %(default)s)'))
    parser.add_argument('--min-lr', dest='min_lr', default=1e-5, type=float,
                        help=('Minimum learning rate'
                              '(default: %(default)s)'))
    parser.add_argument('--lr-steps', dest='lr_steps', nargs='+', default=None)
    parser.add_argument('--save-all-epochs', dest='save_best_only',
                        default=True, action='store_false')
    
    args=parser.parse_args()
    
    train_pdda(**vars(args))

            
