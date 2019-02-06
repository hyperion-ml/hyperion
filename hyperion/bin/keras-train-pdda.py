#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
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
import time
import logging

import numpy as np

from keras import backend as K

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.transforms import TransformList
from hyperion.utils.tensors import to3D_by_class
from hyperion.helpers import VectorClassReader as VCR
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedVAE_qYqZgY as TVAEYZ
from hyperion.keras.vae import TiedVAE_qY as TVAEY



def train_pdda(iv_file, train_list, val_list,
               decoder_file, qy_file, qz_file,
               epochs, batch_size,
               preproc_file, output_path,
               num_samples_y, num_samples_z,
               px_form, qy_form, qz_form,
               min_kl, **kwargs):

    set_float_cpu('float32')
    
    vcr_args = VCR.filter_args(**kwargs)
    opt_args = KOF.filter_args(**kwargs)
    cb_args = KCF.filter_args(**kwargs)

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

        
    vcr_train = VCR(iv_file, train_list, preproc, **vcr_args)
    max_length = vcr_train.max_samples_per_class
    
    x_val = None
    sw_val = None
    if val_list is not None:
        vcr_val = VCR(iv_file, val_list, preproc, **vcr_args)
        max_length = max(max_length, vcr_val.max_samples_per_class)
        x_val, sw_val = vcr_val.read(return_3d=True, max_length=max_length)
        
    x, sw = vcr_train.read(return_3d=True, max_length=max_length)
        
    t1 = time.time()
    decoder = load_model_arch(decoder_file)
    qy = load_model_arch(qy_file)

    if qz_file is None:
        vae = TVAEY(qy, decoder, px_cond_form=px_form,
                    qy_form=qy_form, min_kl=min_kl)
        vae.build(num_samples=num_samples_y, 
                  max_seq_length = x.shape[1])
    else:
        qz = load_model_arch(qz_file)
        vae = TVAEYZ(qy, qz, decoder, px_cond_form=px_form,
                   qy_form=qy_form, qz_form=qz_form, min_kl=min_kl)
        vae.build(num_samples_y=num_samples_y, num_samples_z=num_samples_z,
                  max_seq_length = x.shape[1])
    logging.info(time.time()-t1)
    # opt = create_optimizer(**opt_args)
    # cb = create_basic_callbacks(vae, output_path, **cb_args)
    # h = vae.fit(x, x_val=x_val,
    #             sample_weight_train=sw, sample_weight_val=sw_val,
    #             optimizer=opt, shuffle=True, epochs=100,
    #             batch_size=batch_size, callbacks=cb)

    # opt = create_optimizer(**opt_args)
    # cb = create_basic_callbacks(vae, output_path, **cb_args)
    # h = vae.fit_mdy(x, x_val=x_val,
    #                 sample_weight_train=sw, sample_weight_val=sw_val,
    #                 optimizer=opt, shuffle=True, epochs=200,
    #                 batch_size=batch_size, callbacks=cb)
    
    # y_mean, y_logvar, z_mean, z_logvar = vae.compute_qyz_x(
    #     x, batch_size=batch_size)
    # sw = np.expand_dims(sw, axis=-1)
    # m_y = np.mean(np.mean(y_mean, axis=0))
    # s2_y = np.sum(np.sum(np.exp(y_logvar)+y_mean**2, axis=0)/
    #               y_logvar.shape[0]-m_y**2)
    # m_z = np.mean(np.sum(np.sum(z_mean*sw, axis=1), axis=0)
    #               /np.sum(sw))
    # s2_z = np.sum(np.sum(np.sum((np.exp(z_logvar)+z_mean**2)*sw, axis=1), axis=0)
    #               /np.sum(sw)-m_z**2)
    # logging.info('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
    #       (m_y, s2_y, m_z, s2_z))

    
    cb = KCF.create_callbacks(vae, output_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)

    h = vae.fit(x, x_val=x_val,
                sample_weight_train=sw, sample_weight_val=sw_val,
                optimizer=opt, shuffle=True, epochs=epochs,
                batch_size=batch_size, callbacks=cb)

    if vae.x_chol is not None:
        x_chol = np.array(K.eval(vae.x_chol))
        logging.info(x_chol[:4,:4])
        
    
    logging.info('Train elapsed time: %.2f' % (time.time() - t1))
    
    vae.save(output_path + '/model')

    t1 = time.time()
    elbo = np.mean(vae.elbo(x, num_samples=1, batch_size=batch_size))
    logging.info('elbo: %.2f' % elbo)

    logging.info('Elbo elapsed  time: %.2f' % (time.time() - t1))

    t1 = time.time()
    vae.build(num_samples_y=1, num_samples_z=1, max_seq_length = x.shape[1])
    vae.compile()


    qyz = vae.compute_qyz_x(x, batch_size=batch_size)
    if vae.qy_form == 'diag_normal':
        y_mean, y_logvar = qyz[:2]
        qz = qyz[2:]
    else:
        y_mean, y_logvar, y_chol = qyz[:3]
        qz = qyz[3:]
    if vae.qz_form == 'diag_normal':
        z_mean, z_logvar = qz[:2]
    else:
        z_mean, z_logvar, z_chol = qz[:3]

    sw = np.expand_dims(sw, axis=-1)
    m_y = np.mean(np.mean(y_mean, axis=0))
    s2_y = np.sum(np.sum(np.exp(y_logvar)+y_mean**2, axis=0)/
                  y_logvar.shape[0]-m_y**2)
    m_z = np.mean(np.sum(np.sum(z_mean*sw, axis=1), axis=0)
                  /np.sum(sw))
    s2_z = np.sum(np.sum(np.sum((np.exp(z_logvar)+z_mean**2)*sw, axis=1), axis=0)
                  /np.sum(sw)-m_z**2)
    logging.info('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
          (m_y, s2_y, m_z, s2_z))

    logging.info('Trace elapsed time: %.2f' % (time.time() - t1))

    t1 = time.time()
    vae.build(num_samples_y=1, num_samples_z=1, max_seq_length = 2)
    vae.compile()
    
    x1 = x[:,0,:]
    x2 = x[:,1,:]
    # scores = vae.eval_llr_1vs1_elbo(x1, x2, num_samples=10)
    # tar = scores[np.eye(scores.shape[0], dtype=bool)]
    # non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    # logging.info('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    # logging.info('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))

    # scores = vae.eval_llr_1vs1_cand(x1, x2)
    # tar = scores[np.eye(scores.shape[0], dtype=bool)]
    # non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    # logging.info('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    # logging.info('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))

    scores = vae.eval_llr_1vs1_qscr(x1, x2)
    tar = scores[np.eye(scores.shape[0], dtype=bool)]
    non = scores[np.logical_not(np.eye(scores.shape[0], dtype=bool))]
    logging.info('m_tar: %.2f s_tar: %.2f' % (np.mean(tar), np.std(tar)))
    logging.info('m_non: %.2f s_non: %.2f' % (np.mean(non), np.std(non)))
    
    logging.info('Eval elapsed time: %.2f' % (time.time() - t1))


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train PDDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--decoder-file', dest='decoder_file', required=True)
    parser.add_argument('--qy-file', dest='qy_file', required=True)
    parser.add_argument('--qz-file', dest='qz_file', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--output-path', dest='output_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size', default=512,type=int,
                        help=('Batch size (default: %(default)s)'))

    VCR.add_argparse_args(parser)
    KOF.add_argparse_args(parser)
    KCF.add_argparse_args(parser)

    
    # parser.add_argument('--optimizer', dest='opt_type', type=str.lower,
    #                     default='adam',
    #                     choices=['sgd','nsgd','rmsprop','adam','nadam','adamax'],
    #                     help=('Optimizers: SGD, '
    #                           'NSGD (SGD with Nesterov momentum), '
    #                           'RMSprop, Adam, Adamax, '
    #                           'Nadam (Adam with Nesterov momentum), '
    #                           '(default: %(default)s)'))

    # parser.add_argument('--lr' , dest='lr',
    #                     default=0.002, type=float,
    #                     help=('Initial learning rate (default: %(default)s)'))
    # parser.add_argument('--momentum', dest='momentum', default=0.6, type=float,
    #                     help=('Momentum (default: %(default)s)'))
    # parser.add_argument('--lr-decay', dest='lr_decay', default=1e-6, type=float,
    #                     help=('Learning rate decay in SGD optimizer '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--rho', dest='rho', default=0.9, type=float,
    #                     help=('Rho in RMSprop optimizer (default: %(default)s)'))
    # parser.add_argument('--epsilon', dest='epsilon', default=1e-8, type=float,
    #                     help=('Epsilon in RMSprop and Adam optimizers '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--beta1', dest='beta_1', default=0.9, type=float,
    #                     help=('Beta_1 in Adam optimizers (default: %(default)s)'))
    # parser.add_argument('--beta2', dest='beta_2', default=0.999, type=float,
    #                     help=('Beta_2 in Adam optimizers (default: %(default)s)'))
    # parser.add_argument('--schedule-decay', dest='schedule_decay',
    #                     default=0.004,type=float,
    #                     help=('Schedule decay in Nadam optimizer '
    #                           '(default: %(default)s)'))

    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)

    parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
                        help=('Seed for the random number generator '
                              '(default: %(default)s)'))

    # parser.add_argument('--patience', dest='patience', default=100, type=int,
    #                     help=('Training stops after PATIENCE epochs without '
    #                           'improvement of the validation loss '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--lr-patience', dest='lr_patience', default=10, type=int,
    #                     help=('Multiply the learning rate by LR_FACTOR '
    #                           'after LR_PATIENCE epochs without '
    #                           'improvement of the validation loss '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--lr-factor', dest='lr_factor', default=0.1, type=float,
    #                     help=('Learning rate scaling factor '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--min-delta', dest='min_delta', default=1e-4, type=float,
    #                     help=('Minimum improvement'
    #                           '(default: %(default)s)'))
    # parser.add_argument('--min-lr', dest='min_lr', default=1e-5, type=float,
    #                     help=('Minimum learning rate'
    #                           '(default: %(default)s)'))
    # parser.add_argument('--lr-steps', dest='lr_steps', nargs='+', default=None)
    # parser.add_argument('--save-all-epochs', dest='save_best_only',
    #                     default=True, action='store_false')

    parser.add_argument('--num-samples-y', dest='num_samples_y', type=int,
                        default=1)
    parser.add_argument('--num-samples-z', dest='num_samples_z', type=int,
                        default=1)
    # parser.add_argument('--min-spc', dest='min_spc', type=int,
    #                     default=1)
    # parser.add_argument('--max-spc', dest='max_spc', type=int,
    #                     default=None)

    # parser.add_argument('--max-seq-length', dest='max_seq_length', type=int,
    #                     default=None)

    parser.add_argument('--px-form', dest='px_form', default='diag_normal')
    parser.add_argument('--qy-form', dest='qy_form', default='diag_normal')
    parser.add_argument('--qz-form', dest='qz_form', default='diag_normal')
    
    parser.add_argument('--min-kl', dest='min_kl', default=0.2, type=float)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    train_pdda(**vars(args))

            
