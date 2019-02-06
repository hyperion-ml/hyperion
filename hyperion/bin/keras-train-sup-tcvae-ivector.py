#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains supervised TCVAE
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
from hyperion.utils.multithreading import threadsafe_generator
from hyperion.helpers.sequence_post_class_reader import SequencePostClassReader as SR
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM
from hyperion.keras.keras_utils import *
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.vae.tied_cvae_qyqzgy2 import TiedCVAE_qYqZgY2 as TVAEYZ
#from hyperion.keras.vae import TiedCVAE_qY as TVAEY


@threadsafe_generator
def data_generator(sr, max_length):
    kk=0
    while 1:
        kk+=1
        # logging.info('dg %d.' % kk)
        x, z, sample_weight, _, y = sr.read(return_3d=True, max_seq_length=max_length)

        return_sw = True
        if sr.max_batch_seq_length == max_length and (
                sr.min_seq_length == sr.max_seq_length or
                np.min(sr.seq_length) == sr.max_seq_length):
            return_sw = False
                                      
        if return_sw:
            yield ([x, z], [x, y], [sample_weight, None])
        else:
            yield ([x, z], [x, y])

    
def train_tvae(seq_file, train_list, val_list,
               class_list,
               post_file,
               decoder_file, pt_file, qy_file, qz_file,
               init_path,
               epochs, batch_size,
               preproc_file, output_path,
               num_samples_y, num_samples_z,
               px_form, qy_form, qz_form,
               min_kl, loss_weights, **kwargs):

    set_float_cpu(float_keras())
    
    sr_args = SR.filter_args(**kwargs)
    sr_val_args = SR.filter_val_args(**kwargs)
    opt_args = KOF.filter_args(**kwargs)
    cb_args = KCF.filter_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

        
    sr = SR(seq_file, train_list, post_file, class_list, batch_size=batch_size,
                  preproc=preproc, **sr_args)
    max_length = sr.max_batch_seq_length
    gen_val = None
    if val_list is not None:
        sr_val = SR(seq_file, val_list, post_file, class_list, batch_size=batch_size,
                    preproc=preproc,
                    shuffle_seqs=False,
                    seq_split_mode='sequential', seq_split_overlap=0,
                    reset_rng=True,
                    **sr_val_args)
        max_length = max(max_length, sr_val.max_batch_seq_length)
        gen_val = data_generator(sr_val, max_length)

    gen_train = data_generator(sr, max_length)
    
            
    t1 = time.time()
    decoder = load_model_arch(decoder_file)
    qy = load_model_arch(qy_file)
    pt = load_model_arch(pt_file)

    if init_path is None:
        decoder = load_model_arch(decoder_file)
        qy = load_model_arch(qy_file)

        # if qz_file is None:
        #     vae = TVAEY(qy, decoder, px_cond_form=px_form,
        #                 qy_form=qy_form, min_kl=min_kl)
        #     vae.build(num_samples=num_samples_y, 
        #               max_seq_length = max_length)
        # else:
        qz = load_model_arch(qz_file)
        vae = TVAEYZ(qy, qz, decoder, pt, px_cond_form=px_form,
                     qy_form=qy_form, qz_form=qz_form, min_kl=min_kl, loss_weights=loss_weights)
    else:
        logging.info('loading init model: %s' % init_path)
        vae = TVAEYZ.load(init_path)

    logging.info('max length: %d' % max_length)
    vae.build(num_samples_y=num_samples_y, num_samples_z=num_samples_z,
              max_seq_length = max_length)
    logging.info(time.time()-t1)
    
    cb = KCF.create_callbacks(vae, output_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)

    h = vae.fit_generator(gen_train, x_val=gen_val,
                          steps_per_epoch=sr.num_batches,
                          validation_steps=sr_val.num_batches,
                          optimizer=opt, epochs=epochs,
                          callbacks=cb, max_queue_size=10)

    # if vae.x_chol is not None:
    #     x_chol = np.array(K.eval(vae.x_chol))
    #     logging.info(x_chol[:4,:4])
    
    logging.info('Train elapsed time: %.2f' % (time.time() - t1))
    
    vae.save(output_path + '/model')
    sr_val.reset()
    y_val, sy_val, z_val, srz_val = vae.encoder_net.predict_generator(gen_val, steps=400)

    from scipy import linalg as la
    yy = y_val - np.mean(y_val, axis=0)
    cy = np.dot(yy.T, yy)/yy.shape[0]
    l,v = la.eigh(cy)
    np.savetxt(output_path + '/l1.txt', l)

    sr_val.reset()
    y_val2, sy_val2 = vae.qy_net.predict_generator(gen_val, steps=400)
    yy = y_val2 - np.mean(y_val, axis=0)
    cy = np.dot(yy.T, yy)/yy.shape[0]
    l,v = la.eigh(cy)
    np.savetxt(output_path + '/l2.txt', l)

    logging.info(y_val-y_val2)
    
    # t1 = time.time()
    # elbo = np.mean(vae.elbo(x_train, num_samples=1, batch_size=batch_size))
    # logging.info('elbo: %.2f' % elbo)

    # logging.info('Elbo elapsed  time: %.2f' % (time.time() - t1))

    # t1 = time.time()
    # vae.build(num_samples_y=1, num_samples_z=1, max_seq_length = x_train.shape[1])
    # vae.compile()


    # qyz = vae.compute_qyz_x(x_train, batch_size=batch_size)
    # if vae.qy_form == 'diag_normal':
    #     y_mean, y_logvar = qyz[:2]
    #     qz = qyz[2:]
    # else:
    #     y_mean, y_logvar, y_chol = qyz[:3]
    #     qz = qyz[3:]
    # if vae.qz_form == 'diag_normal':
    #     z_mean, z_logvar = qz[:2]
    # else:
    #     z_mean, z_logvar, z_chol = qz[:3]

    # sw = np.expand_dims(sw_train, axis=-1)
    # m_y = np.mean(np.mean(y_mean, axis=0))
    # s2_y = np.sum(np.sum(np.exp(y_logvar)+y_mean**2, axis=0)/
    #               y_logvar.shape[0]-m_y**2)
    # m_z = np.mean(np.sum(np.sum(z_mean*sw, axis=1), axis=0)
    #               /np.sum(sw))
    # s2_z = np.sum(np.sum(np.sum((np.exp(z_logvar)+z_mean**2)*sw, axis=1), axis=0)
    #               /np.sum(sw)-m_z**2)
    # logging.info('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
    #       (m_y, s2_y, m_z, s2_z))

    # logging.info('Trace elapsed time: %.2f' % (time.time() - t1))



    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train Tied CVAE')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--class-list', dest='class_list', required=True)
    parser.add_argument('--post-file', dest='post_file', required=True)
    parser.add_argument('--decoder-file', dest='decoder_file', required=True)
    parser.add_argument('--qy-file', dest='qy_file', required=True)
    parser.add_argument('--qz-file', dest='qz_file', default=None)
    parser.add_argument('--pt-file', dest='pt_file', default=None)
    parser.add_argument('--init-path', dest='init_path', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--output-path', dest='output_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
                        help=('Batch size (default: %(default)s)'))

    parser.add_argument('--loss-weights', dest='loss_weights', nargs=2, type=float,
                        default=None)
    
    SR.add_argparse_args(parser)
    KOF.add_argparse_args(parser)
    KCF.add_argparse_args(parser)
    
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)

    parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
                        help=('Seed for the random number generator '
                              '(default: %(default)s)'))

    parser.add_argument('--num-samples-y', dest='num_samples_y', type=int,
                        default=1)
    parser.add_argument('--num-samples-z', dest='num_samples_z', type=int,
                        default=1)

    parser.add_argument('--px-form', dest='px_form', default='diag_normal')
    parser.add_argument('--qy-form', dest='qy_form', default='diag_normal')
    parser.add_argument('--qz-form', dest='qz_form', default='diag_normal')
    
    parser.add_argument('--min-kl', dest='min_kl', default=0.2, type=float)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
        
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
        
    train_tvae(**vars(args))

            
