#!/usr/bin/env python

"""
Trains TVAE
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time

import numpy as np
import scipy.stats as scps

from keras import backend as K

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.helpers import SequenceReader as SR
from hyperion.transforms import TransformList
# from hyperion.keras.keras_utils import *
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.vae import TiedVAE_qYqZgY as TVAEYZ
from hyperion.keras.vae import TiedVAE_qY as TVAEY


def data_generator(sr, max_length):
    x, sample_weights = sr.read(return_3d=True, max_length=max_length)
    return_sw = True
    if sr.max_batch_seq_length==max_length and (
            sr.min_length==sr.max_length or
            np.min(sr.seq_length)==sr.max_length):
        return_sw = False
                                      
    if return_sw:
        yield (x, x, sample_weights)
    else:
        yield (x, x)

    
def train_tvae(seq_file, train_list, val_list,
               decoder_file, qy_file, qz_file,
               epochs, batch_size,
               preproc_file, out_path,
               num_samples_y, num_samples_z,
               px_form, qy_form, qz_form,
               min_kl, **kwargs):

    
    sr_args = SR.filter_args(**kwargs)
    opt_args = KOF.filter_args(**kwargs)
    cb_args = KCF.filter_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr_train = SR(seq_file, train_list, batch_size=batch_size,
                  preproc=preproc, **sr_args)
    max_length = sr.max_batch_seq_length
    gen_val = None
    if val_list is not None:
        sr_val = SR(seq_file, val_list, batch_size=batch_size,
                    preproc=preproc, **sr_args)
        max_length = max(max_length, sr_val.max_batch_seq_length)
        gen_val = lambda: data_generator(sr_val, max_length)

    gen_train = lambda: data_generator(sr_train, max_length)
    
            
    t1 = time.time()
    decoder = load_model_arch(decoder_file)
    qy = load_model_arch(qy_file)


    if qz_file is None:
        vae = TVAEY(qy, decoder, px_cond_form=px_form,
                    qy_form=qy_form, min_kl=min_kl)
        vae.build(num_samples=num_samples_y, 
                  max_seq_length = max_length)
    else:
        qz = load_model_arch(qz_file)
        vae = TVAEYZ(qy, qz, decoder, px_cond_form=px_form,
                   qy_form=qy_form, qz_form=qz_form, min_kl=min_kl)
        vae.build(num_samples_y=num_samples_y, num_samples_z=num_samples_z,
                  max_seq_length = max_length)
    print(time.time()-t1)
    
    cb = KCF.create_basic_callbacks(vae, out_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)

    h = vae.fit_generator(gen_train, x_val=gen_val,
                          optimizer=opt, epochs=epochs,
                          batch_size=batch_size, callbacks=cb)

    # if vae.x_chol is not None:
    #     x_chol = np.array(K.eval(vae.x_chol))
    #     print(x_chol[:4,:4])
        
    
    print('Train elapsed time: %.2f' % (time.time() - t1))
    
    vae.save(out_path + '/model')

    # t1 = time.time()
    # elbo = np.mean(vae.elbo(x_train, num_samples=1, batch_size=batch_size))
    # print('elbo: %.2f' % elbo)

    # print('Elbo elapsed  time: %.2f' % (time.time() - t1))

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
    # print('m_y: %.2f, trace_y: %.2f, m_z: %.2f, trace_z: %.2f' %
    #       (m_y, s2_y, m_z, s2_z))

    # print('Trace elapsed time: %.2f' % (time.time() - t1))



    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train TVAE')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--decoder-file', dest='decoder_file', required=True)
    parser.add_argument('--qy-file', dest='qy_file', required=True)
    parser.add_argument('--qz-file', dest='qz_file', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--out-path', dest='out_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
                        help=('Batch size (default: %(default)s)'))


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
    
    args=parser.parse_args()
    
    train_pdda(**vars(args))

            
