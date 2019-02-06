#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Extracts TCVAE i-vectors
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
from keras.layers import Input
from keras.models import Model

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.io import HypDataWriter
from hyperion.helpers import SequencePostReader as SR
from hyperion.transforms import TransformList
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedCVAE_qYqZgY as TVAEYZ
#from hyperion.keras.vae import TiedVAE_qY as TVAEY


    
def extract_ivector(seq_file, file_list, post_file, model_file, preproc_file, output_path,
                    qy_only, max_length, **kwargs):

    set_float_cpu('float32')
    
    sr_args = SR.filter_eval_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr = SR(seq_file, file_list, post_file, batch_size=1,
            shuffle_seqs=False,
            preproc=preproc, **sr_args)
    
    t1 = time.time()

    # if qy_only:
    #     model = TVAEY.load(model_file)
    # else:
    model = TVAEYZ.load(model_file)
        
    #model.build(max_seq_length=sr.max_batch_seq_length)
    model.build(max_seq_length=1)
            
    max_length = np.minimum(sr.max_batch_seq_length, max_length)
    
    y = np.zeros((sr.num_seqs, model.y_dim), dtype=float_keras())
    xx = np.zeros((1, max_length, model.x_dim), dtype=float_keras())
    rr = np.zeros((1, max_length, model.r_dim), dtype=float_keras())
    keys = []

    xp = Input(shape=(max_length, model.x_dim,))
    rp = Input(shape=(max_length, model.r_dim,))
    qy_param = model.qy_net([xp, rp])
    qy_net = Model([xp, rp], qy_param)
    
    for i in xrange(sr.num_seqs):
        ti1 = time.time()
        x, r, key = sr.read_next_seq()
        ti2 = time.time()
        logging.info('Extracting i-vector %d/%d for %s, num_frames: %d' % (i, sr.num_seqs, key, x.shape[0]))
        keys.append(key)
        xx[:,:,:] = 0
        rr[:,:,:] = 0
        
        if x.shape[0] <= max_length:
            xx[0,:x.shape[0]] = x
            rr[0,:x.shape[0]] = r
            y[i] = qy_net.predict([xx, rr], batch_size=1)[0]
        else:
            num_batches = int(np.ceil(x.shape[0]/max_length))
            for j in xrange(num_batches-1):
                start = j*max_length
                xx[0] = x[start:start+max_length]
                rr[0] = r[start:start+max_length]
                y[i] += qy_net.predict([xx, rr], batch_size=1)[0].ravel()
            xx[0] = x[-max_length:]
            rr[0] = r[-max_length:]
            y[i] += qy_net.predict([xx, rr], batch_size=1)[0].ravel()
            y[i] /= num_batches
                
        ti4 = time.time()
        logging.info('Elapsed time i-vector %d/%d for %s, total: %.2f read: %.2f, vae: %.2f' %
              (i, sr.num_seqs, key, ti4-ti1, ti2-ti1, ti4-ti2))
            
    logging.info('Extract elapsed time: %.2f' % (time.time() - t1))
    
    hw = HypDataWriter(output_path)
    hw.write(keys, '', y)

    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Extract TVAE i-vectors')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--file-list', dest='file_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--post-file', dest='post_file', required=True)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--max-length', dest='max_length', default=48000, type=int)

    SR.add_argparse_eval_args(parser)

    parser.add_argument('--qy-only', dest='qy_only',
                        default=False, action='store_true')

    # parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
    #                     help=('Batch size (default: %(default)s)'))

    parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
                        help=('Seed for the random number generator '
                              '(default: %(default)s)'))
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
        
    extract_ivector(**vars(args))

            
