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
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedVAE_qYqZgY as TVAEYZ
from hyperion.keras.vae import TiedVAE_qY as TVAEY


    
def extract_ivector(seq_file, file_list, model_file, preproc_file, output_file,
                    **kwargs):

    set_float_cpu('float32')
    
    sr_args = SR.filter_eval_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr = SR(seq_file, file_list, batch_size=1,
            shuffle_seqs=False,
            preproc=preproc, **sr_args)
    
    t1 = time.time()

    if qy_only:
        model = TVAEY.load(model_file)
    else:
        model = TVAEYZ.load(model_file)

    print(time.time()-t1)

    y = np.zeros((sr.num_seqs,vae.y_dim), dtype=float_keras())
    keys = []
    for i in xrange(sr.num_seqs):
        x, key = sr.read_nex_seq()
        keys.append(key)
        model.build(max_seq_length=2)
        y[i] = vae.compute_qy_x()[0]
            
    print('Extract elapsed time: %.2f' % (time.time() - t1))
    
    hw = HypDataWriter(output_file)
    hw.write(keys, '', y)

    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Extract TVAE i-vectors')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--file-list', dest='file_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-file', dest='out_file', required=True)

    SR.add_argparse_eval_args(parser)

    parser.add_argument('--qy-only', dest='qy_only',
                        default=False, action='store_true')

    # parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
    #                     help=('Batch size (default: %(default)s)'))

    parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
                        help=('Seed for the random number generator '
                              '(default: %(default)s)'))

    args=parser.parse_args()
    
    extract_ivector(**vars(args))

            
