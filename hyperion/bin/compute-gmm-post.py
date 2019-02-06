#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Computes GMM posteriors
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
from hyperion.io import HypDataWriter
from hyperion.helpers import SequenceReader as SR
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM


def to_sparse(r, num_comp):
    index = np.argsort(r, axis=1)[:,-num_comp:]
    r_sparse = np.zeros((r.shape[0], num_comp), dtype=float_cpu())
    for i, index_i in enumerate(index):
        r_sparse[i] = r[i, index_i]
    r_sparse = r_sparse/np.sum(r_sparse, axis=-1, keepdims=True)
    return r_sparse, index


def to_dense(r_sparse, index, num_comp):
    r = np.zeros((r_sparse.shape[0], num_comp), dtype=float_cpu())
    for i in xrange(r_sparse.shape[0]):
        r[i, index[i]] = r_sparse[i]

    return r


def compute_gmm_post(seq_file, file_list, model_file, preproc_file, output_path,
                     num_comp, **kwargs):

    
    sr_args = SR.filter_eval_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    gmm = DiagGMM.load_from_kaldi(model_file)
        
    sr = SR(seq_file, file_list, batch_size=1,
            shuffle_seqs=False,
            preproc=preproc, **sr_args)
    
    t1 = time.time()
            
    logging.info(time.time()-t1)
    index = np.zeros((sr.num_seqs, num_comp), dtype=int)

    hw = HypDataWriter(output_path)
    for i in xrange(sr.num_seqs):
        x, key = sr.read_next_seq()
        logging.info('Extracting i-vector %d/%d for %s, num_frames: %d' % (i, sr.num_seqs, key, x.shape[0]))
        r = gmm.compute_z(x)
        r_s, index = to_sparse(r, num_comp)
        if i==0:
            r2 = to_dense(r_s, index, r.shape[1])
            logging.degug(np.sort(r[0,:])[-12:])
            logging.degug(np.sort(r2[0,:])[-12:])
            logging.degug(np.argsort(r[0,:])[-12:])
            logging.degug(np.argsort(r2[0,:])[-12:])

        hw.write([key], '.r', [r_s])
        hw.write([key], '.index', [index])
            
    logging.info('Extract elapsed time: %.2f' % (time.time() - t1))
    
    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Extract TVAE i-vectors')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--file-list', dest='file_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--num-comp', dest='num_comp', default=10)

    SR.add_argparse_eval_args(parser)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    
    compute_gmm_post(**vars(args))

            
