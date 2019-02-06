#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Evaluate the likelihood of the ubm on some data
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

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.helpers import SequenceReader as SR
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM



def eval_elbo(seq_file, file_list, model_file, preproc_file,
              output_file, ubm_type, **kwargs):

    sr_args = SR.filter_eval_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr = SR(seq_file, file_list, batch_size=1,
            shuffle_seqs=False,
            preproc=preproc, **sr_args)
    
    t1 = time.time()

    if ubm_type == 'diag-gmm':
        model = DiagGMM.load(model_file)
    else:
        model = DiagGMM.load_from_kaldi(model_file)
    model.initialize()
    
    elbo = np.zeros((sr.num_seqs,), dtype=float_cpu())
    num_frames = np.zeros((sr.num_seqs,), dtype=int)
    keys = []
    for i in xrange(sr.num_seqs):
        x, key = sr.read_next_seq()
        keys.append(key)
        elbo[i] = model.elbo(x)
        num_frames[i] = x.shape[0]

    num_total_frames = np.sum(num_frames)
    total_elbo = np.sum(elbo)
    total_elbo_norm = total_elbo/num_total_frames
    logging.info('Extract elapsed time: %.2f' % (time.time() - t1))
    s = 'Total ELBO: %f\nELBO_NORM %f' % (total_elbo, total_elbo_norm)
    logging.info(s)

    with open(output_file,'w') as f:
        f.write(s)
    
    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Evaluate UBM ELBO')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--file-list', dest='file_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-file', dest='output_file', required=True)
    parser.add_argument('--ubm-type', dest='ubm_type', default='diag-gmm',
                        choices=['diag-gmm', 'kaldi-diag-gmm'])
    
    SR.add_argparse_eval_args(parser)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    
    eval_elbo(**vars(args))

