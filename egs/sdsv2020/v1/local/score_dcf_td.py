#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
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
import scipy.sparse as sparse

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import SparseTrialScores, SparseTrialKey
from hyperion.utils import TrialScores, TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval
from hyperion.metrics import effective_prior


def score_dcf(key_file, score_file, output_path):

    logging.info('Load key: %s' % key_file)
    key = TrialKey.load_txt(key_file)
    key_samespk = TrialKey.load_txt(key_file + '_samespk').filter(
        key.model_set, key.seg_set)
    key_samephr = TrialKey.load_txt(key_file + '_samephr').filter(
        key.model_set, key.seg_set)
    key_nonsamespk = key.copy()
    key_nonsamespk.non = np.logical_and(key.non, key_samespk.tar)
    key_nonsamephr = key.copy()
    key_nonsamephr.non = np.logical_and(key.non, key_samephr.tar)
    key_nondiffspkphr = key.copy()
    key_nondiffspkphr.non = np.logical_and(key_samespk.non, key_samephr.non)

    # key = SparseTrialKey.load_txt(key_file)
    # key_samespk = SparseTrialKey.load_txt(key_file + '_samespk').filter(
    #     key.model_set, key.seg_set)
    # key_samephr = SparseTrialKey.load_txt(key_file + '_samephr').filter(
    #     key.model_set, key.seg_set)
    # key_nonsamespk = key.copy()
    # key_nonsamespk.non = key.non.multiply(key_samespk.tar).astype('bool')
    # key_nonsamephr = key.copy()
    # key_nonsamephr.non = key.non.multiply(key_samephr.tar).astype('bool')
    # key_nondiffspkphr = key.copy()
    # key_nondiffspkphr.non = key_samespk.non.multiply(key_samephr.non).astype('bool')

    logging.info('Load scores: %s' % score_file)
    scr = TrialScores.load_txt(score_file)

    prior = effective_prior(0.01, 10, 1)

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = output_path + '_results'
    with open(output_file, 'w') as f:

        logging.info('Tar-samespkphr-vs-Non-diffspkorphr results')
        tar, non = scr.get_tar_non(key)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-vs-Non-diffspkorphr EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)

        logging.info('Tar-vs-Non-samespk results')
        tar, non = scr.get_tar_non(key_nonsamespk)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-vs-Non-samespk EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)

        logging.info('Tar-vs-Non-samephr results')
        tar, non = scr.get_tar_non(key_nonsamephr)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-vs-Non-samephr EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)

        logging.info('Tar-vs-Non-diffspkphr results')
        tar, non = scr.get_tar_non(key_nondiffspkphr)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-vs-Non-diffspkphr EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)

        logging.info('Tar-samespk-vs-Non-diffspk results')
        tar, non = scr.get_tar_non(key_samespk)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-samespk-vs-Non-diffspk EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)

        logging.info('Tar-samephr-vs-Non-diffphr results')
        tar, non = scr.get_tar_non(key_samephr)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'Tar-samephr-vs-Non-diffphr EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
            eer * 100, min_dcf, act_dcf,
            len(tar), len(non))
        f.write(s)
        logging.info(s)



if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Computes EER and DCF')

    parser.add_argument('--key-file', dest='key_file', required=True)
    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)
        
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    score_dcf(**vars(args))


