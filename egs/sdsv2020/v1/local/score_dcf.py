#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import SparseTrialScores, SparseTrialKey
from hyperion.utils import TrialScores, TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval
from hyperion.metrics import effective_prior


def score_dcf(key_file, score_file, output_path):

    logging.info('Load key: %s' % key_file)
    #key = SparseTrialKey.load_txt(key_file)
    key = TrialKey.load_txt(key_file)

    logging.info('Load scores: %s' % score_file)
    #scr = SparseTrialScores.load_txt(score_file)
    scr = TrialScores.load_txt(score_file)

    prior = effective_prior(0.01, 10, 1)

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = output_path + '_results'
    with open(output_file, 'w') as f:

        logging.info('Compute EER, DCF')
        tar, non = scr.get_tar_non(key)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, prior)
        s = 'EER: {0:.2f} DCF: {1:.3f} / {2:.3f} ntar: {3:d} nnon: {4:d}\n'.format(
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


