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

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval


def score_dcf(key_file, score_file, output_path):

    logging.info('Load key: %s' % key_file)
    key = TrialKey.load_txt(key_file)
    logging.info('Load scores: %s' % score_file)
    scr = TrialScores.load_txt(score_file)
    tar, non = scr.get_tar_non(key)

    priors = np.array([0.001, 0.005, 0.01, 0.05 ])
    min_dcf, act_dcf, eer, _ = fast_eval(tar, non, priors)
    
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = output_path + '_results'
    with open(output_file, 'w') as f:
        s = 'EER: {0:.2f} DCF5e-2: {1:.3f} / {2:.3f} DCF1e-2: {3:.3f} / {4:.3f} DCF5e-3: {5:.3f} / {6:.3f} DCF1e-3: {7:.3f} / {8:.3f}'.format(
            eer * 100, min_dcf[3], act_dcf[3],
            min_dcf[2], act_dcf[2],
            min_dcf[1], act_dcf[1],
            min_dcf[0], act_dcf[0])
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


