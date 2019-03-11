#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE18 tel condition
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
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import BinaryLogisticRegression as LR


def train_calibration(score_file, key_file, model_file, prior, verbose):

    logging.info('load key: %s' % key_file)
    key = TrialKey.load_txt(key_file)
    logging.info('load scores: %s' % score_file)
    scr = TrialScores.load_txt(score_file)
    tar, non = scr.get_tar_non(key)
    ntar = len(tar)
    nnon = len(non)

    min_dcf, p_miss, p_fa = compute_min_dcf(tar, non, prior)
    n_miss = p_miss*ntar
    n_fa = p_fa*nnon
    logging.info('min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f' % 
                 (min_dcf, p_miss*100, p_fa*100, n_miss, n_fa))

    logging.info('train calibration')
    x = np.concatenate((tar, non))
    y = np.concatenate((np.ones((ntar,), dtype='int32'),
                        np.zeros((nnon,), dtype='int32')))
    lr = LR(prior=prior, verbose=verbose)
    lr.fit(x, y)
    print(lr.A)
    print(lr.b)
    logging.info('save calibration at %s' % model_file)
    lr.save(model_file)

    logging.info('calibrate scores')
    tar_cal = lr.predict(tar)
    non_cal = lr.predict(non)
    print(tar_cal)
    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss*ntar
    n_fa = p_fa*nnon
    logging.info('act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f' % 
                 (act_dcf, p_miss*100, p_fa*100, n_miss, n_fa))

    
        
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Trains llr calibration')

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--key-file', dest='key_file', required=True)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--prior', dest='prior', type=float,
                        default=0.01)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)
    
    train_calibration(**vars(args))

            
