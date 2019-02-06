#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Evals EER, DCF, DET
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

from hyperion.hyp_defs import config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import compute_eer

def eval_2class_performance(score_file, key_file, output_path):

    scr = TrialScores.load(score_file)
    key = TrialKey.load(key_file)

    output_dir = os.path.dirname(output_path)
    if not(os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
    
    tar, non = scr.get_tar_non(key)
    eer = compute_eer(tar, non)

    output_file=output_path + '.res'
    with open(output_file, 'w') as f:
        f.write('EER %.4f\nNTAR %d\nNNON %d\n'
                % (eer, len(tar), len(non)))
    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Evals EER, DCF, DET')

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--key-file', dest='key_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    eval_2class_performance(**vars(args))
