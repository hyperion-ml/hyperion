#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Score Normalization
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
from hyperion.score_norm import *
from hyperion.utils.trial_scroes import TrialScores
from hyperion.utils.trial_ndx import TrialNdx



def load_scores(score_file, enr_coh_file, coh_test_file, coh_coh_file):

    scores = TrialScores.load(score_file)
    scores_enr_coh = None
    scores_coh_test = None
    scores_coh_coh = None
    
    if enr_coh_file is not None:
        ndx = TrialNdx(scores.model_set, scores_enr_coh.seg_set)
        scores_enr_coh = TrialScores.load(enr_coh_file)
        scores_enr_coh = scores_enr_coh.align_with_ndx(ndx)

    if coh_test_file is not None:
        ndx = TrialNdx(scores_coh_test.model_set, scores.seg_set)
        scores_coh_test = TrialScores.load(coh_test_file)
        scores_coh_test = scores_coh_test.align_with_ndx(ndx)

    if coh_coh_file is not None:
        assert scores_enr_coh is not None and scores_coh_test is not None
        ndx = TrialNdx(scores_coh_test.model_set, scores_enr_coh.seg_set)
        scores_coh_coh = TrialScores.load(coh_coh_file)
        scores_coh_coh = scores_coh_coh.align_with_ndx(ndx)

    return scores, scores_enr_coh, scores_coh_test, scores_coh_coh



def score_norm(score_file, ouput_file, norm_type,
               enr_coh_file=None, coh_test_file=None, coh_coh_file=None,
               adapt_coh=None):

    scores, scores_enr_coh, scores_coh_test, scores_coh_coh = load_scores(
        score_file, enr_coh_file, coh_test_file, coh_coh_file)

    if norm_type == 't_norm':
        assert scores_coh_test is not None
        norm = TNorm()
        scores_norm = norm.predict(scores.scores, scores_coh_test.scores)

    if norm_type == 'z_norm':
        assert scores_enr_coh is not None
        norm = ZNorm()
        scores_norm = norm.predict(scores.scores, scores_enr_coh.scores)

    if norm_type == 'zt_norm':
        assert(scores_enr_coh is not None and scores_coh_test is not None
               and scores_coh_coh is not None)
        norm = ZTNorm()
        scores_norm = norm.predict(scores.scores,
                                   scores_coh_test.scores,
                                   scores_enr_coh.scores,
                                   scores_coh_coh.scores)

    if norm_type == 's_norm':
        assert(scores_enr_coh is not None and scores_coh_test is not None
               and scores_coh_coh is not None)
        norm = SNorm()
        scores_norm = norm.predict(scores.scores,
                                   scores_coh_test.scores,
                                   scores_enr_coh.scores,
                                   scores_coh_coh.scores)

    scores.scores = scores_norm
    scores.save(ouput_file)
    

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval score normalization')

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--output-file', dest='output_file', required=True)
    parser.add_argument('--enr-coh-file', dest='enr_coh_file', default=None)
    parser.add_argument('--coh-test-file', dest='coh_test_file', default=None)
    parser.add_argument('--coh-coh-file', dest='coh_coh_file', default=None)
    parser.add_argument('--norm-type', dest='norm_type', default='s-norm',
                        choices=['t-norm', 'z-norm', 'zt-norm', 's-norm'])
    parser.add_argument('--adapt-coh', dest='adapt_coh', default=None, type=int)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    eval_score_norm(**vars(args))

    
