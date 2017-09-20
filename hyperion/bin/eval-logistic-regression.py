#!/usr/bin/env python
"""
Evals PDDA LLR
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

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.io import HypDataWriter as HDW
from hyperion.helpers import ClassifTrialDataReader as TDR
from hyperion.transforms import TransformList
from hyperion.classifiers import LogisticRegression as LR


def eval_lr(iv_file, class2int_file, test_file,
            preproc_file,
            model_file, score_file, vector_score_file,
            eval_type, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr_args = TDR.filter_args(**kwargs)
    tdr = TDR(iv_file, class2int_file, test_file, preproc, **tdr_args)
    x, ndx = tdr.read()

    model = LR.load(model_file)
    
    t1 = time.time()
    scores = model.predict(x, eval_type)
    
    dt = time.time() - t1
    num_trials = scores.shape[0]*scores.shape[1]
    print('Elapsed time: %.2f s. Elapsed time per trial: %.2f ms.'
          % (dt, dt/num_trials*1000))

    s = TrialScores(ndx.model_set, ndx.seg_set, scores.T)
    s.save(score_file)

    if vector_score_file is not None:
        h5 = HDW(vector_score_file)
        h5.write(ndx.seg_set, '', scores)

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval linear logistic regression classifier')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--class2int-file', dest='class2int_file', required=True)
    parser.add_argument('--test-file', dest='test_file', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    
    TDR.add_argparse_args(parser)
    LR.add_argparse_eval_args(parser)
    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--vector-score-file', dest='vector_score_file', default=None)
    
    args=parser.parse_args()
    print(args)
    eval_lr(**vars(args))

            
