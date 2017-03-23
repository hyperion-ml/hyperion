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
from hyperion.io import HypDataReader
from hyperion.transforms import TransformList
from hyperion.utils.scp_list import SCPList
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.keras.keras_utils import *
from hyperion.distributions.plda import FRPLDA


def load_data(hyp_reader, ndx_file, enroll_file, test_file,
              preproc,
              model_idx, num_model_parts, seg_idx, num_seg_parts,
              eval_set):

    enroll = SCPList.load(enroll_file, sep='=')
    test = None
    if test_file is not None:
        test = SCPList.load(test_file, sep='=')
    ndx = None
    if ndx_file is not None:
        ndx = TrialNdx.load(ndx_file)

    ndx, enroll = TrialNdx.parse_eval_set(ndx, enroll, test, eval_set)
    if num_model_parts > 1 or num_seg_parts > 1:
        ndx = TrialNdx.split(model_idx, num_model_parts, seg_idx, num_seg_parts)
        enroll = enroll.filter(ndx.key)
        
    x_e = hyp_reader.read(enroll.file_path, '.ivec')
    x_t = hyp_reader.read(ndx.seg_set, '.ivec')
    
    if preproc is not None:
        x_e = preproc.predict(x_e)
        x_t = preproc.predict(x_t)

    return x_e, x_t, ndx


def eval_plda(iv_file, ndx_file, enroll_file, test_file,
              preproc_file, model_file, score_file, plda_type, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    hr = HypDataReader(iv_file)
    x_e, x_t, ndx = load_data(hr, ndx_file, enroll_file, test_file, preproc, **kwargs)

    if plda_type == 'frplda':
        model = FRPLDA.load(model_file)
    
    t1 = time.time()
    scores = model.eval_llr_1vs1(x_e, x_t)
    
    dt = time.time() - t1
    num_trials = x_e.shape[0] * x_t.shape[0]
    print('Elapsed time: %.2f s. Elapsed time per trial: %.2f ms.' % (dt, dt/num_trials*1000))

    s = TrialScores(ndx.model_set, ndx.seg_set, scores)
    s.save(score_file)

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Eval PLDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-file', dest='test_file', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--plda-type', dest='plda_type', default='splda',
                        type=str.lower,
                        choices=['plda','splda','frplda'],
                        help=('(default: %(default)s)'))
    
    parser.add_argument('--model-part-idx', dest='model_idx', default=1, type=int)
    parser.add_argument('--num-model-parts', dest='num_model_parts', default=1, type=int)
    parser.add_argument('--seg-part-idx', dest='seg_idx', default=1, type=int)
    parser.add_argument('--num-seg-parts', dest='num_seg_parts', default=1, type=int)

    parser.add_argument('--eval-set', dest='eval_set', type=str.lower,
                        default='enroll-test',
                        choices=['enroll-test','enroll-coh','coh-test','coh-coh'],
                        help=('(default: %(default)s)'))
    
    args=parser.parse_args()

    assert(args.test_file is not None or args.ndx_file is not None)
    eval_plda(**vars(args))

            
