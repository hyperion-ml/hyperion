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
from hyperion.utils.scp_list import SCPList
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.helpers import TrialDataReader as TDR
from hyperion.transforms import TransformList, LNorm



def eval_cos(iv_file, ndx_file, enroll_file, test_file,
             preproc_file,
             scp_sep, v_field, eval_set, score_file, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr = TDR(iv_file, ndx_file, enroll_file, test_file, preproc,
              scp_sep=scp_sep, v_field=v_field, eval_set=eval_set)
    x_e, x_t, enroll, ndx = tdr.read()

    lnorm = LNorm()
    x_e = lnorm.predict(x_e)
    x_t = lnorm.predict(x_t)
    
    t1 = time.time()
    scores = np.dot(x_e, x_t.T)
    
    dt = time.time() - t1
    num_trials = x_e.shape[0] * x_t.shape[0]
    print('Elapsed time: %.2f s. Elapsed time per trial: %.2f ms.'
          % (dt, dt/num_trials*1000))

    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save(score_file)

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Eval Cos')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-file', dest='test_file', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    TDR.add_argparse_args(parser)
    parser.add_argument('--score-file', dest='score_file', required=True)
    
    args=parser.parse_args()

    assert(args.test_file is not None or args.ndx_file is not None)
    eval_cos(**vars(args))

            
