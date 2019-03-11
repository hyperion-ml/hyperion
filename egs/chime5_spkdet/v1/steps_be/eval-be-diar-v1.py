#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
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
from hyperion.utils.scp_list import SCPList
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList


def combine_diar_scores(ndx, diar_ndx, diar2orig, diar_scores):

    d2o = SCPList.load(diar2orig, sep=' ')
    d2o = d2o.filter(diar_ndx.seg_set)
    scores = np.zeros(ndx.trial_mask.shape, dtype=float_cpu())
    for j in xrange(len(ndx.seg_set)):
        idx = d2o.file_path == ndx.seg_set[j]
        diar_scores_j = diar_scores[:, idx]
        scores_j = np.max(diar_scores_j, axis=1)
        scores[:,j] = scores_j

    return scores


def eval_plda(iv_file, ndx_file, diar_ndx_file, enroll_file, diar2orig,
              preproc_file,
              model_file, score_file, plda_type,
              **kwargs):
    
    logging.info('loading data')
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr = TDR(iv_file, diar_ndx_file, enroll_file, None, preproc)
    x_e, x_t, enroll, diar_ndx = tdr.read()

    logging.info('loading plda model: %s' % (model_file))
    model = F.load_plda(plda_type, model_file)
    
    t1 = time.time()
    
    logging.info('computing llr')
    scores = model.llr_1vs1(x_e, x_t)
    
    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info('scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.'
          % (dt, dt/num_trials*1000))

    logging.info('combine cluster scores') 
    ndx = TrialNdx.load(ndx_file)
    scores = combine_diar_scores(ndx, diar_ndx, diar2orig, scores)

    logging.info('saving scores to %s' % (score_file))
    s = TrialScores(enroll, ndx.seg_set, scores)
    s = s.align_with_ndx(ndx)
    s.save_txt(score_file)

    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval PLDA with diarization in test')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', required=True)
    parser.add_argument('--diar-ndx-file', dest='diar_ndx_file', required=True)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--diar2orig', dest='diar2orig', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)

    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_plda(**vars(args))

            
