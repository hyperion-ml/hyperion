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
from hyperion.utils import SCPList, TrialNdx, TrialScores, ExtSegmentList, RTTM
from hyperion.helpers.tracking_data_reader import TrackingDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList


def classify_segments(ndx_seg, scores):

    scores = scores.align_with_ndx(ndx_seg)
    scores.scores[ndx_seg.trial_mask==False] = - np.inf
    pred_model_idx = np.argmax(scores.scores, axis=0)
    pred_scores = scores.scores[(pred_model_idx, np.arange(scores.num_tests))]
    pred_model = scores.model_set[pred_model_idx]

    return ndx_seg.seg_set, pred_model, pred_scores



def tracking_plda(iv_file, ndx_file, enroll_file, segments_file,
              preproc_file,
              model_file, rttm_file, plda_type,
              **kwargs):
    
    logging.info('loading data')
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr = TDR(iv_file, ndx_file, enroll_file, segments_file, preproc)
    x_e, x_t, enroll, ndx_seg, ext_segments = tdr.read()

    logging.info('loading plda model: %s' % (model_file))
    model = F.load_plda(plda_type, model_file)
    
    t1 = time.time()
    
    logging.info('computing llr')
    scores = model.llr_1vs1(x_e, x_t)
    
    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info('scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.'
          % (dt, dt/num_trials*1000))

    scores = TrialScores(enroll, ndx_seg.seg_set, scores)
    ext_segment_ids, pred_model, pred_scores = classify_segments(ndx_seg, scores)
    ext_segments.assign_names(ext_segment_ids, pred_model, pred_scores)
    print(pred_model)
    print(ext_segments.ext_segments)
    rttm = RTTM.create_spkdiar_from_ext_segments(ext_segments)
    rttm.save(rttm_file)

    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval PLDA for tracking spks')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', required=True)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--segments-file', dest='segments_file', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)

    parser.add_argument('--rttm-file', dest='rttm_file', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    tracking_plda(**vars(args))

            
