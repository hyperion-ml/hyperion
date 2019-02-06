#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

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
import logging

import numpy as np

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.transforms import TransformList
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.tensors import to3D_by_class
from hyperion.helpers import TrialDataReader as TDR
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedVAE_qYqZgY as TVAEYZ
from hyperion.keras.vae import TiedVAE_qY as TVAEY



def eval_pdda(iv_file, ndx_file, enroll_file, test_file,
              preproc_file,
              model_file, score_file,
              pool_method, eval_method,
              num_samples_y, num_samples_z, num_samples_elbo, qy_only,
              **kwargs):

    set_float_cpu('float32')
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr_args = TDR.filter_args(**kwargs)
    tdr = TDR(iv_file, ndx_file, enroll_file, test_file, preproc, **tdt_args)
    x_e, x_t, enroll, ndx = tdr.read()
    enroll, ids_e = np.unique(enroll, return_inverse=True)

    if qy_only:
        model = TVAEY.load(model_file)
        model.build(max_seq_length=2, num_samples=num_samples_y)
    else:
        model = TVAEYZ.load(model_file)
        model.build(max_seq_length=2,
                    num_samples_y=num_samples_y, num_samples_z=num_samples_z)

    t1 = time.time()
    scores = model.eval_llr_Nvs1(x_e, ids_e, x_t,
                                 pool_method=pool_method,
                                 eval_method=eval_method,
                                 num_samples=num_samples_elbo)
    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info('Elapsed time: %.2f s. Elapsed time per trial: %.2f ms.' %
                 (dt, dt/num_trials*1000))

    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save(score_file)


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Eval PDDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-file', dest='test_file', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    TDR.add_argparse_args(parser)

    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--score-file', dest='score_file', required=True)
    parser.add_argument('--qy-only', dest='qy_only',
                        default=False, action='store_true')

    parser.add_argument('--pool-method', dest='pool_method', type=str.lower,
                        default='vavg-lnorm',
                        choices=['book','vavg','vavg-lnorm','savg'],
                        help=('(default: %(default)s)'))

    parser.add_argument('--eval-method', dest='eval_method', type=str.lower,
                        default='qscr',
                        choices=['elbo','cand','qscr'],
                        help=('(default: %(default)s)'))

    parser.add_argument('--nb-samples-elbo', dest='num_samples_elbo', default=1, type=int)
    parser.add_argument('--nb-samples-y', dest='num_samples_y', default=1, type=int)
    parser.add_argument('--nb-samples-z', dest='num_samples_z', default=1, type=int)
parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
        
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    assert args.test_file is not None or args.ndx_file is not None
    eval_pdda(**vars(args))

            
