#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
"""

import sys
import os
import argparse
import time
import logging

import numpy as np
from scipy.special import logsumexp

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import TrialNdx, TrialScores, Utt2Info
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList
from hyperion.io import RandomAccessDataReaderFactory as DRF


def eval_plda(
    iv_file,
    ndx_file,
    enroll_file,
    preproc_file,
    model_file,
    score_file,
    plda_type,
    eval_method,
    **kwargs
):

    logging.info("loading data")
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    logging.info("loading data")
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    u2s = Utt2Info.load(enroll_file)
    ndx = TrialNdx.load(ndx_file)
    reader = DRF.create(iv_file)
    x_t = reader.read(ndx.seg_set, squeeze=True)
    x_e = reader.read(u2s.key, squeeze=True)
    enroll, ids_e = np.unique(u2s.info, return_inverse=True)

    logging.info("loading plda model: %s" % (model_file))
    model = F.load_plda(plda_type, model_file)

    t1 = time.time()
    logging.info("computing llr")
    x_e = preproc.predict(x_e)
    x_t = preproc.predict(x_t)
    scores = model.llr_Nvs1(x_e, x_t, ids1=ids_e, method=eval_method)
    scores = -logsumexp(scores, axis=0, keepdims=True) + len(enroll)

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )
    model_set = ["known"]
    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(model_set, ndx.seg_set, scores)
    s.save_txt(score_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Eval PLDA",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--ndx-file", dest="ndx_file", default=None)
    parser.add_argument("--enroll-file", dest="enroll_file", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)
    parser.add_argument(
        "--eval-method", choices=["book", "vavg-lnorm"], default="vavg-lnorm"
    )

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)

    parser.add_argument("--score-file", dest="score_file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_plda(**vars(args))
