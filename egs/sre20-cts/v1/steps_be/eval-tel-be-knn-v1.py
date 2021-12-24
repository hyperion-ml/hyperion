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

from pathlib import Path

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList


def eval_plda_e(
    x_e,
    x_t,
    back_end_dir,
    enroll_id,
    preproc_basename,
    plda_basename,
    plda_type,
    pool_method,
):

    if preproc_basename is not None:
        preproc_file = Path(back_end_dir, enroll_id, preproc_basename)
        logging.info("loading preproc transform: %s" % (preproc_file))
        preproc = TransformList.load(preproc_file)
        x_e = preproc.predict(x_e)
        x_t = preproc.predict(x_t)

    ids_e = np.zeros((x_e.shape[0],), dtype=np.int)
    model_file = Path(back_end_dir, enroll_id, plda_basename)
    logging.info("loading plda model: %s" % (model_file))
    model = F.load_plda(plda_type, model_file)

    logging.info("computing llr")
    scores = model.llr_Nvs1(x_e, x_t, method=pool_method, ids1=ids_e)
    return scores


def eval_plda(
    iv_file,
    ndx_file,
    enroll_file,
    test_file,
    back_end_dir,
    preproc_basename,
    plda_basename,
    score_file,
    plda_type,
    pool_method,
    **kwargs
):

    logging.info("loading data")
    tdr = TDR(iv_file, ndx_file, enroll_file, test_file, None)
    x_e, x_t, enroll, ndx = tdr.read()
    enroll, ids_e = np.unique(enroll, return_inverse=True)

    t1 = time.time()
    scores = np.zeros((len(enroll), x_t.shape[0]), dtype=float_cpu())
    for i in range(len(enroll)):
        enroll_i = enroll[i]
        mask_i = ids_e == i
        x_e_i = x_e[mask_i]
        scores_i = eval_plda_e(
            x_e_i,
            x_t,
            back_end_dir,
            enroll_i,
            preproc_basename,
            plda_basename,
            plda_type,
            pool_method,
        )
        scores[i] = scores_i

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save_txt(score_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Eval PLDA with model adapted to the enrollment cuts",
    )

    parser.add_argument("--iv-file", required=True)
    parser.add_argument("--ndx-file", default=None)
    parser.add_argument("--enroll-file", required=True)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--preproc-basename", default=None)
    parser.add_argument("--plda-basename", default="plda.h5")
    parser.add_argument("--back-end-dir", required=True)
    parser.add_argument(
        "--plda-type",
        default="splda",
        choices=["frplda", "splda", "plda"],
        help=("PLDA type"),
    )
    TDR.add_argparse_args(parser)
    parser.add_argument(
        "--pool-method",
        dest="pool_method",
        type=str.lower,
        default="vavg-lnorm",
        choices=["book", "vavg", "vavg-lnorm", "savg"],
    )

    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    assert args.test_file is not None or args.ndx_file is not None
    eval_plda(**vars(args))
