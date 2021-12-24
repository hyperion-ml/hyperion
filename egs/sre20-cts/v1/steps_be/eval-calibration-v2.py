#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals calibration
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import BinaryLogisticRegression as LR
from hyperion.utils import Utt2Info


def eval_calibration(in_score_file, ndx_file, model_file, cond_file, out_score_file):

    logging.info("load ndx: %s" % ndx_file)
    try:
        ndx = TrialNdx.load_txt(ndx_file)
    except:
        ndx = TrialKey.load_txt(ndx_file)

    logging.info("load scores: %s" % in_score_file)
    scr = TrialScores.load_txt(in_score_file)
    scr = scr.align_with_ndx(ndx)

    enr2cond = Utt2Info.load(cond_file)
    conds, cond_ids = np.unique(
        enr2cond.filter(ndx.model_set).info, return_inverse=True
    )
    num_conds = len(conds)

    logging.info("apply calibration")
    scores_cal = np.zeros_like(scr.scores)
    for cond in range(num_conds):
        idx = cond_ids == cond
        model_file_cond = "{}-{}.h5".format(model_file, cond)
        logging.info("load model: %s" % model_file_cond)
        lr = LR.load(model_file_cond)
        s_cal = lr.predict(scr.scores.ravel())
        scores_cal_cond = np.reshape(s_cal, scr.scores.shape)
        scores_cal[idx] = scores_cal_cond[idx]

    scr.scores = scores_cal
    logging.info("save scores: %s" % out_score_file)
    scr.save_txt(out_score_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Evals linear calibration",
    )

    parser.add_argument("--in-score-file", dest="in_score_file", required=True)
    parser.add_argument("--out-score-file", dest="out_score_file", required=True)
    parser.add_argument("--ndx-file", dest="ndx_file", required=True)
    parser.add_argument("--cond-file", required=True)
    parser.add_argument("--model-file", dest="model_file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_calibration(**vars(args))
