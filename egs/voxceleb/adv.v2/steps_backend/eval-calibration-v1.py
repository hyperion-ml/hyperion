#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals calibration
"""

import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.np.classifiers import BinaryLogisticRegression as LR


def eval_calibration(in_score_file, ndx_file, model_file, out_score_file):

    logging.info("load ndx: %s" % ndx_file)
    try:
        ndx = TrialNdx.load_txt(ndx_file)
    except:
        ndx = TrialKey.load_txt(ndx_file)

    logging.info("load scores: %s" % in_score_file)
    scr = TrialScores.load_txt(in_score_file)
    scr = scr.align_with_ndx(ndx)

    logging.info("load model: %s" % model_file)
    lr = LR.load(model_file)
    logging.info("apply calibration")
    s_cal = lr.predict(scr.scores.ravel())
    scr.scores = np.reshape(s_cal, scr.scores.shape)

    logging.info("save scores: %s" % out_score_file)
    scr.save_txt(out_score_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evals linear calibration")

    parser.add_argument("--in-score-file", dest="in_score_file", required=True)
    parser.add_argument("--out-score-file", dest="out_score_file", required=True)
    parser.add_argument("--ndx-file", dest="ndx_file", required=True)
    parser.add_argument("--model-file", dest="model_file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_calibration(**namespace_to_dict(args))
