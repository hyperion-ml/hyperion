#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE18 tel condition
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.utils import Utt2Info
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import BinaryLogisticRegression as LR


def train_calibration_cond(cond, scr, key, model_file, prior, lambda_reg, verbose):

    tar, non = scr.get_tar_non(key)
    ntar = len(tar)
    nnon = len(non)
    min_dcf, p_miss, p_fa = compute_min_dcf(tar, non, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "cond %s min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (cond, min_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    logging.info("train calibration")
    x = np.concatenate((tar, non))
    y = np.concatenate(
        (np.ones((ntar,), dtype="int32"), np.zeros((nnon,), dtype="int32"))
    )
    lr = LR(
        prior=prior,
        lambda_reg=lambda_reg,
        bias_scaling=1,
        solver="liblinear",
        verbose=verbose,
    )
    lr.fit(x, y)
    model_file = "{}-{}.h5".format(model_file, cond)
    logging.info("save calibration at %s" % model_file)
    lr.save(model_file)

    logging.info("calibrate scores")
    tar_cal = lr.predict(tar)
    non_cal = lr.predict(non)
    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "cond %s act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (cond, act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    return tar_cal, non_cal


def train_calibration(
    score_file, key_file, model_file, cond_file, prior, lambda_reg, verbose
):

    logging.info("load key: %s" % key_file)
    key = TrialKey.load_txt(key_file)
    logging.info("load scores: %s" % score_file)
    scr = TrialScores.load_txt(score_file)
    tar, non = scr.get_tar_non(key)
    enr2cond = Utt2Info.load(cond_file)

    conds, cond_ids = np.unique(
        enr2cond.filter(key.model_set).info, return_inverse=True
    )
    num_conds = len(conds)
    ntar = len(tar)
    nnon = len(non)

    min_dcf, p_miss, p_fa = compute_min_dcf(tar, non, prior)
    del tar, non
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info("global result before calibration")
    logging.info(
        "min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (min_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    tar_cal = []
    non_cal = []
    for cond in range(num_conds):
        logging.info("train calibration cond %d" % (cond))
        model_set_cond = key.model_set[cond_ids == cond]
        key_cond = key.filter(model_set_cond, key.seg_set)
        tar_cal_cond, non_cal_cond = train_calibration_cond(
            conds[cond], scr, key_cond, model_file, prior, lambda_reg, verbose
        )

        tar_cal.append(tar_cal_cond)
        non_cal.append(non_cal_cond)

    tar_cal = np.concatenate(tuple(tar_cal), axis=-1)
    non_cal = np.concatenate(tuple(non_cal), axis=-1)
    logging.info("global result after calibration")
    min_dcf, p_miss, p_fa = compute_min_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (min_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )

    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
        % (act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Trains llr calibration with multiple enrollment conditions",
    )

    parser.add_argument("--score-file", dest="score_file", required=True)
    parser.add_argument("--key-file", dest="key_file", required=True)
    parser.add_argument("--model-file", dest="model_file", required=True)
    parser.add_argument("--cond-file", required=True)
    parser.add_argument("--prior", dest="prior", type=float, default=0.01)
    parser.add_argument("--lambda-reg", dest="lambda_reg", type=float, default=1e-5)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_calibration(**vars(args))
