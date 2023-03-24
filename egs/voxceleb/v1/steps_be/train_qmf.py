#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE18 tel condition
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
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.np.classifiers import BinaryLogisticRegression as LR


def print_q_stats(q, name):
    scores = q.scores[q.score_mask]
    s = f"{name} stats mean={np.mean(scores)} min={np.min(scores)} max={np.max(scores)} median={np.median(scores)}"
    logging.info(s)


def train_calibration(score_file, key_file, model_file, prior, lambda_reg, verbose):

    logging.info("load key: %s", key_file)
    key = TrialKey.load_txt(key_file)
    score_snorm_file = f"{score_file}_snorm"
    logging.info("load scores: %s", score_snorm_file)
    scr = TrialScores.load_txt(score_snorm_file)
    tar, non = scr.get_tar_non(key)
    ntar = len(tar)
    nnon = len(non)

    q_file = f"{score_file}_maxnf"
    logging.info("load max num-frames: %s", q_file)
    q = TrialScores.load_txt(q_file)
    print_q_stats(q, "max-nf")
    maxnf_tar, maxnf_non = q.get_tar_non(key)

    q_file = f"{score_file}_minnf"
    logging.info("load min num-frames: %s", q_file)
    q = TrialScores.load_txt(q_file)
    print_q_stats(q, "min-nf")
    minnf_tar, minnf_non = q.get_tar_non(key)

    q_file = f"{score_file}_maxcohmu"
    logging.info("load max cohort mean: %s", q_file)
    q = TrialScores.load_txt(q_file)
    print_q_stats(q, "max-cohmu")
    maxcohmu_tar, maxcohmu_non = q.get_tar_non(key)

    q_file = f"{score_file}_mincohmu"
    logging.info("load min cohort mean: %s", q_file)
    q = TrialScores.load_txt(q_file)
    print_q_stats(q, "min-cohmu")
    mincohmu_tar, mincohmu_non = q.get_tar_non(key)

    min_dcf, p_miss, p_fa = compute_min_dcf(tar, non, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "min_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f",
        min_dcf,
        p_miss * 100,
        p_fa * 100,
        n_miss,
        n_fa,
    )

    logging.info("train calibration")
    tar = np.vstack((tar, maxnf_tar, minnf_tar, maxcohmu_tar, mincohmu_tar)).T
    non = np.vstack((non, maxnf_non, minnf_non, maxcohmu_non, mincohmu_non)).T

    x = np.vstack((tar, non))
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
    logging.info(f"A={lr.A} b={lr.b}")
    logging.info("save calibration at %s", model_file)
    lr.save(model_file)

    logging.info("calibrate scores")
    tar_cal = lr.predict(tar)
    non_cal = lr.predict(non)
    act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior)
    n_miss = p_miss * ntar
    n_fa = p_fa * nnon
    logging.info(
        "act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f",
        act_dcf,
        p_miss * 100,
        p_fa * 100,
        n_miss,
        n_fa,
    )

    output_file = f"{score_file}_qmf"
    scr_out = TrialScores(key.model_set, key.seg_set)
    scr_out.scores[key.tar] = tar_cal
    scr_out.scores[key.non] = non_cal
    scr_out.score_mask = np.logical_or(key.tar, key.non)
    scr_out.save(output_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Trains QMF calibration")

    parser.add_argument("--score-file", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_calibration(**namespace_to_dict(args))
