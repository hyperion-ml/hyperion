"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains logistic regression calibration
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_scores import TrialScores


def train_verification_calibration(
    score_files, key_files, model_file, prior, lambda_reg, verbose
):
    assert len(score_files) == len(
        key_files
    ), f"{len(score_files)=} != {len(key_files)=}"
    tar = []
    non = []
    for score_file, key_file in zip(score_files, key_files):
        logging.info("load key: %s", key_file)
        key = TrialKey.load(key_file)
        logging.info("load scores: %s", score_file)
        scr = TrialScores.load(score_file)
        tar_i, non_i = scr.get_tar_non(key)
        tar.append(tar_i)
        non.append(non_i)

    tar = np.concatenate(tar, axis=0)
    non = np.concatenate(non, axis=0)
    ntar = len(tar)
    nnon = len(non)

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

    score_file = Path(score_file)
    output_file = score_file.with_suffix(f".cal{score_file.suffix}")
    scr_out = TrialScores(key.model_set, key.seg_set)
    scr_out.scores[key.tar] = tar_cal
    scr_out.scores[key.non] = non_cal
    scr_out.score_mask = np.logical_or(key.tar, key.non)
    scr_out.save(output_file)


def main():
    parser = ArgumentParser(description="Trains verification calibration")

    parser.add_argument("--score-files", nargs="+", required=True)
    parser.add_argument("--key-files", nargs="+", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_verification_calibration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
