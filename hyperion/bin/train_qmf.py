#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE18 tel condition
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

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


def print_q_stats(scr: TrialScores, q_names: Sequence[str]) -> None:
    """Log summary statistics for selected quality-measure vectors.

    Args:
        scr: Trial scores object containing quality measures.
        q_names: Names of quality-measure entries to summarize.
    """
    for k in q_names:
        q_vec = scr.q_measures[k][scr.score_mask]
        s = f"{k} stats mean={np.mean(q_vec)} min={np.min(q_vec)} max={np.max(q_vec)} median={np.median(q_vec)}"
        logging.info(s)


def train_qmf(
    score_file: str,
    key_file: str,
    model_file: str,
    prior: float,
    lambda_reg: float,
    quality_measures: Optional[List[str]],
    verbose: int,
) -> None:
    """Train and apply a QMF calibration model.

    Args:
        score_file: Input score file with quality measures.
        key_file: Trial key file with target/non-target labels.
        model_file: Output path for trained logistic-regression model.
        prior: Target prior probability used for DCF and calibration.
        lambda_reg: L2 regularization for logistic regression.
        quality_measures: Optional list of quality-measure names to use.
        verbose: Verbosity level forwarded to logistic-regression solver.
    """
    logging.info("load key: %s", key_file)
    key = TrialKey.load(key_file)
    logging.info("load scores: %s", score_file)
    scr = TrialScores.load(score_file)
    tar, non = scr.get_tar_non(key)
    ntar = len(tar)
    nnon = len(non)

    if quality_measures is None:
        quality_measures = list(scr.q_measures.keys())
        quality_measures.sort()

    print_q_stats(scr, quality_measures)
    q_tar, q_non = scr.get_tar_non_q_measures(key, quality_measures)

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
    # tar = np.vstack((tar, maxnf_tar, minnf_tar, maxcohmu_tar, mincohmu_tar)).T
    # non = np.vstack((non, maxnf_non, minnf_non, maxcohmu_non, mincohmu_non)).T
    tar = np.hstack((tar[:, None], q_tar))
    non = np.hstack((non[:, None], q_non))

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

    score_file = Path(score_file)
    output_file = score_file.with_suffix(f".qmf{score_file.suffix}")
    scr_out = TrialScores(key.model_set, key.seg_set)
    scr_out.scores[key.tar] = tar_cal
    scr_out.scores[key.non] = non_cal
    scr_out.score_mask = np.logical_or(key.tar, key.non)
    scr_out.save(output_file)


def main() -> None:
    """Parse CLI arguments and train a QMF calibration model."""
    parser = ArgumentParser(description="Trains QMF calibration")

    parser.add_argument(
        "--score-file",
        required=True,
        help="Input score file containing scores and quality measures.",
    )
    parser.add_argument(
        "--key-file",
        required=True,
        help="Trial key file with target/non-target labels.",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Output path for trained QMF calibration model.",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=0.01,
        help="Target prior probability used in calibration and DCF reporting.",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-5,
        help="L2 regularization coefficient for logistic regression.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=error, 1=warning, 2=info, 3=debug.",
    )
    parser.add_argument(
        "--quality-measures",
        default=None,
        nargs="+",
        choices=["snorm-mu/s", "snorm-mu", "speech_duration", "num_speech_frames"],
        help="Optional subset of quality measures used by QMF calibration.",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_qmf(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
