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

import numpy as np
from jsonargparse import ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.np.classifiers import GreedyFusionBinaryLR as GF
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_scores import TrialScores


def train_verification_greedy_fusion(
    score_files,
    system_names,
    key_file,
    model_file,
    prior,
    prior_eval,
    lambda_reg,
    solver,
    max_systems,
    verbose,
):
    num_systems = len(score_files)
    assert num_systems == len(
        system_names
    ), "len(score_files)(%d) != len(system_names)(%d)" % (
        num_systems,
        len(system_names),
    )
    if prior_eval is None:
        prior_eval = [prior]

    logging.info("load key: %s" % key_file)
    key = TrialKey.load(key_file)

    tar = []
    non = []
    for i in range(num_systems):
        logging.info("load scores: %s" % score_files[i])
        scr = TrialScores.load(score_files[i])
        tar_i, non_i = scr.get_tar_non(key)
        tar.append(tar_i[:, None])
        non.append(non_i[:, None])

    tar = np.concatenate(tuple(tar), axis=1)
    non = np.concatenate(tuple(non), axis=1)
    ntar = tar.shape[0]
    nnon = non.shape[0]

    logging.info("train fusion")
    x = np.concatenate((tar, non), axis=0)
    y = np.concatenate(
        (np.ones((ntar,), dtype="int32"), np.zeros((nnon,), dtype="int32"))
    )
    gf = GF(
        system_names=system_names,
        prior=prior,
        prior_eval=prior_eval,
        lambda_reg=lambda_reg,
        solver=solver,
        max_systems=max_systems,
        verbose=verbose,
    )
    gf.fit(x, y)
    logging.info("save calibration at %s" % model_file)
    gf.save(model_file)

    logging.info("fuse scores")
    tar_fus = gf.predict(tar)
    non_fus = gf.predict(non)
    for i in range(len(tar_fus)):
        min_dcf, _, _ = compute_min_dcf(tar_fus[i], non_fus[i], gf.prior_eval)
        act_dcf, p_miss, p_fa = compute_act_dcf(tar_fus[i], non_fus[i], gf.prior_eval)
        if len(gf.prior_eval) == 1:
            min_dcf = min_dcf[None]
            act_dcf = act_dcf[None]
            p_miss = p_miss[None]
            p_fa = p_fa[None]

        info_str = ""
        for j in range(len(gf.prior_eval)):
            n_miss = p_miss[j] * ntar
            n_fa = p_fa[j] * nnon
            info_str = (
                "%s (p=%.3f) min_dcf: %.3f act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
                % (
                    info_str,
                    gf.prior_eval[j],
                    min_dcf[j],
                    act_dcf[j],
                    p_miss[j] * 100,
                    p_fa[j] * 100,
                    n_miss,
                    n_fa,
                )
            )

        logging.info("Best-%d %s" % (i + 1, info_str))


def main():
    parser = ArgumentParser(
        description="Trains greedy binary logistic regression fusion"
    )

    parser.add_argument("--score-files", nargs="+", required=True)
    parser.add_argument("--system-names", nargs="+", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--prior-eval", type=float, nargs="+", default=None)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument(
        "--solver",
        choices=["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        default="liblinear",
    )
    parser.add_argument("--max-systems", type=int, default=10)

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_verification_greedy_fusion(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
