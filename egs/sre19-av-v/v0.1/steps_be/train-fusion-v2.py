#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE21 face reco condition
"""

import sys
import os
from jsonargparse import ArgumentParser, namespace_to_dict
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import GreedyFusionBinaryLR as GF


def train_fusion(
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

    logging.info("load key: %s", key_file)
    key = TrialKey.load_txt(key_file)

    tar = []
    non = []
    for i in range(num_systems):
        logging.info("load scores: %s", score_files[i])
        scr = TrialScores.load_txt(score_files[i])
        tar_i, non_i = scr.get_tar_non(key)
        tar.append(tar_i[:, None])
        non.append(non_i[:, None])

    tar = np.concatenate(tuple(tar), axis=1)
    non = np.concatenate(tuple(non), axis=1)
    ntar = tar.shape[0]
    nnon = non.shape[0]
    n_extra = int(1e6)
    mu_tar = np.mean(tar, axis=0)
    cov_tar = np.cov(tar, rowvar=False)
    mu_non = np.mean(non, axis=0)
    cov_non = np.cov(non, rowvar=False)
    tar_extra = np.random.multivariate_normal(mu_tar, cov_tar, n_extra)
    non_extra = np.random.multivariate_normal(mu_non, cov_non, n_extra)

    logging.info("train fusion")
    x = np.concatenate((tar, tar_extra, non, non_extra), axis=0)
    y = np.concatenate(
        (
            np.ones((ntar + n_extra,), dtype="int32"),
            np.zeros((nnon + n_extra,), dtype="int32"),
        )
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
    logging.info("save calibration at %s", model_file)
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
            info_str = "%s (p=%.3f) min_dcf: %.3f act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f" % (
                info_str,
                gf.prior_eval[j],
                min_dcf[j],
                act_dcf[j],
                p_miss[j] * 100,
                p_fa[j] * 100,
                n_miss,
                n_fa,
            )

        logging.info("Best-%d %s", i + 1, info_str)


if __name__ == "__main__":

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
        dest="solver",
        choices=["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        default="liblinear",
    )
    parser.add_argument("--max-systems", type=int, default=10)

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_fusion(**namespace_to_dict(args))
