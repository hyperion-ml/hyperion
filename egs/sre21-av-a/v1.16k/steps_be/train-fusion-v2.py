#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Trains calibration for SRE18 tel condition
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
from hyperion.classifiers import BinaryLogisticRegression as LR


def train_fusion_condition(
    key_file,
    scrs,
    system_names,
    prior,
    prior_eval,
    lambda_reg,
    solver,
    max_systems,
    verbose,
    cond,
):

    logging.info("load key: %s", key_file)
    key = TrialKey.load_txt(key_file)

    num_systems = len(scrs)
    tar = []
    non = []
    for i in range(num_systems):
        tar_i, non_i = scrs[i].get_tar_non(key)
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

    logging.info("fuse scores")
    tar_fus = gf.predict(tar)
    non_fus = gf.predict(non)
    for i in range(len(tar_fus)):
        min_dcf, _, _ = compute_min_dcf(tar_fus[i], non_fus[i], gf.prior)
        act_dcf, p_miss, p_fa = compute_act_dcf(tar_fus[i], non_fus[i], gf.prior)
        n_miss = p_miss * ntar
        n_fa = p_fa * nnon
        info_str = "(p=%.3f) min_dcf: %.3f act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f" % (
            gf.prior,
            min_dcf,
            act_dcf,
            p_miss * 100,
            p_fa * 100,
            n_miss,
            n_fa,
        )

        logging.info("%s Best-%d %s", cond, i + 1, info_str)

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

        logging.info("%s Best-%d %s", cond, i + 1, info_str)

    return gf, tar_fus, non_fus


def train_fusion(
    score_files,
    system_names,
    key_file,
    model_file,
    prior,
    prior_eval,
    lambda_reg,
    prior_postcal,
    lambda_reg_postcal,
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

    scrs = []
    for i in range(num_systems):
        logging.info("load scores: %s", score_files[i])
        scr_i = TrialScores.load_txt(score_files[i])
        scrs.append(scr_i)

    conds = ["CTS_CTS", "CTS_AFV", "AFV_AFV"]
    fusions = []
    tar_fus = []
    non_fus = []
    for i in range(len(conds)):
        fus_i, tar_fus_i, non_fus_i = train_fusion_condition(
            f"{key_file}_{conds[i]}",
            scrs,
            system_names,
            prior,
            prior_eval,
            lambda_reg,
            solver,
            max_systems,
            verbose,
            conds[i],
        )
        fusions.append(fus_i)
        tar_fus.append(tar_fus_i)
        non_fus.append(non_fus_i)

    for i in range(len(conds)):
        model_file_i = f"{model_file}_nopostcal_{conds[i]}.h5"
        logging.info("save calibration at %s", model_file_i)
        fusions[i].save(model_file_i)

    logging.info("train post-calibration")
    tar_fus = np.concatenate(tar_fus, axis=1)
    non_fus = np.concatenate(non_fus, axis=1)
    print("shape", tar_fus.shape, flush=True)
    ntar = tar_fus.shape[1]
    nnon = non_fus.shape[1]
    y = np.concatenate(
        (np.ones((ntar,), dtype="int32"), np.zeros((nnon,), dtype="int32"))
    )

    for i in range(tar_fus.shape[0]):
        x = np.concatenate((tar_fus[i], non_fus[i]), axis=0)
        lr = LR(
            prior=prior_postcal,
            lambda_reg=lambda_reg_postcal,
            bias_scaling=1,
            solver=solver,
            verbose=verbose,
        )
        lr.fit(x, y)
        logging.info(f"Best-{i} postcal scale={lr.A} bias={lr.b}")
        for j in range(len(conds)):
            logging.info(
                f"{conds[j]} Best-{i} fus weights={fusions[j].weights[i]} bias={fusions[j].bias[i]}"
            )
            fusions[j].weights[i] *= lr.A
            fusions[j].bias[i] = fusions[j].bias[i] * lr.A + lr.b
            logging.info(
                f"{conds[j]} Best-{i} fus+postcal weights={fusions[j].weights[i]} bias={fusions[j].bias[i]}"
            )

        act_dcf, p_miss, p_fa = compute_act_dcf(tar_fus[i], non_fus[i], prior_postcal)
        n_miss = p_miss * ntar
        n_fa = p_fa * nnon
        logging.info(
            "Best-%d before act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
            % (i, act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
        )
        scores_cal = lr.predict(x)
        tar_cal = scores_cal[y == 1]
        non_cal = scores_cal[y == 0]
        act_dcf, p_miss, p_fa = compute_act_dcf(tar_cal, non_cal, prior_postcal)
        n_miss = p_miss * ntar
        n_fa = p_fa * nnon
        logging.info(
            "Best-%d after act_dcf: %.3f p_miss: %.2f p_fa: %.2f n_miss: %.1f n_fa: %.1f"
            % (i, act_dcf, p_miss * 100, p_fa * 100, n_miss, n_fa)
        )

    for i in range(len(conds)):
        model_file_i = f"{model_file}_{conds[i]}.h5"
        logging.info("save calibration at %s", model_file_i)
        fusions[i].save(model_file_i)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Trains source dep. greedy binary logistic regression fusion"
    )

    parser.add_argument("--score-files", nargs="+", required=True)
    parser.add_argument("--system-names", nargs="+", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--prior-eval", type=float, nargs="+", default=None)
    parser.add_argument("--prior-postcal", type=float, default=0.01)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--lambda-reg-postcal", type=float, default=1e-4)
    parser.add_argument(
        "--solver",
        choices=["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        default="liblinear",
    )
    parser.add_argument("--max-systems", type=int, default=10)

    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)

    train_fusion(**namespace_to_dict(args))
