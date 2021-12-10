#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals greedy fusion
"""
import sys
import os
from jsonargparse import ArgumentParser, namespace_to_dict
import time
import logging

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx


def eval_fusion_sanity_dataset(score_files, ndx_file):
    logging.info("load ndx: %s" % ndx_file)
    try:
        ndx = TrialNdx.load_txt(ndx_file)
    except:
        ndx = TrialKey.load_txt(ndx_file).to_ndx()

    num_systems = len(score_files)
    in_scores = []
    for i in range(num_systems):
        logging.info("load scores: %s", score_files[i])
        scr = TrialScores.load_txt(score_files[i])
        scr = scr.align_with_ndx(ndx)
        in_scores.append(scr.scores[ndx.trial_mask][None, :])

    in_scores = np.concatenate(tuple(in_scores), axis=0)
    print(in_scores.shape)
    R = np.dot(in_scores, in_scores.T) / in_scores.shape[1]
    norms = 1 / np.sqrt(np.diag(R))
    R = R * norms
    R = R * norms[:, None]

    idx = np.argsort(in_scores[0])
    sort_scores = in_scores[:, idx]

    return R, sort_scores


def eval_fusion_sanity(
    system_names,
    score_files_dev,
    score_files_eval,
    ndx_file_dev,
    ndx_file_eval,
    output_path,
):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    R_dev, s_dev = eval_fusion_sanity_dataset(score_files_dev, ndx_file_dev)
    R_eval, s_eval = eval_fusion_sanity_dataset(score_files_eval, ndx_file_eval)
    df_dev = pd.DataFrame(data=R_dev, index=system_names, columns=system_names)
    df_dev.to_csv(f"{output_path}/r_dev.csv", sep="\t")
    df_eval = pd.DataFrame(data=R_eval, index=system_names, columns=system_names)
    df_eval.to_csv(f"{output_path}/r_eval.csv", sep="\t")
    logging.info(f"R-dev={R_dev}")
    logging.info(f"R-eval={R_eval}")
    ratio = R_eval / R_dev
    df_ratio = pd.DataFrame(data=ratio, index=system_names, columns=system_names)
    df_ratio.to_csv(f"{output_path}/ratio.csv", sep="\t")
    logging.info(f"R-eval/R-dev={ratio}")

    df = pd.DataFrame(data=s_dev.T, columns=system_names).to_csv(
        f"{output_path}/sort_scores_dev.csv"
    )
    df = pd.DataFrame(data=s_eval.T, columns=system_names).to_csv(
        f"{output_path}/sort_scores_eval.csv"
    )
    t = np.linspace(0, 100, s_dev.shape[1])
    for i in range(len(system_names)):
        plt.plot(t, s_dev[i], linewidth=1.5, label=system_names[i])

    plt.title("sorted dev scores")
    plt.grid(True)
    plt.xlabel("% trials")
    plt.xlabel("LLR")
    plt.legend()
    plt.savefig(f"{output_path}/sort_scores_dev.png")
    plt.clf()

    t = np.linspace(0, 100, s_eval.shape[1])
    for i in range(len(system_names)):
        plt.plot(t, s_eval[i], linewidth=1.5, label=system_names[i])

    plt.title("sorted eval scores")
    plt.grid(True)
    plt.xlabel("% trials")
    plt.xlabel("LLR")
    plt.legend(loc="upper left")
    plt.savefig(f"{output_path}/sort_scores_eval.png")
    plt.clf()


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Compute metrics to assure that there are no errors in fusion systems"
    )

    parser.add_argument("--system-names", required=True, nargs="+")
    parser.add_argument("--score-files-dev", required=True, nargs="+")
    parser.add_argument("--score-files-eval", required=True, nargs="+")
    parser.add_argument("--ndx-file-dev", required=True)
    parser.add_argument("--ndx-file-eval", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_fusion_sanity(**namespace_to_dict(args))
