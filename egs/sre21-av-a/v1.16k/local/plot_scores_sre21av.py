#!/usr/bin/env python
"""
Plot histogram of scores
"""
import sys
import os
from jsonargparse import ArgumentParser, namespace_to_dict
import time

import numpy as np
from scipy.stats import mode
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.utils import TrialScores, TrialNdx, TrialKey


def plot_scores_condition(
    key_sre21a_dev,
    scr_sre21a_dev,
    key_sre21a_eval,
    scr_sre21a_eval,
    key_sre21av_dev,
    scr_sre21av_dev,
    key_sre21av_eval,
    scr_sre21av_eval,
    cond,
    name,
    output_path,
):

    k_sre21a_dev = TrialKey.load_txt(f"{key_sre21a_dev}_{cond}")
    # k_sre21a_eval = TrialKey.load_txt(f"{key_sre21a_eval}_{cond}")
    k_sre21a_eval = TrialNdx.load_txt(f"{key_sre21a_eval}_{cond}")

    tar_sre21a_dev, non_sre21a_dev = scr_sre21a_dev.get_tar_non(k_sre21a_dev)
    # tar_sre21a_eval, non_sre21a_eval = scr_sre21a_eval.get_tar_non(k_sre21a_eval)
    scr_sre21a_eval = scr_sre21a_eval.align_with_ndx(k_sre21a_eval)
    non_sre21a_eval = scr_sre21a_eval.scores[k_sre21a_eval.trial_mask]

    if cond != "CTS_CTS":
        k_sre21av_dev = TrialKey.load_txt(f"{key_sre21av_dev}_{cond}")
        # k_sre21av_eval = TrialKey.load_txt(f"{key_sre21av_eval}_{cond}")
        k_sre21av_eval = TrialNdx.load_txt(f"{key_sre21av_eval}_{cond}")

        tar_sre21av_dev, non_sre21av_dev = scr_sre21av_dev.get_tar_non(k_sre21av_dev)
        # tar_sre21av_eval, non_sre21av_eval = scr_sre21av_eval.get_tar_non(k_sre21av_eval)
        scr_sre21av_eval = scr_sre21av_eval.align_with_ndx(k_sre21av_eval)
        non_sre21av_eval = scr_sre21av_eval.scores[k_sre21av_eval.trial_mask]

    p = 0.05
    thr = -np.log(p / (1 - p))

    plt.hist(
        tar_sre21a_dev,
        5,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
        label="SRE21 dev audio",
    )
    plt.hist(
        non_sre21a_dev,
        100,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
    )

    if cond != "CTS_CTS":
        plt.hist(
            tar_sre21av_dev,
            5,
            histtype="step",
            density=True,
            color="r",
            linestyle="solid",
            linewidth=1.5,
            label="SRE21 dev AV",
        )
        plt.hist(
            non_sre21av_dev,
            100,
            histtype="step",
            density=True,
            color="r",
            linestyle="solid",
            linewidth=1.5,
        )

    plt.axvline(x=thr, color="k")
    plt.title(name)
    plt.xlabel("LLR score")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_path}/hist_{cond}_vid0.png")

    # plt.hist(tar_sre19_eval, 20, histtype='step', density=True, color='k',
    #          linestyle='solid', linewidth=1.5, label='SRE19 eval AV')
    plt.hist(
        non_sre21a_eval,
        200,
        histtype="step",
        density=True,
        color="k",
        linestyle="solid",
        linewidth=1.5,
        label="SRE21 eval audio",
    )

    if cond != "CTS_CTS":
        plt.hist(
            non_sre21av_eval,
            200,
            histtype="step",
            density=True,
            color="g",
            linestyle="solid",
            linewidth=1.5,
            label="SRE21 eval AV",
        )

    plt.axvline(x=thr, color="k")
    plt.title(name)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_path}/hist_{cond}_vid.png")
    plt.clf()


def plot_scores_sre21(
    key_sre21a_dev,
    scores_sre21a_dev,
    key_sre21a_eval,
    scores_sre21a_eval,
    key_sre21av_dev,
    scores_sre21av_dev,
    key_sre21av_eval,
    scores_sre21av_eval,
    name,
    output_path,
):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scr_sre21a_dev = TrialScores.load_txt(scores_sre21a_dev)
    scr_sre21a_eval = TrialScores.load_txt(scores_sre21a_eval)
    scr_sre21av_dev = TrialScores.load_txt(scores_sre21av_dev)
    scr_sre21av_eval = TrialScores.load_txt(scores_sre21av_eval)

    for c in ["CTS_CTS", "CTS_AFV", "AFV_AFV"]:
        plot_scores_condition(
            key_sre21a_dev,
            scr_sre21a_dev,
            key_sre21a_eval,
            scr_sre21a_eval,
            key_sre21av_dev,
            scr_sre21av_dev,
            key_sre21av_eval,
            scr_sre21av_eval,
            c,
            name,
            output_path,
        )


if __name__ == "__main__":

    parser = ArgumentParser(description="Plots hist scores for sre21 av")

    parser.add_argument("--key-sre21a-dev", required=True)
    parser.add_argument("--scores-sre21a-dev", required=True)
    parser.add_argument("--key-sre21a-eval", required=True)
    parser.add_argument("--scores-sre21a-eval", required=True)
    parser.add_argument("--key-sre21av-dev", required=True)
    parser.add_argument("--scores-sre21av-dev", required=True)
    parser.add_argument("--key-sre21av-eval", required=True)
    parser.add_argument("--scores-sre21av-eval", required=True)
    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--name", dest="name", default="")

    args = parser.parse_args()

    plot_scores_sre21(**namespace_to_dict(args))
