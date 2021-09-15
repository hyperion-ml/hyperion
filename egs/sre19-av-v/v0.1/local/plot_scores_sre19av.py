#!/usr/bin/env python
"""
Plot histogram of i-vectors
"""
import sys
import os
import argparse
import time

import numpy as np
from scipy.stats import mode
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_key import TrialKey


def gauss_map_adapt(mu, sigma, mu0, sigma0, N, r_mu, r_s2):

    s2 = sigma ** 2
    s02 = sigma0 ** 2

    alpha_mu = N / (N + r_mu)
    alpha_s2 = N / (N + r_mu)
    mu_map = alpha_mu * mu + (1 - alpha_mu) * mu0
    s2_map = (
        alpha_s2 * s2
        + (1 - alpha_s2) * s02
        + alpha_s2 * (1 - alpha_mu) * (mu - mu0) ** 2
    )
    return mu_map, np.sqrt(s2_map)


def plot_scores_sre18(
    key_sre19_dev,
    scores_sre19_dev,
    key_sre19_eval,
    scores_sre19_eval,
    key_janus_dev,
    scores_janus_dev,
    key_janus_eval,
    scores_janus_eval,
    name,
    output_path,
):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    k_sre19_dev = TrialKey.load_txt(key_sre19_dev)
    scr_sre19_dev = TrialScores.load_txt(scores_sre19_dev)
    k_sre19_eval = TrialKey.load_txt(key_sre19_eval)
    scr_sre19_eval = TrialScores.load_txt(scores_sre19_eval)
    k_janus_dev = TrialKey.load_txt(key_janus_dev)
    scr_janus_dev = TrialScores.load_txt(scores_janus_dev)
    k_janus_eval = TrialKey.load_txt(key_janus_eval)
    scr_janus_eval = TrialScores.load_txt(scores_janus_eval)

    tar_sre19_dev, non_sre19_dev = scr_sre19_dev.get_tar_non(k_sre19_dev)
    tar_sre19_eval, non_sre19_eval = scr_sre19_eval.get_tar_non(k_sre19_eval)
    tar_janus_dev, non_janus_dev = scr_janus_dev.get_tar_non(k_janus_dev)
    tar_janus_eval, non_janus_eval = scr_janus_eval.get_tar_non(k_janus_eval)

    p = 0.05
    thr = -np.log(p / (1 - p))

    plt.hist(
        tar_sre19_dev,
        5,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
        label="SRE19 dev AV",
    )
    plt.hist(
        non_sre19_dev,
        100,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_janus_dev,
        10,
        histtype="step",
        density=True,
        color="r",
        linestyle="solid",
        linewidth=1.5,
        label="JANUS dev CORE",
    )
    plt.hist(
        non_janus_dev,
        200,
        histtype="step",
        density=True,
        color="r",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_janus_eval,
        30,
        histtype="step",
        density=True,
        color="g",
        linestyle="solid",
        linewidth=1.5,
        label="JANUS eval CORE",
    )
    plt.hist(
        non_janus_eval,
        500,
        histtype="step",
        density=True,
        color="g",
        linestyle="solid",
        linewidth=1.5,
    )

    plt.axvline(x=thr, color="k")
    # plt.title(name)
    plt.xlabel("LLR score")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(output_path + "/hist_vid0.png")

    plt.hist(
        tar_sre19_eval,
        20,
        histtype="step",
        density=True,
        color="k",
        linestyle="solid",
        linewidth=1.5,
        label="SRE19 eval AV",
    )
    plt.hist(
        non_sre19_eval,
        200,
        histtype="step",
        density=True,
        color="c",
        linestyle="solid",
        linewidth=1.5,
    )

    plt.axvline(x=thr, color="k")

    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(output_path + "/hist_vid.png")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plots hist scores for sre19 av",
    )

    parser.add_argument("--key-sre19-dev", required=True)
    parser.add_argument("--scores-sre19-dev", required=True)
    parser.add_argument("--key-sre19-eval", required=True)
    parser.add_argument("--scores-sre19-eval", required=True)
    parser.add_argument("--key-janus-dev", required=True)
    parser.add_argument("--scores-janus-dev", required=True)
    parser.add_argument("--key-janus-eval", required=True)
    parser.add_argument("--scores-janus-eval", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--name", default="")

    args = parser.parse_args()

    plot_scores_sre18(**vars(args))
