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
    key_sitw_core,
    scores_sitw_core,
    key_sitw_core_multi,
    scores_sitw_core_multi,
    key_sre18_eval,
    scores_sre18_eval,
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
        os.makedirs(ouput_path)

    k_sitw_core = TrialKey.load_txt(key_sitw_core)
    scr_sitw_core = TrialScores.load_txt(scores_sitw_core)
    k_sitw_core_multi = TrialKey.load_txt(key_sitw_core_multi)
    scr_sitw_core_multi = TrialScores.load_txt(scores_sitw_core_multi)
    k_sre18_eval = TrialKey.load_txt(key_sre18_eval)
    scr_sre18_eval = TrialScores.load_txt(scores_sre18_eval)
    k_sre19_dev = TrialKey.load_txt(key_sre19_dev)
    scr_sre19_dev = TrialScores.load_txt(scores_sre19_dev)
    # k_sre19_eval = TrialKey.load_txt(key_sre19_eval)
    k_sre19_eval = TrialNdx.load_txt(key_sre19_eval)
    scr_sre19_eval = TrialScores.load_txt(scores_sre19_eval)
    k_janus_dev = TrialKey.load_txt(key_janus_dev)
    scr_janus_dev = TrialScores.load_txt(scores_janus_dev)
    k_janus_eval = TrialKey.load_txt(key_janus_eval)
    scr_janus_eval = TrialScores.load_txt(scores_janus_eval)

    tar_sitw_core, non_sitw_core = scr_sitw_core.get_tar_non(k_sitw_core)
    tar_sitw_core_multi, non_sitw_core_multi = scr_sitw_core_multi.get_tar_non(
        k_sitw_core_multi
    )
    tar_sre18_eval, non_sre18_eval = scr_sre18_eval.get_tar_non(k_sre18_eval)
    tar_sre19_dev, non_sre19_dev = scr_sre19_dev.get_tar_non(k_sre19_dev)
    # tar_sre19_eval, non_sre19_eval = scr_sre19_eval.get_tar_non(k_sre19_eval)
    scr_sre19_eval = scr_sre19_eval.align_with_ndx(k_sre19_eval)
    non_sre19_eval = scr_sre19_eval.scores[k_sre19_eval.trial_mask]
    tar_janus_dev, non_janus_dev = scr_janus_dev.get_tar_non(k_janus_dev)
    tar_janus_eval, non_janus_eval = scr_janus_eval.get_tar_non(k_janus_eval)

    p = 0.05
    thr = -np.log(p / (1 - p))

    plt.hist(
        tar_sitw_core,
        80,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
        label="SITW eval core-core",
    )
    plt.hist(
        non_sitw_core,
        1000,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_sitw_core_multi,
        80,
        histtype="step",
        density=True,
        color="r",
        linestyle="solid",
        linewidth=1.5,
        label="SITW eval core-multi",
    )
    plt.hist(
        non_sitw_core_multi,
        1000,
        histtype="step",
        density=True,
        color="r",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_sre18_eval,
        15,
        histtype="step",
        density=True,
        color="g",
        linestyle="solid",
        linewidth=1.5,
        label="SRE18 eval VAST",
    )
    plt.hist(
        non_sre18_eval,
        200,
        histtype="step",
        density=True,
        color="g",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_sre19_dev,
        5,
        histtype="step",
        density=True,
        color="c",
        linestyle="solid",
        linewidth=1.5,
        label="SRE19 dev AV",
    )
    plt.hist(
        non_sre19_dev,
        100,
        histtype="step",
        density=True,
        color="c",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_janus_dev,
        10,
        histtype="step",
        density=True,
        color="y",
        linestyle="solid",
        linewidth=1.5,
        label="JANUS dev CORE",
    )
    plt.hist(
        non_janus_dev,
        200,
        histtype="step",
        density=True,
        color="y",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_janus_eval,
        30,
        histtype="step",
        density=True,
        color="m",
        linestyle="solid",
        linewidth=1.5,
        label="JANUS eval CORE",
    )
    plt.hist(
        non_janus_eval,
        500,
        histtype="step",
        density=True,
        color="m",
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

    # plt.hist(tar_sre19_eval, 20, histtype='step', density=True, color='k',
    #          linestyle='solid', linewidth=1.5, label='SRE19 eval AV')
    plt.hist(
        non_sre19_eval,
        200,
        histtype="step",
        density=True,
        color="k",
        linestyle="solid",
        linewidth=1.5,
        label="SRE19 eval AV",
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
        fromfile_prefix_chars="@",
        description="Plots hist scores for sre19 av",
    )

    parser.add_argument("--key-sitw-core", dest="key_sitw_core", required=True)
    parser.add_argument("--scores-sitw-core", dest="scores_sitw_core", required=True)
    parser.add_argument(
        "--key-sitw-core-multi", dest="key_sitw_core_multi", required=True
    )
    parser.add_argument(
        "--scores-sitw-core-multi", dest="scores_sitw_core_multi", required=True
    )
    parser.add_argument("--key-sre18-eval", dest="key_sre18_eval", required=True)
    parser.add_argument("--scores-sre18-eval", dest="scores_sre18_eval", required=True)
    parser.add_argument("--key-sre19-dev", dest="key_sre19_dev", required=True)
    parser.add_argument("--scores-sre19-dev", dest="scores_sre19_dev", required=True)
    parser.add_argument("--key-sre19-eval", dest="key_sre19_eval", required=True)
    parser.add_argument("--scores-sre19-eval", dest="scores_sre19_eval", required=True)
    parser.add_argument("--key-janus-dev", dest="key_janus_dev", required=True)
    parser.add_argument("--scores-janus-dev", dest="scores_janus_dev", required=True)
    parser.add_argument("--key-janus-eval", dest="key_janus_eval", required=True)
    parser.add_argument("--scores-janus-eval", dest="scores_janus_eval", required=True)
    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--name", dest="name", default="")

    args = parser.parse_args()

    plot_scores_sre18(**vars(args))
