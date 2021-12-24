k  #!/usr/bin/env python

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
    key_sre18_dev,
    scores_sre18_dev,
    key_sre18_eval,
    scores_sre18_eval,
    key_sre19_eval,
    scores_sre19_eval,
    name,
    output_path,
):

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    k_sre18_dev = TrialKey.load_txt(key_sre18_dev)
    scr_sre18_dev = TrialScores.load_txt(scores_sre18_dev)
    k_sre18_eval = TrialKey.load_txt(key_sre18_eval)
    scr_sre18_eval = TrialScores.load_txt(scores_sre18_eval)
    k_sre19_eval = TrialNdx.load_txt(key_sre19_eval)
    scr_sre19_eval = TrialScores.load_txt(scores_sre19_eval)

    tar_sre18_dev, non_sre18_dev = scr_sre18_dev.get_tar_non(k_sre18_dev)
    tar_sre18_eval, non_sre18_eval = scr_sre18_eval.get_tar_non(k_sre18_eval)
    # tar_sre19_eval, non_sre19_eval = scr_sre19_eval.get_tar_non(k_sre19_eval)
    scr_sre19_eval = scr_sre19_eval.align_with_ndx(k_sre19_eval)
    non_sre19_eval = scr_sre19_eval.scores[k_sre19_eval.trial_mask]

    p = 0.0075
    thr = -np.log(p / (1 - p))

    plt.hist(
        tar_sre18_dev,
        100,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
        label="SRE18 dev cmn2",
    )
    plt.hist(
        non_sre18_dev,
        1000,
        histtype="step",
        density=True,
        color="b",
        linestyle="solid",
        linewidth=1.5,
    )
    plt.hist(
        tar_sre18_eval,
        100,
        histtype="step",
        density=True,
        color="r",
        linestyle="solid",
        linewidth=1.5,
        label="SRE18 eval cmn2",
    )
    plt.hist(
        non_sre18_eval,
        1000,
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
    plt.show()
    plt.savefig(output_path + "/hist_tel0.png")

    # plt.hist(tar_sre19_eval, 100, histtype='step', density=True, color='g',
    #          linestyle='solid', linewidth=1.5, label='SRE18 eval cmn2')
    plt.hist(
        non_sre19_eval,
        1000,
        histtype="step",
        density=True,
        color="g",
        linestyle="solid",
        linewidth=1.5,
        label="SRE19 eval cmn2",
    )

    plt.axvline(x=thr, color="k")

    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(output_path + "/hist_tel.png")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Plots hist scores for sre19 cmn2",
    )

    parser.add_argument("--key-sre18-dev", dest="key_sre18_dev", required=True)
    parser.add_argument("--scores-sre18-dev", dest="scores_sre18_dev", required=True)
    parser.add_argument("--key-sre18-eval", dest="key_sre18_eval", required=True)
    parser.add_argument("--scores-sre18-eval", dest="scores_sre18_eval", required=True)
    parser.add_argument("--key-sre19-eval", dest="key_sre19_eval", required=True)
    parser.add_argument("--scores-sre19-eval", dest="scores_sre19_eval", required=True)
    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument("--name", dest="name", default="")

    args = parser.parse_args()

    plot_scores_sre18(**vars(args))
