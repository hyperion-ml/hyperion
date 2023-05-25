#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""


import sys
import os
import argparse
import time
import logging

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.np.metrics.verification_evaluator import (
    VerificationAdvAttackEvaluator as Eval,
)


def evaluate_attacks(
    key_file,
    clean_score_file,
    attack_score_files,
    attack_stats_files,
    output_path,
    prior,
):

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    evaluator = Eval(
        key_file, clean_score_file, attack_score_files, attack_stats_files, prior
    )

    # performance vs SNR
    logging.info("compute perf vs snr for all trials")
    df_clean = evaluator.compute_dcf_eer(return_df=True)
    df_clean.insert(0, "snr", np.inf)

    df = evaluator.compute_dcf_eer_vs_stats(
        "snr",
        [-10, 0, 10, 20, 30, 40, 50, 60],
        "all",
        higher_better=True,
        return_df=True,
    )
    file_path = "%s_attack_all_snr_results.csv" % (output_path)
    df = pd.concat([df_clean, df], ignore_index=True)
    df.to_csv(file_path)
    file_path = "%s_attack_all_snr" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "snr", file_path, clean_ref=0, xlabel="SNR(dB)", higher_better=True
    )

    logging.info("compute perf vs snr for tar trials")
    df = evaluator.compute_dcf_eer_vs_stats(
        "snr",
        [-10, 0, 10, 20, 30, 40, 50, 60],
        "tar",
        higher_better=True,
        return_df=True,
    )
    file_path = "%s_attack_tar_snr_results.csv" % (output_path)
    df = pd.concat([df_clean, df], ignore_index=True)
    df.to_csv(file_path)
    file_path = "%s_attack_tar_snr" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "snr", file_path, clean_ref=0, xlabel="SNR(dB)", higher_better=True
    )

    logging.info("compute perf vs snr for non trials")
    df = evaluator.compute_dcf_eer_vs_stats(
        "snr",
        [-10, 0, 10, 20, 30, 40, 50, 60],
        "non",
        higher_better=True,
        return_df=True,
    )
    file_path = "%s_attack_non_snr_results.csv" % (output_path)
    df = pd.concat([df_clean, df], ignore_index=True)
    df.to_csv(file_path)
    file_path = "%s_attack_non_snr" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "snr", file_path, clean_ref=0, xlabel="SNR(dB)", higher_better=True
    )

    logging.info("find best attacks from snr point of view")
    for i in range(len(attack_score_files)):
        file_path = "%s_best_snr_tar_attacks_%d.csv" % (output_path, i)
        evaluator.save_best_attacks(
            file_path,
            "snr",
            "tar",
            num_best=10,
            min_delta=1,
            attack_idx=i,
            higher_better=True,
        )

        file_path = "%s_best_snr_non_attacks_%d.csv" % (output_path, i)
        evaluator.save_best_attacks(
            file_path,
            "snr",
            "non",
            num_best=10,
            min_delta=1,
            attack_idx=i,
            higher_better=True,
        )

    # performance vs Linf
    logging.info("compute perf vs linf for all trials")
    eps = np.ceil(np.asarray([0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) * 2 ** 15)
    df = evaluator.compute_dcf_eer_vs_stats(
        "n_linf", eps, "all", higher_better=False, return_df=True
    )
    file_path = "%s_attack_all_linf_results.csv" % (output_path)
    df.to_csv(file_path)
    file_path = "%s_attack_all_linf" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "n_linf", file_path, clean_ref=0, xlabel=r"$L_{\infty}$", log_x=True
    )

    logging.info("compute perf vs linf for tar trials")
    df = evaluator.compute_dcf_eer_vs_stats(
        "n_linf", eps, "tar", higher_better=False, return_df=True
    )
    file_path = "%s_attack_tar_linf_results.csv" % (output_path)
    df.to_csv(file_path)
    file_path = "%s_attack_tar_linf" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "n_linf", file_path, clean_ref=0, xlabel=r"$L_{\infty}$", log_x=True
    )

    logging.info("compute perf vs linf for non trials")
    df = evaluator.compute_dcf_eer_vs_stats(
        "n_linf", eps, "non", higher_better=False, return_df=True
    )
    file_path = "%s_attack_non_linf_results.csv" % (output_path)
    df.to_csv(file_path)
    file_path = "%s_attack_non_linf" % (output_path)
    evaluator.plot_dcf_eer_vs_stat_v1(
        df, "n_linf", file_path, clean_ref=0, xlabel=r"$L_{\infty}$", log_x=True
    )

    # find the best attacks in terms of linf
    logging.info("find best attacks from linf point of view")
    for i in range(len(attack_score_files)):
        file_path = "%s_best_linf_tar_attacks_%d.csv" % (output_path, i)
        evaluator.save_best_attacks(
            file_path,
            "n_linf",
            "tar",
            num_best=10,
            min_delta=1,
            attack_idx=i,
            higher_better=False,
        )

        file_path = "%s_best_linf_non_attacks_%d.csv" % (output_path, i)
        evaluator.save_best_attacks(
            file_path,
            "n_linf",
            "non",
            num_best=10,
            min_delta=1,
            attack_idx=i,
            higher_better=False,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Analyses performance of adversarial attacks for spk. verif.",
    )

    parser.add_argument("--key-file", required=True)
    parser.add_argument("--clean-score-file", required=True)
    parser.add_argument("--attack-score-files", required=True, nargs="+")
    parser.add_argument("--attack-stats-files", required=True, nargs="+")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prior", default=0.05, type=float)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    evaluate_attacks(**vars(args))
