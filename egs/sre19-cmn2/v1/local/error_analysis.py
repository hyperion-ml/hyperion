#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""


import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval


def score_dcf(key_file, score_file, output_path):

    logging.info("Load key: %s" % key_file)
    key = TrialKey.load_txt(key_file)
    logging.info("Load scores: %s" % score_file)
    scr = TrialScores.load_txt(score_file)
    scr = scr.align_with_ndx(key)

    p = 0.05
    thr = -np.log(p / (1 - p))

    tar_enr_scr = np.sum(scr.scores * key.tar, axis=1) / (
        np.sum(key.tar, axis=1) + 1e-5
    )
    tar_tst_scr = np.sum(scr.scores * key.tar, axis=0) / (
        np.sum(key.tar, axis=0) + 1e-5
    )
    non_enr_scr = np.sum(scr.scores * key.non, axis=1) / (
        np.sum(key.non, axis=1) + 1e-5
    )
    non_tst_scr = np.sum(scr.scores * key.non, axis=0) / (
        np.sum(key.non, axis=0) + 1e-5
    )

    tar_enr_err = np.sum(np.logical_and(scr.scores < thr, key.tar), axis=1)
    tar_tst_err = np.sum(np.logical_and(scr.scores < thr, key.tar), axis=0)
    non_enr_err = np.sum(np.logical_and(scr.scores > thr, key.non), axis=1)
    non_tst_err = np.sum(np.logical_and(scr.scores > thr, key.non), axis=0)

    tar_enr_idx = np.argsort(tar_enr_scr)
    tar_tst_idx = np.argsort(tar_tst_scr)
    non_enr_idx = np.argsort(non_enr_scr)[::-1]
    non_tst_idx = np.argsort(non_tst_scr)[::-1]

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = output_path + "_worse_tar_enr"
    with open(output_file, "w") as f:
        for idx in tar_enr_idx:
            f.write(
                "%s %f %d\n" % (key.model_set[idx], tar_enr_scr[idx], tar_enr_err[idx])
            )

    output_file = output_path + "_worse_tar_tst"
    with open(output_file, "w") as f:
        for idx in tar_tst_idx:
            f.write(
                "%s %f %d\n" % (key.seg_set[idx], tar_tst_scr[idx], tar_tst_err[idx])
            )

    output_file = output_path + "_worse_non_enr"
    with open(output_file, "w") as f:
        for idx in non_enr_idx:
            f.write(
                "%s %f %d\n" % (key.model_set[idx], non_enr_scr[idx], non_enr_err[idx])
            )

    output_file = output_path + "_worse_non_tst"
    with open(output_file, "w") as f:
        for idx in non_tst_idx:
            f.write(
                "%s %f %d\n" % (key.seg_set[idx], non_tst_scr[idx], non_tst_err[idx])
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Computes EER and DCF",
    )

    parser.add_argument("--key-file", dest="key_file", required=True)
    parser.add_argument("--score-file", dest="score_file", required=True)
    parser.add_argument("--output-path", dest="output_path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    score_dcf(**vars(args))
