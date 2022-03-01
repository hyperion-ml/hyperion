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
from hyperion.utils import SparseTrialScores, SparseTrialKey
from hyperion.np.metrics import fast_eval_dcf_eer as fast_eval


def score_dcf(key_file, score_file, output_path):

    logging.info("Load key: %s" % key_file)
    key = SparseTrialKey.load_txt(key_file)
    logging.info("Load scores: %s" % score_file)
    scr = SparseTrialScores.load_txt(score_file)
    logging.info("separating tar/non")
    tar, non = scr.get_tar_non(key)
    logging.info("computing EER/DCF")
    priors = np.array([0.001, 0.005, 0.01, 0.05])
    min_dcf, act_dcf, eer, _, min_pmiss, min_pfa, act_pmiss, act_pfa = fast_eval(
        tar, non, priors, return_probs=True
    )

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ntar = len(tar)
    nnon = len(non)

    output_file = output_path + "_results"
    with open(output_file, "w") as f:
        s = "EER: {0:.2f} DCF5e-2: {1:.3f} / {2:.3f} DCF1e-2: {3:.3f} / {4:.3f} DCF5e-3: {5:.3f} / {6:.3f} DCF1e-3: {7:.3f} / {8:.3f} ntar: {9:d} nnon: {10:d}\n".format(
            eer * 100,
            min_dcf[3],
            act_dcf[3],
            min_dcf[2],
            act_dcf[2],
            min_dcf[1],
            act_dcf[1],
            min_dcf[0],
            act_dcf[0],
            ntar,
            nnon,
        )
        f.write(s)
        logging.info(s)
        s = "min-pmiss={} min-pfa={} act-pmiss={} act-pfa={}".format(
            min_pmiss, min_pfa, act_pmiss, act_pfa
        )
        logging.info(s)
        s = "min-Nmiss={} min-Nfa={} act-Nmiss={} act-Nfa={}".format(
            min_pmiss * ntar, min_pfa * nnon, act_pmiss * ntar, act_pfa * nnon
        )
        logging.info(s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Computes EER and DCF",
    )

    parser.add_argument("--key-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    score_dcf(**vars(args))
