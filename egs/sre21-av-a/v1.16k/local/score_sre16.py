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
from jsonargparse import ArgumentParser, namespace_to_dict

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval


def score_dcf(key_file, score_file, output_file):

    logging.info("Load key: %s" % key_file)
    key = TrialKey.load_txt(key_file)
    logging.info("Load scores: %s" % score_file)
    scr = TrialScores.load_txt(score_file)
    tar, non = scr.get_tar_non(key)

    priors = np.array([0.01, 0.05])
    min_dcf, act_dcf, eer, _ = fast_eval(tar, non, priors)

    min_cp = np.mean(min_dcf)
    act_cp = np.mean(act_dcf)

    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ed = lambda x: np.expand_dims(x, axis=-1)
    table = pd.DataFrame(
        {
            "dataset": ["sre16-eval40 yue"],
            "condition": "all",
            "eer": ed(eer) * 100,
            f"min_dcf({priors[1]})": ed(min_dcf[1]),
            f"act_dcf({priors[1]})": ed(act_dcf[1]),
            f"min_dcf({priors[0]})": ed(min_dcf[0]),
            f"act_dcf({priors[0]})": ed(act_dcf[0]),
            "min_cp": ed(min_cp),
            "act_cp": ed(act_cp),
            "num_target_trials": ed(len(tar)),
            "num_nontarget_trials": ed(len(non)),
        }
    )
    table.to_csv(output_file, sep=",", index=False, float_format="%.3f")
    logging.info(f"results:\n{table}")


if __name__ == "__main__":

    parser = ArgumentParser(description="Computes EER and DCF for SRE16")

    parser.add_argument("--key-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    score_dcf(**vars(args))
