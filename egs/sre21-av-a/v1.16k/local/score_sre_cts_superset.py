#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import os
import logging
from jsonargparse import ArgumentParser, namespace_to_dict

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.metrics import fast_eval_dcf_eer as fast_eval


def score(key_file, score_file, output_file):

    conds = [
        "",
        "ENG_ENG",
        "ENG_CMN",
        "ENG_YUE",
        "CMN_CMN",
        "CMN_YUE",
        "YUE_YUE",
        "OTHER_ENG",
        "OTHER_CMN",
        "OTHER_YUE",
        "OTHER_OTHER",
        "samephn",
        "diffphn",
        "female",
        "male",
        "nenr1",
        "nenr3",
    ]

    logging.info("Load scores: %s", score_file)
    scr = TrialScores.load_txt(score_file)
    priors = np.array([0.01, 0.05])

    dsets = []
    eers = []
    min_dcfs = []
    act_dcfs = []
    ntars = []
    nnons = []
    cond_names = []
    for cond in conds:
        key_file_cond = key_file if cond == "" else key_file + "_" + cond
        dset = "sre-cts-superset-dev"
        cond_name = "all" if cond == "" else cond
        logging.info("Load key: %s", key_file_cond)
        key = TrialKey.load_txt(key_file_cond)
        tar, non = scr.get_tar_non(key)
        min_dcf, act_dcf, eer, _ = fast_eval(tar, non, priors)

        dsets.append(dset)
        eers.append(eer)
        min_dcfs.append(min_dcf)
        act_dcfs.append(act_dcf)
        ntars.append(len(tar))
        nnons.append(len(non))
        cond_names.append(cond_name)

    eers = np.asarray(eers)
    min_dcfs = np.vstack(min_dcfs)
    act_dcfs = np.vstack(act_dcfs)
    ntars = np.asarray(ntars)
    nnons = np.asarray(nnons)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    min_cp = np.mean(min_dcfs, axis=-1)
    act_cp = np.mean(act_dcfs, axis=-1)

    table = pd.DataFrame(
        {
            "dataset": dsets,
            "condition": cond_names,
            "eer": eers * 100,
            f"min_dcf({priors[1]})": min_dcfs[:, 1],
            f"act_dcf({priors[1]})": act_dcfs[:, 1],
            f"min_dcf({priors[0]})": min_dcfs[:, 0],
            f"act_dcf({priors[0]})": act_dcfs[:, 0],
            "min_cp": min_cp,
            "act_cp": act_cp,
            "num_target_trials": ntars,
            "num_nontarget_trials": nnons,
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

    score(**vars(args))
