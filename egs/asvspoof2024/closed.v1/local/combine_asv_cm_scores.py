"""
 Copyright 2024 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import logging
import math

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.np.metrics import effective_prior
from hyperion.utils import SegmentSet, TrialScores


def log_sigmoid(x):
    y = np.copy(x)
    idx = y > -6
    y[idx] = -np.log(1 + np.exp(-x[idx]))
    return y


def inv_log_sigmoid(x):
    y = np.exp(x)
    return np.log(y / (1 - y))


def combine_asv_cm_scores(
    asv_score_file,
    cm_score_file,
    out_score_file,
    p_tar,
    p_spoof,
    c_miss,
    c_fa,
    c_fa_spoof,
):

    p_non = 1.0 - p_tar - p_spoof

    logging.info("loading score-files %s %s", asv_score_file, cm_score_file)
    asv_scores = TrialScores.load(asv_score_file)
    cm_scores = TrialScores.load(cm_score_file)
    # align cm_scores to asv_scores
    cm_scores = cm_scores.filter(cm_scores.model_set, asv_scores.seg_set)

    logging.info("combining scores")
    # We combine ASV and Countermeasure scores using
    # P(same, bonafide | x_e, x_t) = P(same | x_e, x_t, bonafide) P(bonafide | x_t)
    # Input scores are assumed well-calibrated LLR
    # asv-score = LLR_sv = log P(x_e, x_t | same, bonafide) / P(x_e, x_t | diff, bonafide)
    # cm-score  = LLR_cm = log P(x_t | bonafide) / P(x_t | spoof)

    # logP(same | x_e, x_t, bonafide) = log-sigmoid(LLR_sv + log P(tar|bonafide)/P(non|bonafide))
    p_tar_bonafide = p_tar / (p_tar + p_non)
    p_tar_bonafide_eff = effective_prior(p_tar_bonafide, c_miss, c_fa)
    p_non_bonafide_eff = 1 - p_tar_bonafide_eff
    log_p_same_given_data_bonafide = log_sigmoid(
        asv_scores.scores + math.log(p_tar_bonafide_eff) - math.log(p_non_bonafide_eff)
    )

    # logP(bonafide | x_t) = log-sigmoid(LLR_sv + log P(bonafide)/P(spoof))
    p_bonafide_eff = effective_prior(1-p_spoof, c_miss, c_fa_spoof)
    log_p_bonafide_given_data = log_sigmoid(
        cm_scores.scores + math.log(p_bonafide_eff) - math.log(1-p_bonafide_eff)
    )

    # log P(same, bonafide | x_e, x_t) =
    log_p_same_bonafide_given_data = (
        log_p_same_given_data_bonafide + log_p_bonafide_given_data
    )

    # Transform log P(same, bonafide | x_e, x_t) into LLR
    # LLR = log P(x_e, x_t | same, bonafide) / P(x_e, x_t | diff or spoof)
    # Solving:
    # P(same, bonafide | x_e, x_t) = sigmoid(LLR + log P(same, bonafide) / P(diff or spoof))
    # where P(same, bonafide) = P(tar) and P(diff or spoof) = P(non) + P(spoof) = 1 - P(tar)
    p_tar_eff = effective_prior(p_tar, c_miss, (c_fa * p_non + c_fa_spoof*p_spoof)/(1-p_tar))
    llr = (
        inv_log_sigmoid(log_p_same_bonafide_given_data)
        - math.log(p_tar_eff)
        + math.log(1 - p_tar_eff)
    )
    asv_scores.scores = llr

    logging.info("saving scores %s", out_score_file)
    asv_scores.save(out_score_file)


def main():
    parser = ArgumentParser(description="""Transform Spoofing logits to TrialScores""")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--asv-score-file", required=True)
    parser.add_argument("--cm-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument("--p-tar", default=0.9405, type=float)
    parser.add_argument("--p-spoof", default=0.05, type=float)
    parser.add_argument("--c-miss", default=1.0, type=float)
    parser.add_argument("--c-fa", default=10.0, type=float)
    parser.add_argument("--c-fa-spoof", default=10.0, type=float)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    del args.cfg
    logging.debug(args)

    combine_asv_cm_scores(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
