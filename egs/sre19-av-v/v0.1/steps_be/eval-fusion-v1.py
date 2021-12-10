#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals greedy fusion
"""
import sys
import os
from jsonargparse import ArgumentParser, namespace_to_dict
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.classifiers import GreedyFusionBinaryLR as GF


def eval_fusion(in_score_files, ndx_file, model_file, out_score_file, fus_idx):

    logging.info("load ndx: %s", ndx_file)
    try:
        ndx = TrialNdx.load_txt(ndx_file)
    except:
        ndx = TrialKey.load_txt(ndx_file)

    num_systems = len(in_score_files)
    in_scores = []
    for i in range(num_systems):
        logging.info("load scores: %s", in_score_files[i])
        scr = TrialScores.load_txt(in_score_files[i])
        scr = scr.align_with_ndx(ndx)
        in_scores.append(scr.scores.ravel()[:, None])

    in_scores = np.concatenate(tuple(in_scores), axis=1)

    logging.info("load model: %s", model_file)
    gf = GF.load(model_file)
    logging.info("apply fusion")
    s_fus = gf.predict(in_scores, fus_idx=fus_idx)
    scr.scores = np.reshape(s_fus, scr.scores.shape)

    logging.info("save scores: %s", out_score_file)
    scr.save_txt(out_score_file)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evals linear fusion from greedy fusion trainer"
    )

    parser.add_argument("--in-score-files", required=True, nargs="+")
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--fus-idx", required=True, type=int)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_fusion(**namespace_to_dict(args))
