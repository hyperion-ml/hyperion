#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals calibration
"""

import logging
from collections import OrderedDict as ODict
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.utils.list_utils import ismember
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores

conditions = ODict()
# conditions["gender"] = ["male", "female"]
conditions["source_type_match"] = ["Y", "N"]
conditions["language_match"] = ["Y", "N"]


def eval_verification_calibration(in_score_file, ndx_file, model_file, out_score_file):

    logging.info("load ndx: %s", ndx_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file)

    ndx_ext = Path(ndx_file).suffix
    sep = "\t" if ndx_ext == ".tsv" else ","
    df_ndx = pd.read_csv(ndx_file, sep=sep)

    logging.info("load scores: %s", in_score_file)
    scr = TrialScores.load(in_score_file)
    scr = scr.align_with_ndx(ndx)

    scores = [scr.scores.ravel()]
    for cond_key, cond_vals in conditions.items():
        for cond_val in cond_vals:
            logging.info("making condition matrix: %s==%s", cond_key, cond_val)
            df_cond = df_ndx[df_ndx[cond_key] == cond_val]
            cond_mat = np.zeros_like(scr.scores)
            model_set = df_cond["modelid"]
            seg_set = df_cond["segmentid"]
            f, enr_idx = ismember(model_set, ndx.model_set)
            f, test_idx = ismember(seg_set, ndx.seg_set)
            cond_mat[enr_idx, test_idx] = 1.0
            scores.append(cond_mat.ravel())

    scores = np.vstack(scores).T
    logging.info("load model: %s", model_file)
    lr = LR.load(model_file)
    logging.info("apply calibration")
    s_cal = lr.predict(scores)
    scr.scores = np.reshape(s_cal, scr.scores.shape)

    logging.info("save scores: %s", out_score_file)
    scr.save(out_score_file)


def main():
    parser = ArgumentParser(description="Evals linear calibration")

    parser.add_argument("--in-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_verification_calibration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
