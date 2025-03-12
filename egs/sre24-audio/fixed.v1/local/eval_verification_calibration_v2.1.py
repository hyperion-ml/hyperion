#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals calibration
"""
import math
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
from hyperion.np.pdfs import GMMDiagCov
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.utils.list_utils import ismember
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.math_funcs import softmax

conditions = ODict()
# conditions["gender"] = ["male", "female"]
conditions["source_type_match"] = ["Y", "N"]
conditions["language_match"] = ["Y", "N"]


def eval_verification_calibration(
        in_score_file, ndx_file, model_files, gmm_files, out_score_file, max_trials,
):

    logging.info("load ndx: %s", ndx_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file).to_ndx()

    ndx_ext = Path(ndx_file).suffix
    sep = "\t" if ndx_ext == ".tsv" else ","
    df_ndx = pd.read_csv(ndx_file, sep=sep)

    logging.info("load scores: %s", in_score_file)
    scr = TrialScores.load(in_score_file)
    scr = scr.align_with_ndx(ndx)

    scores = [scr.scores[ndx.trial_mask]]
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
            scores.append(cond_mat[ndx.trial_mask])

    scores = np.vstack(scores).T
    final_s_cal = np.zeros((scores.shape[0],))
    if max_trials is None:
        max_trials = scores.shape[0]
        
    idx_start = 0
    for j in range(math.ceil(scores.shape[0]/max_trials)):
        idx_end=min(idx_start+max_trials, scores.shape[0])
        num_trials_j = idx_end - idx_start
        scores_i = scores[idx_start:idx_end,:]
        llk = np.zeros((num_trials_j, len(gmm_files)))
        s_cal = np.zeros((num_trials_j, len(gmm_files)))
        assert len(model_files) == len(gmm_files)
        for i, (model_file, gmm_file) in enumerate(zip(model_files, gmm_files)):
            logging.info("load gmm: %s", gmm_file)
            gmm = GMMDiagCov.load(gmm_file)
            llk_i = gmm.log_prob_std(scores_i[:, :1])
            # llk_i = gmm.log_prob_std(scores)
            llk[:, i] = llk_i
            logging.info("load model: %s", model_file)
            lr = LR.load(model_file)
            logging.info("apply calibration")
            s_cal_i = lr.predict(scores_i)
            s_cal[:, i] = s_cal_i

        p_cal = softmax(llk, axis=1)
        s_cal = np.sum(p_cal * s_cal, axis=1)
        final_s_cal[idx_start:idx_end] = s_cal
        idx_start += max_trials
        
    #scr.scores = np.reshape(final_s_cal, scr.scores.shape)
    scr.scores[ndx.trial_mask] = final_s_cal

    logging.info("save scores: %s", out_score_file)
    scr.save(out_score_file)


def main():
    parser = ArgumentParser(description="Evals linear calibration")

    parser.add_argument("--in-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--model-files", nargs="+", required=True)
    parser.add_argument("--gmm-files", nargs="+", required=True)
    parser.add_argument("--max-trials", default=1000000, type=int)
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_verification_calibration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
