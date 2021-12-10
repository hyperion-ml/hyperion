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
from hyperion.utils import TrialScores, TrialKey, TrialNdx, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.classifiers import GreedyFusionBinaryLR as GF


def read_ndx(ndx_file):
    logging.info("load ndx: %s" % ndx_file)
    try:
        ndx = TrialNdx.load_txt(ndx_file)
    except:
        ndx = TrialKey.load_txt(ndx_file)

    return ndx


def read_sources(ndx_file, ndx):
    ns = np.sum(ndx.trial_mask)
    src = np.zeros((ns, 3), dtype=float_cpu())
    sources = ["CTS", "AFV"]
    k = 0
    for i in range(len(sources)):
        for j in range(i, len(sources)):
            li = sources[i]
            lj = sources[j]
            logging.info("load key: %s", f"{ndx_file}_{li}_{lj}")
            try:
                ndx_c = TrialKey.load_txt(f"{ndx_file}_{li}_{lj}").to_ndx()
            except:
                ndx_c = TrialNdx.load_txt(f"{ndx_file}_{li}_{lj}")

            mask = np.zeros_like(ndx.trial_mask, dtype=float_cpu())
            f, enr_idx = ismember(ndx_c.model_set, ndx.model_set)
            f, test_idx = ismember(ndx_c.seg_set, ndx.seg_set)
            mask[np.ix_(enr_idx, test_idx)] = ndx_c.trial_mask
            cond_c = TrialScores(ndx.model_set, ndx.seg_set, mask)
            src[:, k] = cond_c.scores[ndx.trial_mask]
            k += 1

    assert np.all(np.sum(src, axis=1)), "non all trials have source info"
    return src


def read_scores(in_score_files, ndx):
    num_systems = len(in_score_files)
    in_scores = []
    for i in range(num_systems):
        logging.info("load scores: %s", in_score_files[i])
        scr = TrialScores.load_txt(in_score_files[i])
        scr = scr.align_with_ndx(ndx)
        in_scores.append(scr.scores[ndx.trial_mask][:, None])

    in_scores = np.concatenate(tuple(in_scores), axis=1)
    return in_scores


def load_models(model_file):
    fusions = []
    sources = ["CTS_CTS", "CTS_AFV", "AFV_AFV"]
    for i in range(len(sources)):
        source = sources[i]
        model_file_i = f"{model_file}_{source}.h5"
        logging.info("load model: %s", model_file_i)
        gf = GF.load(model_file_i)
        fusions.append(gf)

    return fusions


def eval_fusion(in_score_files, ndx_file, model_file, out_score_file, fus_idx):

    ndx = read_ndx(ndx_file)
    src = read_sources(ndx_file, ndx)
    in_scores = read_scores(in_score_files, ndx)

    fusions = load_models(model_file)
    logging.info("apply fusion")
    out_scores = np.zeros((in_scores.shape[0],), dtype=float_cpu())
    for i in range(3):
        mask = src[:, i] == 1
        if np.any(mask):
            out_scores[mask] = fusions[i].predict(in_scores[mask], fus_idx=fus_idx)

    scr = TrialScores(
        ndx.model_set,
        ndx.seg_set,
        np.zeros_like(ndx.trial_mask, dtype=float_cpu()),
        ndx.trial_mask,
    )
    scr.scores[ndx.trial_mask] = out_scores

    logging.info("save scores: %s" % out_score_file)
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
