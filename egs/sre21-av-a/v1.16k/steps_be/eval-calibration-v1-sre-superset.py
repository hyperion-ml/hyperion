#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals calibration
"""

import sys
import os
import time
import logging
from jsonargparse import ArgumentParser, namespace_to_dict
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import TrialScores, TrialKey, TrialNdx, Utt2Info
from hyperion.utils.list_utils import ismember
from hyperion.metrics import compute_act_dcf, compute_min_dcf
from hyperion.classifiers import BinaryLogisticRegression as LR


def read_ndx_and_scores(ndx_file, score_file):
    logging.info("load scores: %s", score_file)
    scr = TrialScores.load_txt(score_file)
    logging.info("load key: %s", ndx_file)
    try:
        ndx = TrialKey.load_txt(ndx_file).to_ndx()
    except:
        ndx = TrialNdx.load_txt(ndx_file)

    scr = scr.align_with_ndx(ndx)
    return ndx, scr.scores[ndx.trial_mask]


def read_nenr_sre16(nenr_file, ndx):
    nenr = Utt2Info.load(nenr_file)
    nenr = nenr.filter(ndx.model_set)
    ns = np.sum(ndx.trial_mask)
    nenr = np.zeros((ns, 2), dtype=float_cpu())
    for i, c in enumerate(["1", "3"]):
        mask = np.zeros_like(ndx.trial_mask, dtype=float_cpu())
        mask[nenr.info == c, :] = 1.0
        cond_c = TrialScores(ndx.model_set, ndx.seg_set, mask)
        nenr_c = cond_c.scores[ndx.trial_mask]
        nenr[:, i] = nenr_c

    assert np.all(np.sum(nenr, axis=1)), "non all trials have nenroll info"
    return nenr


def read_nenr(ndx_file, ndx):
    ns = np.sum(ndx.trial_mask)
    nenr = np.zeros((ns, 2), dtype=float_cpu())
    for i, c in enumerate(["nenr1", "nenr3"]):
        logging.info("load key: %s", f"{ndx_file}_{c}")
        try:
            ndx_c = TrialKey.load_txt(f"{ndx_file}_{c}").to_ndx()
        except:
            ndx_c = TrialNdx.load_txt(f"{ndx_file}_{c}")
        mask = np.zeros_like(ndx.trial_mask, dtype=float_cpu())
        f, idx = ismember(ndx_c.model_set, ndx.model_set)
        mask[idx, :] = 1.0
        cond_c = TrialScores(ndx.model_set, ndx.seg_set, mask)
        nenr_c = cond_c.scores[ndx.trial_mask]
        nenr[:, i] = nenr_c

    assert np.all(np.sum(nenr, axis=1)), "non all trials have nenroll info"
    return nenr


def read_langs(ndx_file, ndx):
    ns = np.sum(ndx.trial_mask)
    scr_langs = np.zeros((ns, 6), dtype=float_cpu())
    langs = ["ENG", "CMN", "YUE"]
    k = 0
    for i in range(len(langs)):
        for j in range(i, len(langs)):
            li = langs[i]
            lj = langs[j]
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
            scr_langs[:, k] = cond_c.scores[ndx.trial_mask]
            k += 1

    # for other langs we put equal prob for all conditions
    idx_other = np.sum(scr_langs, axis=1) == 0
    scr_langs[idx_other] = 1 / 6
    return scr_langs


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


def make_yueyue(ndx):
    ns = np.sum(ndx.trial_mask)
    scr_langs = np.zeros((ns, 6), dtype=float_cpu())
    scr_langs[:, -1] = 1
    return scr_langs


def make_ctscts(ndx):
    ns = np.sum(ndx.trial_mask)
    src = np.zeros((ns, 3), dtype=float_cpu())
    src[:, 0] = 1
    return src


def eval_calibration(in_score_file, ndx_file, model_file, out_score_file):

    ndx, scores = read_ndx_and_scores(ndx_file, in_score_file)
    nenr = read_nenr(ndx_file, ndx)
    langs = read_langs(ndx_file, ndx)
    src = make_ctscts(ndx)

    x = np.concatenate((scores[:, None], nenr, langs, src), axis=1)

    logging.info("load model: %s" % model_file)
    lr = LR.load(model_file)
    logging.info("apply calibration")
    s_cal = lr.predict(x)
    scr = TrialScores(
        ndx.model_set,
        ndx.seg_set,
        np.zeros_like(ndx.trial_mask, dtype=float_cpu()),
        ndx.trial_mask,
    )
    scr.scores[scr.score_mask] = s_cal

    logging.info("save scores: %s" % out_score_file)
    scr.save_txt(out_score_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evals linear calibration")

    parser.add_argument("--in-score-file", required=True)
    parser.add_argument("--out-score-file", required=True)
    parser.add_argument("--ndx-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_calibration(**vars(args))
