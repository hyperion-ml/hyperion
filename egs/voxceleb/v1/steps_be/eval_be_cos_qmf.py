#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

"""
import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import TrialNdx, TrialScores, Utt2Info
from hyperion.utils.math import cosine_scoring
from hyperion.np.pdfs import PLDA
from hyperion.utils.list_utils import ismember
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList
from hyperion.np.score_norm import AdaptSNorm as SNorm
from hyperion.np.classifiers import BinaryLogisticRegression as LR


def get_score_filename(score_file, q_name, i, j, p):
    if q_name is not None:
        score_file = "%s_%s" % (score_file, q_name)

    if p:
        score_file = "%s-%03d-%03d" % (score_file, i, j)

    return score_file


def save_empty(score_file, q_name, i, j, p):
    score_file = get_score_filename(score_file, q_name, i, j, p)
    logging.info("saving scores to %s", score_file)
    with open(score_file, "w") as f:
        pass


def save_scores(s, score_file, q_name, i, j, p):
    score_file = get_score_filename(score_file, q_name, i, j, p)
    logging.info("saving scores to %s", score_file)
    s.save_txt(score_file)


def eval_plda(
    v_file,
    ndx_file,
    enroll_file,
    num_frames_file,
    coh_file,
    coh_v_file,
    score_file,
    qmf_file,
    preproc_file,
    model_part_idx,
    num_model_parts,
    seg_part_idx,
    num_seg_parts,
    coh_nbest,
    **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    logging.info("loading data")
    tdr = TDR(
        v_file,
        ndx_file,
        enroll_file,
        None,
        preproc,
        model_part_idx,
        num_model_parts,
        seg_part_idx,
        num_seg_parts,
    )
    logging.info("read x-vectors and ndx")
    x_e, x_t, enroll, ndx = tdr.read()
    enroll_segs = tdr.enroll.key

    parallel = num_model_parts > 1 or num_seg_parts > 1

    if not np.any(ndx.trial_mask):
        save_empty(score_file, None, model_part_idx, seg_part_idx, parallel)
        save_empty(score_file, "snorm", model_part_idx, seg_part_idx, parallel)
        if qmf_file is None:
            for q_name in ["snorm", "maxnf", "minnf", "maxcohmu", "mincohmu"]:
                save_empty(score_file, q_name, model_part_idx, seg_part_idx, parallel)
        else:
            save_empty(score_file, "qmf", model_part_idx, seg_part_idx, parallel)
        return

    logging.info("read num_frames")
    u2nf = Utt2Info.load(num_frames_file)
    enroll_nf = np.log(
        np.clip(
            u2nf.filter(enroll_segs).info.astype(float) / 100 - 2.0,
            a_min=0.1,
            a_max=12.0,  # 6.0,
        )
    )
    test_nf = np.log(
        np.clip(
            u2nf.filter(ndx.seg_set).info.astype(float) / 100 - 2.0,
            a_min=0.1,
            a_max=12.0,  # 6.0,
        )
    )
    t1 = time.time()
    logging.info("computing llr")
    scores = cosine_scoring(x_e, x_t)

    logging.info("read cohort x-vectors")
    vcr = VCR(coh_v_file, coh_file, preproc=preproc)
    x_coh, ids_coh = vcr.read()
    D_coh = PLDA.compute_stats_hard(x_coh, class_ids=ids_coh)
    x_coh = D_coh[1] / np.expand_dims(D_coh[0], axis=-1)

    t2 = time.time()
    logging.info("score cohort vs test")
    scores_coh_test = cosine_scoring(x_coh, x_t)
    logging.info("score enroll vs cohort")
    scores_enr_coh = cosine_scoring(x_e, x_coh)

    dt = time.time() - t2
    logging.info("cohort-scoring elapsed time: %.2f s.", dt)

    t2 = time.time()
    logging.info("apply s-norm")
    snorm = SNorm(nbest=coh_nbest, nbest_sel_method="highest-other-side")
    scores_norm, mu_z, s_z, mu_t, s_t = snorm(
        scores, scores_coh_test, scores_enr_coh, return_stats=True
    )
    mu_z = mu_z / s_z
    mu_t = mu_t / s_t

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    q_measures = {
        "maxnf": np.maximum(enroll_nf[:, None], test_nf[None, :]),
        "minnf": np.minimum(enroll_nf[:, None], test_nf[None, :]),
        "maxcohmu": np.maximum(mu_z, mu_t),
        "mincohmu": np.minimum(mu_z, mu_t),
    }

    f, loc = ismember(enroll, ndx.model_set)
    trial_mask = ndx.trial_mask[loc]
    s = TrialScores(enroll, ndx.seg_set, scores, score_mask=trial_mask)
    save_scores(s, score_file, None, model_part_idx, seg_part_idx, parallel)
    s.scores = scores_norm
    save_scores(s, score_file, "snorm", model_part_idx, seg_part_idx, parallel)
    if qmf_file is None:
        for q_name in ["maxnf", "minnf", "maxcohmu", "mincohmu"]:
            s.scores = q_measures[q_name]
            save_scores(s, score_file, q_name, model_part_idx, seg_part_idx, parallel)

        return

    logging.info("applying qmf")
    scores_fus = [scores.ravel()]
    scores_fus = [scores_norm.ravel()]
    for q_name in ["maxnf", "minnf", "maxcohmu", "mincohmu"]:
        scores_fus.append(q_measures[q_name].ravel())

    scores_fus = np.vstack(scores_fus).T
    lr = LR.load(qmf_file)
    scores_fus = lr.predict(scores_fus)
    scores_fus = np.reshape(scores_fus, (s.num_models, s.num_tests))
    s.scores = scores_fus
    save_scores(s, score_file, "qmf", model_part_idx, seg_part_idx, parallel)


if __name__ == "__main__":

    parser = ArgumentParser(description="Eval cosine-scoring with QMF")

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--ndx-file", default=None)
    parser.add_argument("--enroll-file", required=True)
    parser.add_argument("--num-frames-file", required=True)
    parser.add_argument("--coh-v-file", required=True)
    parser.add_argument("--coh-file", required=True)
    parser.add_argument("--coh-nbest", type=int, default=400)
    parser.add_argument("--qmf-file", default=None)
    parser.add_argument("--preproc-file", default=None)

    TDR.add_argparse_args(parser)

    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_plda(**namespace_to_dict(args))
