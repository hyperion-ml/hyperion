#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_scores import TrialScores

from face_video_trial_data_reader import FaceVideoTrialDataReaderV1 as TDR
from face_be_utils import *


def eval_be(
    ref_v_file,
    enr_v_file,
    test_v_file,
    ndx_file,
    enroll_file,
    test_file,
    score_file,
    self_att_a,
    **kwargs
):

    logging.info("loading data")

    tdr = TDR(ref_v_file, enr_v_file, test_v_file, ndx_file, enroll_file, test_file)
    x_ref, x_e, x_t, enroll, ndx = tdr.read()

    # in some files the face embedding extractor don't find a face in the place indicated by the labels
    # we replace the embedding with the average of the embeddings obtained with automatic facedet
    x_ref = fill_missing_ref_with_facedet_avg(x_ref, x_e, enroll)
    x_t = fill_missing_test_with_zero(x_t, ndx.seg_set)

    # compute the average for each enrollment video
    x_ref = compute_median_per_vid(x_ref)

    x_t = compute_self_att_embeds(x_t, self_att_a)

    # concat the lists of embedding matrices
    x_ref, seg_idx_ref = concat_embed_matrices(x_ref)
    x_t, seg_idx_t = concat_embed_matrices(x_t)

    t1 = time.time()
    logging.info("computing llr")
    scores = cosine_scr(x_ref, x_t)

    scores = max_combine_scores_NvsM(scores, seg_idx_ref, seg_idx_t)

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save_txt(score_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Eval back-end for SR19 Video condition",
    )

    parser.add_argument("--ref-v-file", required=True)
    parser.add_argument("--enr-v-file", required=True)
    parser.add_argument("--test-v-file", required=True)
    parser.add_argument("--ndx-file", default=None)
    parser.add_argument("--enroll-file", required=True)
    parser.add_argument("--test-file", default=None)

    TDR.add_argparse_args(parser)

    parser.add_argument("--self-att-a", type=float, default=1)
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    assert args.test_file is not None or args.ndx_file is not None
    eval_be(**vars(args))
