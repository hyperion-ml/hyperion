#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Evals linear GBE with uncertainty propagation.
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.io import HypDataWriter as HDW
from hyperion.helpers import ClassifTrialDataReader as TDR
from hyperion.transforms import TransformList
from hyperion.classifiers import LinearGBEUP as GBE


def eval_linear_gbe(
    iv_file,
    class2int_file,
    test_file,
    preproc_file,
    model_file,
    score_file,
    vector_score_file,
    normalize,
    eval_method,
    **kwargs
):

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr_args = TDR.filter_args(**kwargs)
    tdr = TDR(iv_file, class2int_file, test_file, preproc, **tdr_args)
    x, ndx = tdr.read()

    model = GBE.load(model_file)

    t1 = time.time()
    scores = model.predict(x, eval_method, normalize)

    dt = time.time() - t1
    num_trials = scores.shape[0] * scores.shape[1]
    logging.info(
        "Elapsed time: %.2f s. Elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    s = TrialScores(ndx.model_set, ndx.seg_set, scores.T)
    s.save(score_file)

    if vector_score_file is not None:
        h5 = HDW(vector_score_file)
        h5.write(ndx.seg_set, "", scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Eval linear Gaussian back-end",
    )

    parser.add_argument("--iv-file", dest="iv_file", required=True)
    parser.add_argument("--class2int-file", dest="class2int_file", required=True)
    parser.add_argument("--test-file", dest="test_file", required=True)
    parser.add_argument("--preproc-file", dest="preproc_file", default=None)

    TDR.add_argparse_args(parser)
    GBE.add_argparse_eval_args(parser)
    parser.add_argument("--score-file", dest="score_file", required=True)
    parser.add_argument("--vector-score-file", dest="vector_score_file", default=None)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose

    eval_linear_gbe(**vars(args))
