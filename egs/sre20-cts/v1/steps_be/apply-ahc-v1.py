#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
"""

import sys
import os
import argparse
import time
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.helpers import VectorClassReader as VCR

# from hyperion.utils.trial_ndx import TrialNdx
# from hyperion.utils.trial_scores import TrialScores
# from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList
from hyperion.score_norm import AdaptSNorm as SNorm
from hyperion.clustering import AHC
from hyperion.utils import Utt2Info
from hyperion.classifiers import BinaryLogisticRegression as LR


def apply_ahc(
    v_file,
    input_list,
    output_list,
    preproc_file,
    model_file,
    plda_type,
    cal_file,
    score_hist_file,
    threshold,
    pool_method,
    coh_nbest,
    class_prefix,
    **kwargs
):

    logging.info("loading data")
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    # Read data
    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(v_file, input_list, preproc=preproc, **vcr_args)
    x, class_ids = vcr.read()

    logging.info("loading plda model: %s" % (model_file))
    model = F.load_plda(plda_type, model_file)

    t1 = time.time()
    logging.info("computing llr")
    scores = model.llr_1vs1(x, x)
    # scores = model.llr_NvsM(x, x, method=pool_method, ids1=class_ids, ids2=class_ids)

    dt = time.time() - t1
    num_trials = x.shape[0] ** 2
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    t2 = time.time()
    logging.info("apply s-norm")
    snorm = SNorm(nbest=coh_nbest)
    scores = snorm.predict(scores, scores, scores)
    dt = time.time() - t2
    logging.info("s-norm elapsed time: %.2f s." % (dt))

    if cal_file is not None:
        logging.info("load calibration model: %s" % cal_file)
        lr = LR.load(cal_file)
        logging.info("apply calibration")
        s_cal = lr.predict(scores.ravel())
        scores = np.reshape(s_cal, scores.shape)

    if score_hist_file is not None:
        plt.hist(
            scores.ravel(),
            1000,
            histtype="step",
            density=True,
            color="b",
            linestyle="solid",
            linewidth=1.5,
        )
        plt.axvline(x=threshold, color="k")
        plt.xlabel("LLR score")
        plt.grid(True)
        # plt.legend()
        plt.savefig(score_hist_file)

    dt = time.time() - t1
    logging.info(
        "total-scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    ahc = AHC(method="average", metric="llr")
    ahc.fit(scores)
    class_ids_ahc = ahc.get_flat_clusters(threshold, criterion="threshold")

    logging.info("saving clustering to %s" % (output_list))
    if class_prefix is not None:
        class_ids_ahc = np.asarray(
            ["%s-%06d" % (class_prefix, i) for i in class_ids_ahc]
        )
    u2c_out = Utt2Info.create(vcr.u2c.key, class_ids_ahc)
    u2c_out.save(output_list, sep=" ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        description="Does AHC from PLDA + SNorm + Calibration scores",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-list", required=True)
    parser.add_argument("--preproc-file", default=None)

    VCR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)
    parser.add_argument(
        "--pool-method",
        type=str.lower,
        default="vavg-lnorm",
        choices=["book", "vavg", "vavg-lnorm", "savg"],
    )

    parser.add_argument("--cal-file", default=None)
    parser.add_argument("--score-hist-file", default=None)
    parser.add_argument("--coh-nbest", type=int, default=100)
    parser.add_argument("--class-prefix", default=None)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    apply_ahc(**vars(args))
