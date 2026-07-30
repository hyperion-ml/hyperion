#!/usr/bin/env python
"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Evals calibration
"""

import logging

import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.np.classifiers import BinaryLogisticRegression as LR
from hyperion.np.metrics import compute_act_dcf, compute_min_dcf
from hyperion.utils.misc import PathLike
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores


def eval_verification_calibration(
    in_score_file: PathLike,
    ndx_file: PathLike,
    model_file: PathLike,
    out_score_file: PathLike,
) -> None:
    """Apply a trained logistic calibration model to trial scores."""

    logging.info("load ndx: %s", ndx_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file)

    logging.info("load scores: %s", in_score_file)
    scr = TrialScores.load(in_score_file)
    scr = scr.align_with_ndx(ndx)

    logging.info("load model: %s", model_file)
    lr = LR.load(model_file)
    logging.info("apply calibration")
    s_cal = lr.predict(scr.scores.ravel())
    scr.scores = np.reshape(s_cal, scr.scores.shape)

    logging.info("save scores: %s", out_score_file)
    scr.save(out_score_file)


def main() -> None:
    """Parse CLI arguments and run verification score calibration."""
    parser = ArgumentParser(
        description="Evaluate logistic calibration for verification"
    )

    parser.add_argument(
        "--in-score-file",
        required=True,
        help="input trial score file to calibrate",
    )
    parser.add_argument(
        "--out-score-file",
        required=True,
        help="output trial score file with calibrated scores",
    )
    parser.add_argument(
        "--ndx-file",
        required=True,
        help="trial index/key file used to align scores",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="trained logistic-regression calibration model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_verification_calibration(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
