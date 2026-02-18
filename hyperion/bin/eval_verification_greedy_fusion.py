#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Evals greedy fusion
"""
import logging
import os
import sys
import time
from typing import List

import numpy as np
from jsonargparse import ActionYesNo, ArgumentParser, namespace_to_dict

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.np.classifiers import GreedyFusionBinaryLR as GF
from hyperion.utils.misc import PathLike
from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores


def sanity_check(scores: np.ndarray) -> None:
    """Log the normalized correlation matrix across system scores."""
    R = np.dot(scores.T, scores) / scores.shape[0]
    norms = 1 / np.sqrt(np.diag(R))
    R = R * norms
    R = R * norms[:, None]
    logging.info(f"R={R}")


def eval_verification_greedy_fusion(
    in_score_files: List[PathLike],
    ndx_file: PathLike,
    model_file: PathLike,
    out_score_file: PathLike,
    fus_idx: int,
    check_sanity: bool,
) -> None:
    """Fuse verification scores using a trained greedy-fusion LR model."""

    logging.info("load ndx: %s", ndx_file)
    try:
        ndx = TrialNdx.load(ndx_file)
    except:
        ndx = TrialKey.load(ndx_file)

    num_systems = len(in_score_files)
    in_scores = []
    for i in range(num_systems):
        logging.info("load scores: %s", in_score_files[i])
        scr = TrialScores.load(in_score_files[i])
        scr = scr.align_with_ndx(ndx)
        in_scores.append(scr.scores.ravel()[:, None])

    in_scores = np.concatenate(tuple(in_scores), axis=1)
    if check_sanity:
        sanity_check(in_scores)

    logging.info("load model: %s", model_file)
    gf = GF.load(model_file)
    logging.info("apply fusion")
    s_fus = gf.predict(in_scores, fus_idx=fus_idx)
    scr.scores = np.reshape(s_fus, scr.scores.shape)

    logging.info("save scores: %s", out_score_file)
    scr.save(out_score_file)


def main() -> None:
    """Parse CLI arguments and run greedy score fusion."""
    parser = ArgumentParser(
        description="Evals linear fusion from greedy fusion trainer"
    )

    parser.add_argument(
        "--in-score-files",
        required=True,
        nargs="+",
        help="input trial score files to fuse",
    )
    parser.add_argument(
        "--out-score-file",
        required=True,
        help="output fused trial score file",
    )
    parser.add_argument(
        "--ndx-file",
        required=True,
        help="trial index/key file used to align scores",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="trained greedy-fusion logistic-regression model",
    )
    parser.add_argument(
        "--fus-idx",
        required=True,
        type=int,
        help="fusion stage index to apply from the greedy-fusion model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level (0=warning, 1=info, 2=debug, 3=trace)",
    )
    parser.add_argument(
        "--check-sanity",
        default=False,
        action=ActionYesNo,
        help="log score correlation matrix before fusion",
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_verification_greedy_fusion(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
