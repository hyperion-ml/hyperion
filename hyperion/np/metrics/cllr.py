"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..utils.math import neglogsigmoid
from .utils import opt_loglr


def compute_cllr(tar, non):
    """CLLR: Measure of goodness of log-likelihood-ratio detection output. This measure          ps both:
             - The quality of the score (over the whole DET curve), and
            -  The quality of the calibration
    Args:
      tar: Scores of target trials.
      non: Scores of non-target trials.

    Returns:
      CLLR

    """
    c1 = np.mean(neglogsigmoid(tar)) / np.log(2)
    c2 = np.mean(neglogsigmoid(non)) / np.log(2)

    return (c1 + c2) / 2


def compute_min_cllr(tar, non):

    tar_llr, non_llr = opt_loglr(tar, non, "raw")
    return compute_cllr(tar_llr, non_llr)
