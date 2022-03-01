"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .roc import compute_rocch, rocch2eer


def compute_eer(tar, non):
    """Computes equal error rate.

    Args:
      tar: Scores of target trials.
      non: Scores of non-target trials.

    Returns:
      EER
    """
    p_miss, p_fa = compute_rocch(tar, non)
    return rocch2eer(p_miss, p_fa)


def compute_prbep(tar, non):
    """Computes precission-recall break-even point
       where #FA == #Miss

    Args:
      tar: Scores of target trials.
      non: Scores of non-target trials.

    Returns:
      PREBP value
    """
    p_miss, p_fa = compute_rocch(tar, non)
    N_miss = p_miss * len(tar)
    N_fa = p_fa * len(non)
    return rocch2eer(N_miss, N_fa)
