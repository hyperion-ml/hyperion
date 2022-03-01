"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Utility functions to evaluate performance
"""

import numpy as np

from ...hyp_defs import float_cpu


def effective_prior(p_tar, c_miss, c_fa):
    """This function adjusts a given prior probability of target p_targ,
    to incorporate the effects of a cost of miss, cmiss, and a cost of false-alarm, cfa.

    Args:
       p_tar: target prior
       c_miss: cost of miss
       c_fa: cost of false alarm
    Returns:
       Effective prior

    """
    beta = p_tar * c_miss / (1 - p_tar) / c_fa
    p_eff = beta / (1 + beta)
    return p_eff


def pavx(y):
    """PAV: Pool Adjacent Violators algorithm. Non-paramtetric optimization subject to monotonicity.

     ghat = pav(y)
     fits a vector ghat with nondecreasing components to the
     data vector y such that sum((y - ghat).^2) is minimal.
     (Pool-adjacent-violators algorithm).

    Author: This code is and adaptation from Bosaris Toolkit and
            it is a simplified version of the 'IsoMeans.m' code made available
            by Lutz Duembgen at:
              http://www.imsv.unibe.ch/~duembgen/software

    Args:
     y: uncalibrated scores

    Returns:
      Calibrated scores
      Width of pav bins, from left to right
         (the number of bins is data dependent)
      Height: corresponding heights of bins (in increasing order)

    """
    assert isinstance(y, np.ndarray)

    n = len(y)
    assert n > 0
    index = np.zeros(y.shape, dtype=int)
    l = np.zeros(y.shape, dtype=int)
    # An interval of indices is represented by its left endpoint
    # ("index") and its length "len"
    ghat = np.zeros_like(y)

    ci = 0
    index[ci] = 0
    l[ci] = 1
    ghat[ci] = y[0]
    # ci is the number of the interval considered currently.
    # ghat[ci] is the mean of y-values within this interval.
    for j in range(1, n):
        # a new index intervall, {j}, is created:
        ci = ci + 1
        index[ci] = j
        l[ci] = 1
        ghat[ci] = y[j]
        # while ci >= 1 and ghat[np.maximum(ci-1,0)] >= ghat[ci]:
        while ci >= 1 and ghat[ci - 1] >= ghat[ci]:
            # "pool adjacent violators":
            nw = l[ci - 1] + l[ci]
            ghat[ci - 1] = ghat[ci - 1] + (l[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            l[ci - 1] = nw
            ci = ci - 1

    height = np.copy(ghat[: ci + 1])
    width = l[: ci + 1]

    # Now define ghat for all indices:
    while n >= 1:
        for j in range(index[ci], n):
            ghat[j] = ghat[ci]

        n = index[ci]
        ci = ci - 1

    return ghat, width, height


def opt_loglr(tar, non, method="laplace"):
    """Non-parametric optimization of score to log-likelihood-ratio mapping.

    Taken from Bosaris toolkit.
          Niko Brummer and Johan du Preez, Application-Independent Evaluation of Speaker Detection, Computer Speech and Language, 2005

    Args:
      tar: target scores.
      non: non-target scores.
      method: laplace(default, avoids inf log-LR)/raw

    Returns:
       Calibrated tar and non-tar log-LR
    """
    ntar = len(tar)
    nnon = len(non)
    n = ntar + nnon

    scores = np.concatenate((tar, non))
    p_ideal = np.zeros((n,), dtype=float_cpu())
    p_ideal[:ntar] = 1

    sort_idx = np.argsort(scores, kind="mergesort")
    # print(scores)
    # print(sort_idx)
    p_ideal = p_ideal[sort_idx]

    if method == "laplace":
        # The extra targets and non-targets at scores of -inf and +inf effectively
        # implement Laplace's rule of succession to avoid log LRs of infinite magnitudes.
        p_ideal = np.concatenate(([1, 0], p_ideal, [1, 0]))

    p_opt, _, _ = pavx(p_ideal)

    if method == "laplace":
        p_opt = p_opt[2:-2]

    # Posterior to loglr
    # This LR is prior-independent in the sense that if we weight the data with a synthetic prior,
    # it makes no difference to the optimizing LR mapping.
    # (A synthetic prior DOES change Popt: The posterior log-odds changes by an additive term. But this
    # this cancels again when converting to log LR. )
    # print(p_opt)
    post_log_odds = np.log(p_opt) - np.log(1 - p_opt)
    prior_log_odds = np.log(ntar / nnon)
    llr = post_log_odds - prior_log_odds
    llr += 1e-6 * np.arange(n) / n

    llr[sort_idx] = llr
    tar_llr = llr[:ntar]
    non_llr = llr[ntar:]

    return tar_llr, non_llr
