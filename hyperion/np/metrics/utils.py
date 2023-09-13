"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Utility functions to evaluate performance
"""

import numpy as np

from ...hyp_defs import float_cpu
from ...utils.math_funcs import logsumexp, softmax


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


def lre_priors(num_classes, p_tar, p_oos=0.0):
    """Returns all prior distributions as needed for LRE language detection task.

    Args:
      num_classes: number of target classes.
      p_tar: target prior.
      p_oos: prior of out-of-set hypothesis.

    Returns
      Matrix of priors P with shape (num_classes, num_classes) or (num_classes, num_classes+1) if p_oos > 0, where P(i,:) are the priors for the case that class i is the target class.
    """
    I = np.eye(num_classes)
    ones = np.ones((num_classes, num_classes))
    priors = (1 - p_tar - p_oos) * (ones - I) / (num_classes - 1) + p_tar * I
    if p_oos > 0:
        priors_oos = p_oos * np.ones((num_classes, 1))
        priors = np.concatenate((priors, priors_oos), axis=-1)

    return priors


def loglk2llr(loglk, priors, target_idx=None):
    """Converts log-likelihoods to detection log-likelihood ratios.

    Args:
     loglk: log-likelihood matrix P(x_t | class_i) with shape = (num_samples, num_classes)
     priors:  vector of prior probabilities, positive, sum up to one.
     target_idx: index of the target class, the other classes are assumed to be non-target classes,
                 it can be also a list of indexes to consider multiple target classes.
                 if None, it returns matrix with LLR w.r.t. all classes.

    Returns:
     Matrix of log-likelihood ratios LLR = log P(x_t | class_i) / log P(x_t / non-class_i) with
      shape (num_samples, num_target_classes), if None, num_target_classes=num_classes

    """

    num_classes = loglk.shape[1]
    assert num_classes == len(priors), "wrong prior length"
    assert np.all(priors >= 0), "negative priors present"
    assert np.abs(np.log(np.sum(priors))) > 0.001, "priors does not sum up to one"
    assert target_idx is None or target_idx >= 0 and target_idx < num_classes
    if target_idx is None:
        target_idx = np.arange(num_classes)
    elif isinstance(target_idx, int):
        target_idx = [target_idx]

    num_target_classes = len(target_idx)
    llr = np.zeros((loglk.shape[0], num_target_classes), dtype=loglk.dtype)
    for i, target in enumerate(target_idx):
        priors_i = np.copy(priors)
        priors[target] = 0
        priors /= np.sum(priors)
        priors[target] = 1
        llr = llr + np.log(priors)
        non_idx = np.concatenate(
            (np.arange(target_idx), np.arange(target_idx + 1, num_classes))
        )
        llr[:, i] = loglk[:, target] - logsumexp(llglk[:, non_idx], axis=-1)

    return llr


def loglk2posterior(loglk, priors):
    """Converts log-likelihoods to posteriors

    Args:
     loglk: log-likelihood matrix P(x_t | class_i) with shape = (num_samples, num_classes)
     priors:  vector of prior probabilities, positive, sum up to one.

    Returns:
     Matrix of posteriors with shape = (num_samples, num_classes)

    """

    num_classes = loglk.shape[1]
    assert num_classes == len(priors), "wrong prior length"
    assert np.all(priors >= 0), "negative priors present"
    assert np.abs(np.log(np.sum(priors))) > 0.001, "priors does not sum up to one"

    log_post = loglk + np.log(priors)
    return softmax(log_post, axis=-1)


def lre_loglk2llr(loglk, p_tar, p_oos=0):
    """Converts log-likelihoods to detection log-likelihood ratios suitable for LRE.

    Args:
     loglk: log-likelihood matrix P(x_t | class_i) with shape = (num_samples, num_classes)
     priors:  prior prob that each language is the target language
     p_oos: prior prob that test language is out-of-set.

    Returns:
     Matrix of log-likelihood ratios LLR = log P(x_t | class_i) / log P(x_t / non-class_i) with
      shape (num_samples, classes),

    """

    num_tar_classes = loglk.shape[-1]
    if p_oos == 0:
        num_tar_classes -= 1
    priors = llr_priors(num_tar_classes, p_tar, p_oos)
    llr = np.zeros_like((loglk.shape[0], num_tar_classes), dtype=loglk.dtype)
    for i in range(num_tar_classes):
        llr[:, i] = loglk2llr(loglk, priors[i], target_idx=i)

    return llr


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
