"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .roc import compute_rocch, rocch2eer


def compute_dcf(p_miss, p_fa, prior, normalize=True):
    """Computes detection cost function
        DCF = prior*p_miss + (1-prior)*p_fa

    Args:
       p_miss: Vector of miss probabilities.
       p_fa:   Vector of false alarm probabilities.
       prior:  Target prior or vector of target priors.
       normalize: if true, return normalized DCF, else unnormalized.

    Returns:
       Matrix of DCF for each pair of (p_miss, p_fa) and each value of prior.
       [len(prior) x len(p_miss)]
    """

    prior = np.asarray(prior)
    if prior.ndim == 1:
        prior = prior[:, None]

    dcf = prior * p_miss + (1 - prior) * p_fa
    if normalize:
        dcf /= np.minimum(prior, 1 - prior)
    return dcf


def compute_min_dcf(tar, non, prior, normalize=True):
    """Computes minimum DCF
        min_DCF = min_t prior*p_miss(t) + (1-prior)*p_fa(t)
       where t is the decission threshold.

    Args:
      tar: Target scores.
      non: Non-target scores.
      prior: Target prior or vector of target priors.
      normalize: if true, return normalized DCF, else unnormalized.

    Returns:
      Vector Minimum DCF for each prior.
      Vector of P_miss corresponding to each min DCF.
      Vector of P_fa corresponding to each min DCF.
    """

    p_miss, p_fa = compute_rocch(tar, non)
    dcf = compute_dcf(p_miss, p_fa, prior, normalize)
    idx_min_dcf = np.argmin(dcf, axis=-1)
    if dcf.ndim == 1:
        min_dcf = dcf[idx_min_dcf]
        p_miss = p_miss[idx_min_dcf]
        p_fa = p_fa[idx_min_dcf]
    else:
        i1 = np.arange(dcf.shape[0])
        min_dcf = dcf[i1, idx_min_dcf]
        p_miss = p_miss[idx_min_dcf]
        p_fa = p_fa[idx_min_dcf]
    return min_dcf, p_miss, p_fa


def compute_act_dcf(tar, non, prior, normalize=True):
    """Computes actual DCF by making decisions assuming that scores
       are calibrated to act as log-likelihood ratios.

    Args:
      tar: Target scores.
      non: Non-target scores.
      prior: Target prior or vector of target priors.
      normalize: if true, return normalized DCF, else unnormalized.

    Returns:
      Vector actual DCF for each prior.
      Vector of P_miss corresponding to each act DCF.
      Vector of P_fa corresponding to each act DCF.
    """
    prior = np.asarray(prior)

    if prior.ndim == 1:
        assert np.all(
            prior == np.sort(prior, kind="mergesort")
        ), "priors must be in ascending order"
    else:
        prior = prior[None]

    num_priors = len(prior)

    ntar = len(tar)
    nnon = len(non)

    # thresholds
    t = -np.log(prior) + np.log(1 - prior)

    ttar = np.concatenate((t, tar))
    ii = np.argsort(ttar, kind="mergesort")
    r = np.zeros((num_priors + ntar), dtype="int32")
    r[ii] = np.arange(1, num_priors + ntar + 1)
    r = r[:num_priors]
    n_miss = r - np.arange(num_priors, 0, -1)

    tnon = np.concatenate((t, non))
    ii = np.argsort(tnon, kind="mergesort")
    r = np.zeros((num_priors + nnon), dtype="int32")
    r[ii] = np.arange(1, num_priors + nnon + 1)
    r = r[:num_priors]
    n_fa = nnon - r + np.arange(num_priors, 0, -1)

    # n_miss2 = np.zeros((num_priors,), dtype='int32')
    # n_fa2 = np.zeros((num_priors,), dtype='int32')

    # for i in range(len(t)):
    #     n_miss2[i] = np.sum(tar<t[i])
    #     n_fa2[i] = np.sum(non>t[i])

    # assert np.all(n_miss2 == n_miss)
    # assert np.all(n_fa2 == n_fa)
    # print(n_miss)
    # print(n_fa)

    p_miss = n_miss / ntar
    p_fa = n_fa / nnon

    act_dcf = prior * p_miss + (1 - prior) * p_fa
    if normalize:
        act_dcf /= np.minimum(prior, 1 - prior)

    if len(act_dcf) == 1:
        act_dcf = act_dcf[0]

    return act_dcf, p_miss, p_fa


def fast_eval_dcf_eer(tar, non, prior, normalize_dcf=True, return_probs=False):
    """Computes actual DCF, minimum DCF, EER and PRBE all togther

    Args:
      tar: Target scores.
      non: Non-target scores.
      prior: Target prior or vector of target priors.
      normalize_cdf: if true, return normalized DCF, else unnormalized.

    Returns:
      Vector Minimum DCF for each prior.
      Vector Actual DCF for each prior.
      EER value
      PREBP value
    """

    p_miss, p_fa = compute_rocch(tar, non)
    eer = rocch2eer(p_miss, p_fa)

    N_miss = p_miss * len(tar)
    N_fa = p_fa * len(non)
    prbep = rocch2eer(N_miss, N_fa)

    dcf = compute_dcf(p_miss, p_fa, prior, normalize_dcf)
    min_dcf = np.min(dcf, axis=-1)

    act_dcf, act_pmiss, act_pfa = compute_act_dcf(tar, non, prior, normalize_dcf)

    if not return_probs:
        return min_dcf, act_dcf, eer, prbep

    idx = np.argmin(dcf, axis=-1)
    min_pmiss = p_miss[idx]
    min_pfa = p_fa[idx]
    return min_dcf, act_dcf, eer, prbep, min_pmiss, min_pfa, act_pmiss, act_pfa
