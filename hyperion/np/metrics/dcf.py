"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Sequence, Tuple, Union

import numpy as np

from .roc import compute_roc, compute_rocch, roc2eer, rocch2eer
from .utils import compute_equalized_partition_weights


def compute_dcf(
    p_miss: np.ndarray,
    p_fa: np.ndarray,
    prior: Union[float, np.ndarray, Sequence[float]],
    normalize: bool = True,
) -> np.ndarray:
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


def compute_min_dcf(
    tar: np.ndarray,
    non: np.ndarray,
    prior: Union[float, np.ndarray, Sequence[float]],
    normalize: bool = True,
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """Computes minimum DCF
        min_DCF = min_t prior*p_miss(t) + (1-prior)*p_fa(t)
       where t is the decision threshold.

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


def compute_act_dcf(
    tar: np.ndarray,
    non: np.ndarray,
    prior: Union[float, np.ndarray, Sequence[float]],
    normalize: bool = True,
) -> Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:
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

    p_miss = n_miss / ntar
    p_fa = n_fa / nnon

    act_dcf = prior * p_miss + (1 - prior) * p_fa
    if normalize:
        act_dcf /= np.minimum(prior, 1 - prior)

    if len(act_dcf) == 1:
        act_dcf = act_dcf[0]

    return act_dcf, p_miss, p_fa


def fast_eval_dcf_eer(
    tar: np.ndarray,
    non: np.ndarray,
    prior: Union[float, np.ndarray, Sequence[float]],
    normalize_dcf: bool = True,
    return_probs: bool = False,
) -> Union[
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, float],
    Tuple[
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """Computes actual DCF, minimum DCF, EER and PRBEP all together.

    Args:
      tar: Target scores.
      non: Non-target scores.
      prior: Target prior or vector of target priors.
      normalize_dcf: If True, returns normalized DCF; otherwise unnormalized.
      return_probs: If True, returns miss/false-alarm operating points as well.

    Returns:
      Vector Minimum DCF for each prior.
      Vector Actual DCF for each prior.
      EER value
      PRBEP value
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


def fast_eval_equalized_dcf_eer(
    tars: Sequence[np.ndarray],
    nons: Sequence[np.ndarray],
    prior: Union[float, np.ndarray, Sequence[float]],
    normalize_dcf: bool = True,
    return_probs: bool = False,
) -> Union[
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, float],
    Tuple[
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """Computes actual DCF, minimum DCF, EER and PRBE all together, equalized by common conditions

    Args:
      tars: Tuple of Target scores, each element of the tuple is a condition
      nons: Non-target scores, each element of the tuple is a condition.
      prior: Target prior or vector of target priors.
      normalize_dcf: If True, returns normalized DCF; otherwise unnormalized.
      return_probs: If True, returns miss/false-alarm operating points as well.

    Returns:
      Vector Minimum DCF for each prior.
      Vector Actual DCF for each prior.
      EER value
      PRBEP value
    """
    ntars = [len(tar) for tar in tars]
    nnons = [len(non) for non in nons]
    tar_weights, non_weights = compute_equalized_partition_weights(ntars, nnons)
    tar_weights = np.concatenate(
        [w_i * np.ones((n_i,), dtype=float) for w_i, n_i in zip(tar_weights, ntars)]
    )
    non_weights = np.concatenate(
        [w_i * np.ones((n_i,), dtype=float) for w_i, n_i in zip(non_weights, nnons)]
    )
    tar = np.concatenate(tars)
    non = np.concatenate(nons)
    p_miss, p_fa = compute_roc(tar, non, tar_weights, non_weights)
    eer = roc2eer(p_miss, p_fa)

    N_miss = p_miss * len(tar)
    N_fa = p_fa * len(non)
    prbep = roc2eer(N_miss, N_fa)

    dcf = compute_dcf(p_miss, p_fa, prior, normalize_dcf)
    min_dcf = np.min(dcf, axis=-1)

    # act_dcf, act_pmiss, act_pfa = compute_act_dcf(tar, non, prior, normalize_dcf)
    for i, (tar, non) in enumerate(zip(tars, nons)):
        act_dcf_i, act_pmiss_i, act_pfa_i = compute_act_dcf(
            tar, non, prior, normalize_dcf
        )
        if i == 0:
            act_dcf = act_dcf_i
            act_pmiss = act_pmiss_i
            act_pfa = act_pfa_i
        else:
            act_dcf += act_dcf_i
            act_pmiss += act_pmiss_i
            act_pfa += act_pfa_i

    act_dcf /= len(tars)
    act_pmiss /= len(tars)
    act_pfa /= len(tars)

    if not return_probs:
        return min_dcf, act_dcf, eer, prbep

    idx = np.argmin(dcf, axis=-1)
    min_pmiss = p_miss[idx]
    min_pfa = p_fa[idx]
    return min_dcf, act_dcf, eer, prbep, min_pmiss, min_pfa, act_pmiss, act_pfa
