"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
import re

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats

matplotlib.use("Agg")
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt

from ..hyp_defs import float_cpu
from ..np.metrics.dcf import fast_eval_dcf_eer, fast_eval_equalized_dcf_eer
from ..np.metrics.utils import effective_prior
from ..utils import SparseTrialKey, SparseTrialScores, TrialKey, TrialScores
from ..utils.trial_stats import TrialStats


class VerificationEvaluator:
    """Class computes performance metrics for verification problems.
       Same metrics can be obtained from fast_eval_dcf_eer functions

    Attributes:
       key: TrialKey object or file_name.
       scores: TrialScores object or file_name
       p_tar: target prior float or list/nparray sorted in ascending order
       c_miss: cost of miss
       c_fa: cost of false alarm
       key_name: name describing the key
       score_name: name describing the score
       sparse: use sparse versions of TrialScores and Keys
    """

    def __init__(
        self,
        key,
        scores,
        p_tar,
        c_miss=None,
        c_fa=None,
        key_name=None,
        score_name=None,
        sparse=False,
    ):
        if isinstance(key, str):
            logging.info("Load key: %s", key)
            if sparse:
                key = SparseTrialKey.load(key)
            else:
                key = TrialKey.load(key)

        if isinstance(scores, str):
            logging.info("Load scores: %s", scores)
            if sparse:
                scores = SparseTrialScores.load(scores)
            else:
                scores = TrialScores.load(scores)

        self.key = key
        self.scores = scores.align_with_ndx(key)
        self.key_name = key_name
        self.score_name = score_name

        # compute effective prior is c_miss and c_fa are given
        if isinstance(p_tar, float):
            p_tar = [p_tar]

        p_tar = np.asarray(p_tar)
        if c_miss is not None and c_fa is not None:
            assert len(c_miss) == len(p_tar)
            assert len(c_fa) == len(p_tar)
            c_miss = np.asarray(c_miss)
            c_fa = np.asarray(c_fa)
            p_tar = effective_prior(p_tar, c_miss, c_fa)

        self._p_tar_sort = np.argsort(p_tar)
        self.p_tar = p_tar

    def __call__(self, return_df=True):
        return self.compute_dcf_eer(return_df)

    def compute_dcf_eer(self, return_df=True):
        """
        Computes DCF/EER

        Args:
           return_df: if True, it returns the result in a pandas DataFrame object.

        Returns:
           min_dcf, act_dcf, eer tuple or pandas DataFrame
        """
        logging.info("separating tar/non")

        tar, non = self.scores.get_tar_non(self.key)
        ntar = len(tar)
        nnon = len(non)
        if ntar == 0 or nnon == 0:
            logging.warning("ntar=%d nnon=%d, no metrics will be produced", ntar, nnon)
            return None
        logging.info("computing EER/DCF")
        min_dcf, act_dcf, eer, _ = fast_eval_dcf_eer(
            tar, non, self.p_tar[self._p_tar_sort]
        )

        if len(self.p_tar) > 1:
            min_dcf[self._p_tar_sort] = min_dcf.copy()
            act_dcf[self._p_tar_sort] = act_dcf.copy()

        if not return_df:
            return min_dcf, act_dcf, eer, ntar, nnon

        if len(self.p_tar) == 1:
            eer = np.asarray([eer])
            min_dcf = np.asarray([min_dcf])
            act_dcf = np.asarray([act_dcf])

        df = pd.DataFrame(
            {
                "scores": [self.score_name],
                "key": [self.key_name],
                "eer": eer,
                "eer(%)": eer * 100,
            }
        )
        for i in range(len(min_dcf)):
            pi = self.p_tar[i]
            df["min-dcf-%.3f" % (pi)] = min_dcf[i]
            df["act-dcf-%.3f" % (pi)] = act_dcf[i]

        if len(min_dcf) > 1:
            df["min-dcf-avg"] = np.mean(min_dcf)
            df["act-dcf-avg"] = np.mean(act_dcf)

        df["num_targets"] = ntar
        df["num_nontargets"] = nnon
        tar_mean = np.mean(tar)
        tar_std = np.std(tar)
        sem = stats.sem(tar)  # standard error of the mean
        ci = stats.t.interval(0.95, len(tar) - 1, loc=0, scale=sem)
        df["tar_mean"] = tar_mean
        df["tar_mean_ci_95"] = ci[1]
        df["tar_std"] = tar_std
        non_mean = np.mean(non)
        non_std = np.std(non)
        sem = stats.sem(non)  # standard error of the mean
        ci = stats.t.interval(0.95, len(non) - 1, loc=0, scale=sem)
        df["non_mean"] = non_mean
        df["non_mean_ci_95"] = ci[1]
        df["non_std"] = non_std
        return df

    def get_tar_non(self):
        """
        Returns the target and non-target scores

        Returns:
           tar, non np.array
        """
        logging.info("separating tar/non")
        return self.scores.get_tar_non(self.key)

    def compute_equalized_dcf_eer(self, tars, nons, return_df=True):
        """
        Computes Equalized EER, Actual

        Args:
           tars: target scores tuple of np.arrays for different conditions
           nons: non-target scores tuple of np.arrays for different conditions
           return_df: if True, it returns the result in a pandas DataFrame object.

        Returns:
           min_dcf, act_dcf, eer tuple or pandas DataFrame
        """
        ntar = np.sum([len(tar) for tar in tars])
        nnon = np.sum([len(non) for non in nons])
        if ntar == 0 or nnon == 0:
            logging.warning("ntar=%d nnon=%d, no metrics will be produced", ntar, nnon)
            return None

        logging.info("computing Equalized EER/DCF")
        min_dcf, act_dcf, eer, _ = fast_eval_equalized_dcf_eer(
            tars, nons, self.p_tar[self._p_tar_sort]
        )

        if len(self.p_tar) > 1:
            min_dcf[self._p_tar_sort] = min_dcf.copy()
            act_dcf[self._p_tar_sort] = act_dcf.copy()

        if not return_df:
            return min_dcf, act_dcf, eer, ntar, nnon

        if len(self.p_tar) == 1:
            eer = np.asarray([eer])
            min_dcf = np.asarray([min_dcf])
            act_dcf = np.asarray([act_dcf])

        df = pd.DataFrame(
            {
                "scores": [self.score_name],
                "key": "equalized",
                "eer": eer,
                "eer(%)": eer * 100,
            }
        )
        for i in range(len(min_dcf)):
            pi = self.p_tar[i]
            df["min-dcf-%.3f" % (pi)] = min_dcf[i]
            df["act-dcf-%.3f" % (pi)] = act_dcf[i]

        if len(min_dcf) > 1:
            df["min-dcf-avg"] = np.mean(min_dcf)
            df["act-dcf-avg"] = np.mean(act_dcf)

        df["num_targets"] = ntar
        df["num_nontargets"] = nnon
        return df
