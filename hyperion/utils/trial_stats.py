"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import os.path as path
import logging
import copy

import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu
from .trial_ndx import TrialNdx
from .trial_key import TrialKey


class TrialStats(object):
    """Contains anciliary statistics from the trial such us quality measures like SNR

        This class was created to store statistics about adversarial attacks like
        SNR (signal-to-perturbation ratio), Linf, L2 norms of the perturbation etc.

    Attributes:
       df_stats: pandas dataframe containing the stats. The dataframe needs to include the modelid and segmentid columns

    """

    def __init__(self, df_stats):
        self.df_stats = df_stats
        assert "modelid" in df_stats.columns
        assert "segmentid" in df_stats.columns
        self.df_stats.set_index(["modelid", "segmentid"], inplace=True)
        self._stats_mats = dict()

    @classmethod
    def load(cls, file_path):
        """Loads stats file

        Args:
           file_path: stats file in csv format

        Returns:
          TrialScores object.
        """
        df = pd.read_csv(file_path)
        return cls(df)

    def save_h5(self, file_path):
        """Saves object to file.

        Args:
          file_path: CSV format file
        """
        self.df_stats.to_csv(file_path)

    def get_stats_mat(self, stat_name, ndx, raise_missing=True):
        """Returns a matrix of trial statistics sorted to match a give Ndx or Key object

        Args:
          stat_name: name of the statatistic (e.g. snr, linf), as given in the column name of the dataframe.
          ndx: Ndx or Key object

        Returns:
          Stat matrix (n_models x n_tests)
        """
        if stat_name in self._stats_mats:
            return self._stats_mats[stat_name]

        if isinstance(ndx, TrialKey):
            trial_mask = np.logical_or(ndx.tar, ndx.non)
        else:
            trial_mask = ndx.trial_mask
        stats_mat = np.zeros(trial_mask.shape, dtype=float_cpu())
        for i in range(stats_mat.shape[0]):
            for j in range(stats_mat.shape[1]):
                if trial_mask[i, j]:
                    try:
                        stats_mat[i, j] = self.df_stats.loc[
                            ndx.model_set[i], ndx.seg_set[j]
                        ][stat_name]
                    except:
                        err_str = "%s not found for %s-%s" % (
                            stat_name,
                            ndx.model_set[i],
                            ndx.seg_set[j],
                        )
                        if raise_missing:
                            raise Exception(err_str)
                        else:
                            logging.warning(err_str)

        self._stats_mats[stat_name] = stats_mat
        return stats_mat

    def reset_stats_mats(self):

        for k in list(self._stats_mats.keys()):
            del self._stats_mats[k]
