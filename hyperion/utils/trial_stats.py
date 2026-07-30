"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu
from .misc import PathLike
from .trial_key import TrialKey
from .trial_ndx import TrialNdx

TrialIndex = Union[TrialKey, TrialNdx]


class TrialStats:
    """Contains anciliary statistics from the trial such us quality measures like SNR

        This class was created to store statistics about adversarial attacks like
        SNR (signal-to-perturbation ratio), Linf, L2 norms of the perturbation etc.

    Attributes:
       df_stats: pandas dataframe containing the stats. The dataframe needs to include the modelid and segmentid columns

    """

    def __init__(self, df_stats: pd.DataFrame) -> None:
        self.df_stats = df_stats
        assert "modelid" in df_stats.columns
        assert "segmentid" in df_stats.columns
        self.df_stats.set_index(["modelid", "segmentid"], inplace=True)
        self._stats_mats: dict[tuple[str, int], np.ndarray] = {}

    @classmethod
    def load(cls, file_path: PathLike) -> "TrialStats":
        """Loads stats file

        Args:
           file_path: stats file in csv format

        Returns:
          TrialScores object.
        """
        df = pd.read_csv(file_path)
        return cls(df)

    def save(self, file_path: PathLike) -> None:
        """Saves object to file.

        Args:
          file_path: CSV format file
        """
        self.df_stats.to_csv(file_path)

    def get_stats_mat(
        self,
        stat_name: str,
        ndx: TrialIndex,
        raise_missing: bool = True,
    ) -> np.ndarray:
        """Returns a matrix of trial statistics sorted to match a give Ndx or Key object

        Args:
          stat_name: name of the statatistic (e.g. snr, linf), as given in the column name of the dataframe.
          ndx: Ndx or Key object

        Returns:
          Stat matrix (n_models x n_tests)
        """
        if stat_name not in self.df_stats.columns:
            raise KeyError(f"stat '{stat_name}' not found in trial stats table")

        cache_key = (stat_name, id(ndx))
        if cache_key in self._stats_mats:
            return self._stats_mats[cache_key]

        if isinstance(ndx, TrialKey):
            trial_mask = np.logical_or(ndx.tar, ndx.non)
        else:
            trial_mask = ndx.trial_mask
        stats_mat = np.zeros(trial_mask.shape, dtype=float_cpu())
        for i in range(stats_mat.shape[0]):
            for j in range(stats_mat.shape[1]):
                if trial_mask[i, j]:
                    model_id = ndx.model_set[i]
                    seg_id = ndx.seg_set[j]
                    try:
                        stats_mat[i, j] = self.df_stats.at[
                            (model_id, seg_id), stat_name
                        ]
                    except KeyError:
                        err_str = f"{stat_name} not found for {model_id}-{seg_id}"
                        if raise_missing:
                            raise KeyError(err_str)
                        logging.warning(err_str)

        self._stats_mats[cache_key] = stats_mat
        return stats_mat

    def reset_stats_mats(self) -> None:

        self._stats_mats.clear()
