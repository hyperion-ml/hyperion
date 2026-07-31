"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
import os
import re
from typing import List, Optional, Sequence, Tuple, Union

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
from ..np.metrics.dcf_plot import NormDCFPlot
from ..np.metrics.det_plot import DETPlot
from ..np.metrics.utils import effective_prior
from ..utils import SparseTrialKey, SparseTrialScores, TrialKey, TrialScores
from ..utils.misc import PathLike
from ..utils.trial_stats import TrialStats


class VerificationEvaluator:
    """Compute verification metrics and optionally update DET/DCF plots.

    Attributes:
       key: Trial key object used to identify target/non-target trials.
       scores: Trial scores aligned to ``key``.
       p_tar: Target prior(s), sorted via ``_p_tar_sort`` for metric computation.
       key_name: Optional label describing the key set.
       score_name: Optional label describing the score set.
       dcf_plot: Optional DET plot object used to visualize ROC/DCF operating points.
       norm_dcf_plot: Optional normalized-DCF plot object.
       color: Default plotting color used for both plot objects.
       det_line_type: Default line style used for DET curve plotting.
       line_width: Default line width used for DET and normalized-DCF curves.
       plot_det_min_dcf: Whether to plot DET min-DCF points.
       plot_det_act_dcf: Whether to plot DET act-DCF points.
       plot_legend: Optional explicit legend string for plotting methods.

    Examples:
       >>> from hyperion.np.metrics.det_plot import DETPlot
       >>> from hyperion.np.metrics.dcf_plot import NormDCFPlot
       >>> det_plot = DETPlot("sre10", plot_title="DET", priors=[0.01, 0.1])
       >>> norm_plot = NormDCFPlot(min_prior=1e-3, max_prior=0.5, plot_title="Norm DCF")
       >>> ev = VerificationEvaluator(
       ...     key="trials.key",
       ...     scores="scores.h5",
       ...     p_tar=[0.01, 0.1],
       ...     key_name="eval",
       ...     score_name="systemA",
       ...     dcf_plot=det_plot,
       ...     norm_dcf_plot=norm_plot,
       ...     color="b",
       ...     det_line_type="-",
       ...     line_width=1.5,
       ... )
       >>> df = ev.compute_dcf_eer(return_df=True)
       >>> det_plot.save("det.png", dpi=200)
       >>> norm_plot.save("norm_dcf.png", dpi=200)
    """

    def __init__(
        self,
        key: Union[PathLike, TrialKey, SparseTrialKey],
        scores: Union[PathLike, TrialScores, SparseTrialScores],
        p_tar: Union[float, Sequence[float], np.ndarray],
        c_miss: Optional[Union[Sequence[float], np.ndarray]] = None,
        c_fa: Optional[Union[Sequence[float], np.ndarray]] = None,
        key_name: Optional[str] = None,
        score_name: Optional[str] = None,
        dcf_plot: Optional[DETPlot] = None,
        norm_dcf_plot: Optional[NormDCFPlot] = None,
        color: str = "b",
        det_line_type: str = "-",
        line_width: float = 1.5,
        plot_det_min_dcf: bool = True,
        plot_det_act_dcf: bool = True,
        plot_legend: Optional[str] = None,
        sparse: bool = False,
    ) -> None:
        """Initializes the evaluator.

        Args:
           key: Key object/path (dense or sparse).
           scores: Score object/path (dense or sparse).
           p_tar: Target prior or list/array of priors.
           c_miss: Optional miss costs (same length as ``p_tar``).
           c_fa: Optional false-alarm costs (same length as ``p_tar``).
           key_name: Optional key label for reports.
           score_name: Optional score label for reports.
           dcf_plot: Optional :class:`DETPlot` instance.
           norm_dcf_plot: Optional :class:`NormDCFPlot` instance.
           color: Default color for DET and normalized-DCF plots.
           det_line_type: Default DET line style.
           line_width: Default line width for DET and normalized-DCF curves.
           plot_det_min_dcf: If True, plot min-DCF points on DET.
           plot_det_act_dcf: If True, plot act-DCF points on DET.
           plot_legend: Optional legend string used as system name in plots.
           sparse: If True, loads sparse key/score classes from file paths.
        """
        if isinstance(key, (str, os.PathLike)):
            logging.info("Load key: %s", key)
            if sparse:
                key = SparseTrialKey.load(key)
            else:
                key = TrialKey.load(key)

        if isinstance(scores, (str, os.PathLike)):
            logging.info("Load scores: %s", scores)
            if sparse:
                scores = SparseTrialScores.load(scores)
            else:
                scores = TrialScores.load(scores)

        self.key = key
        self.scores = scores.align_with_ndx(key)
        self.key_name = key_name
        self.score_name = score_name
        self.dcf_plot = dcf_plot
        self.norm_dcf_plot = norm_dcf_plot
        self.color = color
        self.det_line_type = det_line_type
        self.line_width = line_width
        self.plot_det_min_dcf = plot_det_min_dcf
        self.plot_det_act_dcf = plot_det_act_dcf
        self.plot_legend = plot_legend

        # compute effective prior if c_miss and c_fa are given
        p_tar = np.atleast_1d(np.asarray(p_tar, dtype=float_cpu()))
        if c_miss is not None and c_fa is not None:
            c_miss = np.atleast_1d(np.asarray(c_miss, dtype=float_cpu()))
            c_fa = np.atleast_1d(np.asarray(c_fa, dtype=float_cpu()))
            if c_miss.size == 1 and p_tar.size > 1:
                c_miss = np.full_like(p_tar, c_miss.item(), dtype=float_cpu())
            if c_fa.size == 1 and p_tar.size > 1:
                c_fa = np.full_like(p_tar, c_fa.item(), dtype=float_cpu())
            if c_miss.size != p_tar.size or c_fa.size != p_tar.size:
                raise ValueError(
                    "c_miss and c_fa must have the same length as p_tar, "
                    "or be scalars"
                )
            p_tar = effective_prior(p_tar, c_miss, c_fa)

        self._p_tar_sort = np.argsort(p_tar)
        self.p_tar = p_tar

    def __call__(self, return_df: bool = True) -> Optional[
        Union[
            Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, int, int],
            pd.DataFrame,
        ]
    ]:
        """Alias of :meth:`compute_dcf_eer`.

        Args:
           return_df: If True, return a pandas DataFrame; else return raw values.

        Returns:
           Metrics DataFrame or tuple ``(min_dcf, act_dcf, eer, ntar, nnon)``.
        """
        return self.compute_dcf_eer(return_df)

    def compute_dcf_eer(self, return_df: bool = True) -> Optional[
        Union[
            Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, int, int],
            pd.DataFrame,
        ]
    ]:
        """Computes minDCF, actDCF, and EER for current key/scores.

        Args:
           return_df: If True, returns a pandas DataFrame with metrics/statistics.

        Returns:
           DataFrame or tuple ``(min_dcf, act_dcf, eer, ntar, nnon)``.
           Returns None when target/non-target sets are empty.
        """
        logging.info("separating tar/non")

        tar, non = self.scores.get_tar_non(self.key)
        ntar = len(tar)
        nnon = len(non)
        if ntar == 0 or nnon == 0:
            logging.warning("ntar=%d nnon=%d, no metrics will be produced", ntar, nnon)
            return None

        if self.plot_legend is not None:
            system_name = self.plot_legend
        elif self.key_name is not None and self.score_name is not None:
            system_name = f"{self.key_name} {self.score_name}"
        elif self.key_name is not None:
            system_name = self.key_name
        elif self.score_name is not None:
            system_name = self.score_name
        else:
            system_name = ""

        if self.dcf_plot is not None:
            self.dcf_plot.plot_curve_from_scores(
                tar_scores=tar,
                non_scores=non,
                method="rocch",
                system_name=system_name,
                color=self.color,
                line_type=self.det_line_type,
                line_width=self.line_width,
                min_dcf=self.plot_det_min_dcf,
                act_dcf=self.plot_det_act_dcf,
            )

        if self.norm_dcf_plot is not None:
            self.norm_dcf_plot.set_system_from_scores(
                tar_scores=tar,
                non_scores=non,
                system_name=system_name,
                color=self.color,
            )
            # keep default line styles configured by NormDCFPlot methods
            self.norm_dcf_plot.plot_both_dcf(
                color=self.color, line_width=self.line_width
            )

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
            eer = np.atleast_1d(np.asarray(eer))
            min_dcf = np.atleast_1d(np.asarray(min_dcf))
            act_dcf = np.atleast_1d(np.asarray(act_dcf))

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

    def get_tar_non(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns target and non-target scores aligned with current key.

        Returns:
           Tuple ``(tar, non)`` with numpy arrays of target/non-target scores.
        """
        logging.info("separating tar/non")
        return self.scores.get_tar_non(self.key)

    def compute_equalized_dcf_eer(
        self,
        tars: Sequence[np.ndarray],
        nons: Sequence[np.ndarray],
        return_df: bool = True,
    ) -> Optional[
        Union[
            Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, int, int],
            pd.DataFrame,
        ]
    ]:
        """Computes equalized minDCF/actDCF/EER from per-condition scores.

        Args:
           tars: Sequence of target score arrays for each condition.
           nons: Sequence of non-target score arrays for each condition.
           return_df: If True, returns a pandas DataFrame with metrics.

        Returns:
           DataFrame or tuple ``(min_dcf, act_dcf, eer, ntar, nnon)``.
           Returns None when target/non-target sets are empty.
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
            eer = np.atleast_1d(np.asarray(eer))
            min_dcf = np.atleast_1d(np.asarray(min_dcf))
            act_dcf = np.atleast_1d(np.asarray(act_dcf))

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
