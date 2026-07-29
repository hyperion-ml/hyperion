"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt

from ..hyp_defs import float_cpu
from ..np.metrics.dcf import fast_eval_dcf_eer
from ..utils import TrialKey, TrialScores
from ..utils.misc import PathLike
from ..utils.trial_stats import TrialStats
from .verification_evaluator import VerificationEvaluator


class VerificationAdvAttackEvaluator(VerificationEvaluator):
    """Evaluate verification performance under adversarial attacks.

    This class extends :class:`VerificationEvaluator` with attack-conditioned
    analyses, including:
    - DCF/EER curves versus an attack statistic (for example, SNR or Linf).
    - Mining of top successful attack trials under decision-threshold criteria.
    - Optional plotting helpers for performance-vs-budget curves.

    Attributes:
        key: Trial key defining target/non-target trials.
        scores: Baseline (non-attacked) score container aligned with ``key``.
        attack_scores: Tensor with attacked scores of shape
            ``(num_attacks, num_models, num_tests)``.
        attack_stats: List of :class:`TrialStats` objects aligned with
            ``attack_scores``.
        p_tar: Target prior(s) used for DCF evaluation.
        _last_stat_name: Cache key for the last statistic matrix request.
        _last_stats_mat: Cached statistic tensor for ``_last_stat_name``.

    Examples:
        >>> from hyperion.metrics import VerificationAdvAttackEvaluator
        >>> ev = VerificationAdvAttackEvaluator(
        ...     key="trials.key",
        ...     scores="scores_clean.h5",
        ...     attack_scores=["scores_fgsm_eps1.h5", "scores_fgsm_eps2.h5"],
        ...     attack_stats=["stats_fgsm_eps1.csv", "stats_fgsm_eps2.csv"],
        ...     p_tar=[0.01, 0.05],
        ... )
        >>> df = ev.compute_dcf_eer_vs_stats(
        ...     stat_name="snr",
        ...     stat_bins=[40, 30, 20, 10, 5],
        ...     attacked_trials="all",
        ...     higher_better=True,
        ... )
        >>> print(df.head())
    """

    def __init__(
        self,
        key: Union[PathLike, TrialKey],
        scores: Union[PathLike, TrialScores],
        attack_scores: Union[
            PathLike,
            TrialScores,
            Sequence[Union[PathLike, TrialScores]],
        ],
        attack_stats: Union[
            PathLike,
            TrialStats,
            Sequence[Union[PathLike, TrialStats]],
        ],
        p_tar: Union[float, Sequence[float], np.ndarray],
        c_miss: Optional[Union[Sequence[float], np.ndarray]] = None,
        c_fa: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> None:
        """Initialize adversarial-attack evaluator.

        Args:
            key: Trial key object/path.
            scores: Baseline (non-attacked) scores object/path.
            attack_scores: One or more attacked score objects/paths.
            attack_stats: One or more attack-stat objects/paths aligned with
                ``attack_scores``.
            p_tar: Target prior(s) used in DCF computation.
            c_miss: Optional miss costs.
            c_fa: Optional false-alarm costs.
        """
        super().__init__(key, scores, p_tar, c_miss, c_fa)
        if isinstance(attack_scores, (list, tuple)):
            attack_scores = list(attack_scores)
        else:
            attack_scores = [attack_scores]

        if isinstance(attack_stats, (list, tuple)):
            attack_stats = list(attack_stats)
        else:
            attack_stats = [attack_stats]

        if len(attack_scores) != len(attack_stats):
            raise ValueError(
                "num_attack_scores({}) != num_attack_stats({})".format(
                    len(attack_scores), len(attack_stats)
                )
            )

        loaded_attack_scores = []
        for attack_score in attack_scores:
            if attack_score is None:
                raise ValueError("attack_scores cannot contain None")
            if isinstance(attack_score, (str, Path)):
                logging.info("Load attack scores: %s", attack_score)
                attack_score = TrialScores.load(attack_score)
            loaded_attack_scores.append(attack_score)
        attack_scores = loaded_attack_scores

        # align attack scores to key
        attack_scores_mat = np.zeros(
            (len(attack_scores), self.key.num_models, self.key.num_tests),
            dtype=float_cpu(),
        )

        for i, s in enumerate(attack_scores):
            s = s.align_with_ndx(self.key)
            attack_scores_mat[i] = s.scores

        loaded_attack_stats = []
        for attack_stat in attack_stats:
            if attack_stat is None:
                raise ValueError("attack_stats cannot contain None")
            if isinstance(attack_stat, (str, Path)):
                logging.info("Load attack stats: %s", attack_stat)
                attack_stat = TrialStats.load(attack_stat)
            loaded_attack_stats.append(attack_stat)
        attack_stats = loaded_attack_stats

        self.attack_scores = attack_scores_mat
        self.attack_stats = attack_stats

        self._last_stat_name = None
        self._last_stats_mat = None

    @property
    def num_attacks(self) -> int:
        """Number of attack configurations stored in ``attack_scores``."""
        return self.attack_scores.shape[0]

    @staticmethod
    def _sort_stats_bins(
        stat_bins: Union[Sequence[float], np.ndarray], higher_better: bool
    ) -> np.ndarray:
        """Sort statistic bins from best to worst.

        Args:
            stat_bins: Statistic bins.
            higher_better: True when larger values are better (for example SNR),
                False when smaller values are better (for example Linf/L2).

        Returns:
            Sorted numpy array of bins.
        """
        stat_bins = np.sort(stat_bins)
        if higher_better:
            stat_bins = stat_bins[::-1]
        return stat_bins

    def _get_stats_mat(self, stat_name: str) -> np.ndarray:
        """Get attack statistic tensor aligned with trial-score matrices.

        Args:
            stat_name: Statistic name matching a column in :class:`TrialStats`.

        Returns:
            Array with shape ``(num_attacks, num_models, num_tests)``.
        """
        if self._last_stat_name == stat_name:
            return self._last_stats_mat

        stats_mat = np.zeros(
            (self.num_attacks, self.key.num_models, self.key.num_tests),
            dtype=float_cpu(),
        )
        for i in range(self.num_attacks):
            stats_mat[i] = self.attack_stats[i].get_stats_mat(stat_name, self.key)
            self.attack_stats[i].reset_stats_mats()  # release some mem

        self._last_stat_name = stat_name
        self._last_stats_mat = stats_mat

        return self._last_stats_mat

    def compute_dcf_eer_vs_stats(
        self,
        stat_name: str,
        stat_bins: Union[Sequence[float], np.ndarray],
        attacked_trials: str = "all",
        higher_better: bool = False,
        return_df: bool = True,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        pd.DataFrame,
    ]:
        """Compute DCF/EER curves versus an attack statistic.

        Args:
            stat_name: Attack statistic name (x-axis).
            stat_bins: Budget/stat bins used to sweep operating points.
            attacked_trials: One of ``"all"``, ``"tar"``, ``"non"``.
            higher_better: Whether larger ``stat_name`` values are better.
            return_df: If True, return a DataFrame; otherwise return arrays.

        Returns:
            Either ``(stat_bins, min_dcf, act_dcf, eer)`` or a DataFrame.
        """

        # Sort bins from "best" to "worst" so the curve follows a monotonic
        # degradation trajectory as attack quality gets worse.
        stat_bins = self._sort_stats_bins(stat_bins, higher_better)

        # Select the subset of trials where we allow attacks to be applied.
        # - "all": both target and non-target trials can be attacked.
        # - "tar": only target trials can be attacked.
        # - "non": only non-target trials can be attacked.
        if attacked_trials == "all":
            mask = np.logical_or(self.key.tar, self.key.non)
        elif attacked_trials == "tar":
            mask = self.key.tar
        elif attacked_trials == "non":
            mask = self.key.non
        else:
            raise ValueError(
                f"Unsupported attacked_trials='{attacked_trials}'. "
                "Valid options are: 'all', 'tar', 'non'."
            )

        # stats_mat has one stat map per attack configuration:
        # shape = (num_attacks, num_models, num_tests).
        stats_mat = self._get_stats_mat(stat_name)

        num_bins = len(stat_bins)
        eer = np.zeros((num_bins,), dtype=float_cpu())
        min_dcf = np.zeros((num_bins, len(self.p_tar)), dtype=float_cpu())
        act_dcf = np.zeros((num_bins, len(self.p_tar)), dtype=float_cpu())

        if higher_better:
            cmp_func = lambda x, y: np.logical_and(np.greater_equal(x, y), mask)
            sort_func = lambda x: np.argmin(x)
        else:
            cmp_func = lambda x, y: np.logical_and(np.less_equal(x, y), mask)
            sort_func = lambda x: np.argmax(x)

        # We reuse a TrialScores container to evaluate each bin-specific score
        # matrix after mixing clean and adversarial scores.
        scores_attack = copy.deepcopy(self.scores)
        logging.debug(
            "max %s per attack: %s", stat_name, np.max(stats_mat, axis=(1, 2))
        )
        for b in range(num_bins):
            # Start each bin from the clean (non-attacked) scores.
            scores = copy.copy(self.scores.scores)

            # score_mask marks, for each attack config and trial, whether that
            # attack satisfies the current stat threshold and the attacked-trial
            # policy selected above.
            score_mask = cmp_func(stats_mat, stat_bins[b])
            logging.debug(
                "bin %d selected trials per attack: %s",
                b,
                np.sum(score_mask, axis=(1, 2)),
            )

            if self.num_attacks == 1:
                # Single-attack case: copy adversarial scores wherever the mask
                # is true; leave clean scores elsewhere.
                scores[score_mask[0]] = self.attack_scores[score_mask]
            else:
                # Multi-attack case:
                # For each trial (i, j), several attacks may satisfy the current
                # bin. We select one attack according to sort_func:
                # - higher_better=True  -> pick minimum stat among candidates
                # - higher_better=False -> pick maximum stat among candidates
                #
                # IMPORTANT: sort_func returns an index *within the candidate
                # subset*, so we map it back to the global attack index before
                # indexing self.attack_scores.
                for i in range(scores.shape[0]):
                    for j in range(scores.shape[1]):
                        mask_ij = score_mask[:, i, j]
                        if np.any(mask_ij):
                            cand_idx = np.flatnonzero(mask_ij)
                            k_local = sort_func(stats_mat[cand_idx, i, j])
                            k = int(cand_idx[k_local])
                            scores[i, j] = self.attack_scores[k, i, j]

            # Evaluate DCF/EER for this bin-specific score matrix.
            scores_attack.scores = scores
            tar, non = scores_attack.get_tar_non(self.key)
            min_dcf_b, act_dcf_b, eer_b, _ = fast_eval_dcf_eer(tar, non, self.p_tar)
            eer[b] = eer_b
            min_dcf[b] = min_dcf_b
            act_dcf[b] = act_dcf_b

        if not return_df:
            return stat_bins, min_dcf, act_dcf, eer

        df = pd.DataFrame({stat_name: stat_bins, "eer": eer})

        for i in range(min_dcf.shape[1]):
            pi = self.p_tar[i]
            df["min-dcf-%.3f" % (pi)] = min_dcf[:, i]
            df["act-dcf-%.3f" % (pi)] = act_dcf[:, i]

        return df

    def find_best_attacks(
        self,
        stat_name: str,
        attacked_trials: str,
        num_best: int = 10,
        min_delta: float = 1,
        attack_idx: int = 0,
        threshold: Optional[float] = None,
        prior_idx: int = 0,
        higher_better: bool = False,
        return_df: bool = True,
    ) -> Optional[
        Union[
            Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray],
            pd.DataFrame,
        ]
    ]:
        """Find top successful attacks according to a chosen statistic.

        Args:
            stat_name: Statistic name used to rank selected attacks.
            attacked_trials: One of ``"tar"`` or ``"non"``.
            num_best: Maximum number of returned trials.
            min_delta: Minimum score margin required after attack.
            attack_idx: Index of attack configuration in ``attack_scores``.
            threshold: Decision threshold; if None uses prior-based threshold.
            prior_idx: Prior index in ``self.p_tar`` when ``threshold`` is None.
            higher_better: Whether larger ``stat_name`` is preferred.
            return_df: If True, return a DataFrame; else return raw arrays/lists.

        Returns:
            ``None`` if no successful trials are found, otherwise either raw
            outputs ``(modelid, segmentid, scores, attack_scores, stat_values)``
            or a DataFrame.
        """

        if threshold is None:
            prior = self.p_tar[prior_idx]
            threshold = -np.log(prior / (1 - prior))

        scores = self.scores.scores
        attack_scores = self.attack_scores[attack_idx]
        if attacked_trials == "tar":
            success_mask = np.logical_and(
                np.logical_and(self.key.tar, scores > threshold),
                np.logical_and(
                    attack_scores < threshold, scores - attack_scores > min_delta
                ),
            )
        elif attacked_trials == "non":
            success_mask = np.logical_and(
                np.logical_and(self.key.non, scores < threshold),
                np.logical_and(
                    attack_scores > threshold, attack_scores - scores > min_delta
                ),
            )
        else:
            raise ValueError(
                f"Unsupported attacked_trials='{attacked_trials}'. "
                "Valid options are: 'tar', 'non'."
            )

        if not np.any(success_mask):
            return None

        stats_mat = self._get_stats_mat(stat_name)[attack_idx]
        ii, jj = np.where(success_mask)
        cand_stats = stats_mat[ii, jj]
        if higher_better:
            order = np.argsort(-cand_stats, kind="stable")
        else:
            order = np.argsort(cand_stats, kind="stable")

        num_best = min(len(order), num_best)
        order = order[:num_best]
        best_i = ii[order]
        best_j = jj[order]

        rmodelid = [self.key.model_set[i] for i in best_i]
        rsegmentid = [self.key.seg_set[j] for j in best_j]
        rscores = scores[best_i, best_j].astype(float_cpu(), copy=False)
        rascores = attack_scores[best_i, best_j].astype(float_cpu(), copy=False)
        rstat = stats_mat[best_i, best_j].astype(float_cpu(), copy=False)

        if not return_df:
            return rmodelid, rsegmentid, rscores, rascores, rstat

        logging.debug(
            "best attacks modelids=%s segmentids=%s scores=%s attack_scores=%s stats=%s",
            rmodelid,
            rsegmentid,
            rscores,
            rascores,
            rstat,
        )
        df = pd.DataFrame(
            {
                "modelid": rmodelid,
                "segmentid": rsegmentid,
                "scores": rscores,
                "attack-scores": rascores,
                stat_name: rstat,
            }
        )
        return df

    def save_best_attacks(
        self,
        file_path: PathLike,
        stat_name: str,
        attacked_trials: str,
        num_best: int = 10,
        min_delta: float = 1,
        attack_idx: int = 0,
        threshold: Optional[float] = None,
        prior_idx: int = 0,
        higher_better: bool = False,
    ) -> None:
        """Find top attacks and save results to CSV.

        Args:
            file_path: Output CSV path.
            stat_name: Statistic name used to rank selected attacks.
            attacked_trials: One of ``"tar"`` or ``"non"``.
            num_best: Maximum number of returned trials.
            min_delta: Minimum score margin required after attack.
            attack_idx: Index of attack configuration in ``attack_scores``.
            threshold: Decision threshold; if None uses prior-based threshold.
            prior_idx: Prior index in ``self.p_tar`` when ``threshold`` is None.
            higher_better: Whether larger ``stat_name`` is preferred.
        """

        df = self.find_best_attacks(
            stat_name,
            attacked_trials,
            num_best,
            min_delta,
            attack_idx,
            threshold,
            prior_idx,
            higher_better,
            return_df=True,
        )
        if df is None:
            return
        df.to_csv(file_path)

    @staticmethod
    def _process_perf_name(name: str) -> Tuple[int, str]:
        """Parse metric column name and map it to plot type/label.

        Args:
            name: Metric column name (for example ``eer``, ``min-dcf-0.010``).

        Returns:
            Tuple ``(plot_group, ylabel)`` where plot_group is:
            ``0`` for EER, ``1`` for minDCF, ``2`` for actDCF.
        """

        m = re.match(r"eer", name)
        if m is not None:
            return 0, "EER(%)"

        m = re.match(r"min-dcf", name)
        if m is not None:
            last = m.span()[1]
            if len(name[last:]) == 0:
                return 1, "MinDCF"
            else:
                p = float(name[last + 1 :])
                return 1, "MinDCF(p=%.3f)" % (p)

        m = re.match(r"act-dcf", name)
        if m is not None:
            last = m.span()[1]
            if len(name[last:]) == 0:
                return 2, "ActDCF"
            else:
                p = float(name[last + 1 :])
                return 2, "ActDCF(p=%.3f)" % (p)

        raise ValueError(f"Unsupported performance column name: {name}")

    @staticmethod
    def plot_dcf_eer_vs_stat_v1(
        df: Union[pd.DataFrame, List[pd.DataFrame]],
        stat_name: str,
        output_path: PathLike,
        eer_max: float = 50.0,
        min_dcf_max: float = 1.0,
        act_dcf_max: float = 1.0,
        log_x: bool = False,
        clean_ref: Optional[int] = None,
        file_format: str = "pdf",
        xlabel: str = "",
        higher_better: bool = False,
        legends: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        fmt: Sequence[str] = ("b", "r", "g", "m", "c", "y"),
        legend_loc: str = "upper left",
        legend_font: str = "medium",
        font_size: int = 10,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        """Plot EER/MinDCF/ActDCF versus stat (SNR, Linf) with matplotlib and save figs to file.

        Args:
            df: One DataFrame or list of DataFrames from
                :meth:`compute_dcf_eer_vs_stats`.
            stat_name: X-axis statistic column name.
            output_path: Output path prefix (without metric suffix/extension).
            eer_max: Y-axis upper bound for EER plots.
            min_dcf_max: Y-axis upper bound for minDCF plots.
            act_dcf_max: Y-axis upper bound for actDCF plots.
            log_x: If True, use logarithmic x-axis.
            clean_ref: Optional row index containing clean (no-attack) reference.
            file_format: Figure file format (for example ``pdf`` or ``png``).
            xlabel: X-axis label prefix.
            higher_better: Whether larger ``stat_name`` values are better.
            legends: Optional legend strings for each DataFrame.
            title: Optional figure title.
            fmt: Matplotlib format/color cycle.
            legend_loc: Matplotlib legend location.
            legend_font: Matplotlib legend font size.
            font_size: Global matplotlib font size.
            colors: Optional explicit colors overriding ``fmt`` colors.
        """
        matplotlib.rc("font", size=font_size)
        matplotlib.rc("legend", fontsize=legend_font)
        matplotlib.rc("legend", loc=legend_loc)

        if isinstance(df, pd.DataFrame):
            df = [df]
        elif not isinstance(df, list):
            df = list(df)

        num_df = len(df)
        if num_df == 0:
            raise ValueError("df must contain at least one DataFrame")
        if len(fmt) == 0:
            raise ValueError("fmt must contain at least one style string")
        if legends is not None and len(legends) < num_df:
            raise ValueError(
                "legends must have at least one entry per DataFrame: "
                f"{len(legends)} < {num_df}"
            )
        if colors is not None and len(colors) == 0:
            raise ValueError("colors must contain at least one color when provided")

        columns = [c for c in df[0].columns if c != stat_name]
        ylim = [eer_max, min_dcf_max, act_dcf_max]
        x = df[0][stat_name].values
        # remove infs
        noinf = x != np.inf
        x = x[noinf]
        if log_x:
            x[x == 0] = 0.01

        for c in columns:
            file_path = "%s_%s.%s" % (output_path, c, file_format)
            t, ylabel = VerificationAdvAttackEvaluator._process_perf_name(c)
            plt.figure()
            for i in range(num_df):
                style = fmt[i % len(fmt)]
                color = None if colors is None else colors[i % len(colors)]
                y = df[i][c].values
                if clean_ref is not None and i == 0:
                    y_clean = y[clean_ref]
                    if t == 0:
                        y_clean *= 100
                        label = None if legends is None else "original"
                        plt.hlines(
                            y_clean,
                            np.min(x),
                            np.max(x),
                            color="k",
                            linestyles="dashed",
                            linewidth=1.5,
                            label=label,
                        )

                y = y[noinf]
                if t == 0:
                    y *= 100

                label = None if legends is None else legends[i]
                if color is None:
                    plt.plot(x, y, style, linewidth=1.5, label=label)
                else:
                    plt.plot(x, y, style, linewidth=1.5, label=label, color=color)

            if log_x:
                plt.xscale("log")
                if higher_better:
                    plt.xlim(np.max(x), max(0.1, np.min(x)))
                else:
                    plt.xlim(max(0.1, np.min(x)), np.max(x))
            else:
                if higher_better:
                    plt.xlim(np.max(x), np.min(x))
                else:
                    plt.xlim(np.min(x), np.max(x))

            plt.ylim(0, ylim[t])
            plt.ylabel(ylabel)
            plt.legend()
            plt.xlabel("%s perturb. budget." % (xlabel))
            # plt.xlabel('$L_{\infty}$ perturb. budget.')
            plt.grid(True)
            if title is not None:
                plt.title(title)
            # plt.show()
            plt.tight_layout()
            plt.savefig(file_path)
            plt.clf()
            plt.close()

    @staticmethod
    def plot_dcf_eer_vs_stat_v2(
        df: Union[pd.DataFrame, List[pd.DataFrame]],
        stat_name: str,
        output_path: PathLike,
        eer_max: float = 50.0,
        dcf_max: float = 1.0,
        log_x: bool = False,
        clean_ref: Optional[int] = None,
        file_format: str = "pdf",
        xlabel: str = "",
        higher_better: bool = False,
        legends: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        fmt: Sequence[str] = ("b", "r", "g", "m", "c", "y"),
        legend_loc: str = "upper left",
        legend_font: str = "medium",
        font_size: int = 10,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        """Plot EER/MinDCF/ActDCF versus stat (SNR, Linf) with matplotlib and save figs to file.
           In this version minimum and actual DCF are plotted in the same figure.

        Args:
            df: One DataFrame or list of DataFrames from
                :meth:`compute_dcf_eer_vs_stats`.
            stat_name: X-axis statistic column name.
            output_path: Output path prefix (without metric suffix/extension).
            eer_max: Y-axis upper bound for EER plots.
            dcf_max: Y-axis upper bound for DCF plots.
            log_x: If True, use logarithmic x-axis.
            clean_ref: Optional row index containing clean (no-attack) reference.
            file_format: Figure file format (for example ``pdf`` or ``png``).
            xlabel: X-axis label prefix.
            higher_better: Whether larger ``stat_name`` values are better.
            legends: Optional legend strings for each DataFrame.
            title: Optional figure title.
            fmt: Matplotlib format/color cycle.
            legend_loc: Matplotlib legend location.
            legend_font: Matplotlib legend font size.
            font_size: Global matplotlib font size.
            colors: Optional explicit colors overriding ``fmt`` colors.
        """

        matplotlib.rc("font", size=font_size)
        matplotlib.rc("legend", fontsize=legend_font)
        matplotlib.rc("legend", loc=legend_loc)

        if isinstance(df, pd.DataFrame):
            df = [df]
        elif not isinstance(df, list):
            df = list(df)

        num_df = len(df)
        if num_df == 0:
            raise ValueError("df must contain at least one DataFrame")
        if len(fmt) == 0:
            raise ValueError("fmt must contain at least one style string")
        if legends is not None and len(legends) < num_df:
            raise ValueError(
                "legends must have at least one entry per DataFrame: "
                f"{len(legends)} < {num_df}"
            )
        if colors is not None and len(colors) == 0:
            raise ValueError("colors must contain at least one color when provided")

        columns = [
            c
            for c in df[0].columns
            if (
                c != stat_name
                and VerificationAdvAttackEvaluator._process_perf_name(c)[0] != 2
            )
        ]

        ylim = [eer_max, dcf_max, dcf_max]
        x = df[0][stat_name].values
        # remove infs
        noinf = x != np.inf
        x = x[noinf]
        if log_x:
            x[x == 0] = 0.01

        for c in columns:

            t, ylabel = VerificationAdvAttackEvaluator._process_perf_name(c)
            plt.figure()
            if t == 0:
                columns2 = [c]
                file_path = "%s_%s.%s" % (output_path, c, file_format)
            else:
                columns2 = [re.sub("min-dcf", "act-dcf", c), c]
                ylabel = re.sub("Min", "", ylabel)
                file_path = "%s_%s.%s" % (
                    output_path,
                    re.sub("min-dcf", "dcf", c),
                    file_format,
                )

            for k in range(len(columns2)):
                cc = columns2[k]
                for i in range(num_df):
                    style = fmt[i % len(fmt)]
                    color = None if colors is None else colors[i % len(colors)]
                    y = df[i][cc].values
                    if clean_ref is not None and i == 0:
                        y_clean = y[clean_ref]
                        if t == 0:
                            y_clean *= 100

                        if k == 0:
                            label = None if legends is None else "original"
                            plt.hlines(
                                y_clean,
                                np.min(x),
                                np.max(x),
                                color="k",
                                linestyles="solid",
                                linewidth=1.5,
                                label=label,
                            )
                        else:
                            plt.hlines(
                                y_clean,
                                np.min(x),
                                np.max(x),
                                color="k",
                                linestyles="dashed",
                                linewidth=1.5,
                            )

                    y = y[noinf]
                    if t == 0:
                        y *= 100

                    if k == 0:
                        label = None if legends is None else legends[i]
                        if color is None:
                            plt.plot(
                                x,
                                y,
                                style,
                                linestyle="solid",
                                linewidth=1.5,
                                label=label,
                            )
                        else:
                            plt.plot(
                                x,
                                y,
                                style,
                                linestyle="solid",
                                linewidth=1.5,
                                label=label,
                                color=color,
                            )
                    else:
                        if color is None:
                            plt.plot(x, y, style, linestyle="dashed", linewidth=1.5)
                        else:
                            plt.plot(
                                x,
                                y,
                                style,
                                linestyle="dashed",
                                linewidth=1.5,
                                color=color,
                            )

            if log_x:
                plt.xscale("log")
                if higher_better:
                    plt.xlim(np.max(x), max(0.1, np.min(x)))
                else:
                    plt.xlim(max(0.1, np.min(x)), np.max(x))
            else:
                if higher_better:
                    plt.xlim(np.max(x), np.min(x))
                else:
                    plt.xlim(np.min(x), np.max(x))

            plt.ylim(0, ylim[t])
            plt.ylabel(ylabel)
            if legends is not None:
                plt.legend()
            plt.xlabel("%s perturb. budget." % (xlabel))
            # plt.xlabel('$L_{\infty}$ perturb. budget.')
            plt.grid(True)
            if title is not None:
                plt.title(title)
            plt.tight_layout()
            # plt.show()
            plt.savefig(file_path)
            plt.clf()
            plt.close()
