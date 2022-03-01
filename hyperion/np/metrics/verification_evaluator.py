"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import logging
import re
import numpy as np
import pandas as pd
import copy

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt

from ...hyp_defs import float_cpu
from ...utils import TrialKey, TrialScores
from ...utils.trial_stats import TrialStats
from .utils import effective_prior
from .dcf import fast_eval_dcf_eer


class VerificationEvaluator(object):
    """Class computes performance metrics for verification problems.
       Same metrics can be obtained from fast_eval_dcf_eer functions

    Attributes:
       key: TrialKey object or file_name.
       scores: TrialScores object or file_name
       p_tar: target prior float or list/nparray sorted in ascending order
       c_miss: cost of miss
       c_fa: cost of false alarm

    """

    def __init__(self, key, scores, p_tar, c_miss=None, c_fa=None):

        if isinstance(key, str):
            logging.info("Load key: %s" % key)
            key = TrialKey.load(key)

        if isinstance(scores, str):
            logging.info("Load scores: %s" % scores)
            scores = TrialScores.load(scores)

        self.key = key
        self.scores = scores.align_with_ndx(key)

        # compute effective prior is c_miss and c_fa are given
        if isinstance(p_tar, float):
            p_tar = [p_tar]

        p_tar = np.asarray(p_tar)
        if c_miss is not None and c_fa is not None:
            c_miss = np.asarray(c_miss)
            c_fa = np.asarray(c_fa)
            p_tar = effective_prior(p_tar, c_miss, c_fa)

        self.p_tar = p_tar

    def compute_dcf_eer(self, return_df=False):
        """
        Computes DCF/EER

        Args:
           return_df: if True, it returns the result in a pandas DataFrame object.

        Returns:
           min_dcf, act_dcf, eer tuple or pandas DataFrame
        """
        logging.info("separating tar/non")
        tar, non = self.scores.get_tar_non(self.key)
        logging.info("computing EER/DCF")
        min_dcf, act_dcf, eer, _ = fast_eval_dcf_eer(tar, non, self.p_tar)

        if not return_df:
            return min_dcf, act_dcf, eer

        if len(self.p_tar) == 1:
            eer = [eer]
            min_dcf = [min_dcf]
            act_dcf = [act_dcf]

        df = pd.DataFrame({"eer": eer})

        for i in range(len(min_dcf)):
            pi = self.p_tar[i]
            df["min-dcf-%.3f" % (pi)] = min_dcf[i]
            df["act-dcf-%.3f" % (pi)] = act_dcf[i]

        return df


class VerificationAdvAttackEvaluator(VerificationEvaluator):
    """Class computes performance metrics for verification problems
       under adversarial attacks

    Attributes:
       key: TrialKey object or file_name.
       scores: TrialScores object or file_name for the system without attack
       attack_scores: TrialScores list or file_name list for the system under attack,
                      each element of the list are the scores for a different attack for the same
                      trial list. For example, FGSM attacks for several eps values.
       attack_stats: Pandas DataFrame or file_name list, one for each element in attack scores.
                     The dataframe contain statistics about the adversarial signals like snr, linf, etc.
       p_tar: target prior float or list/nparray sorted in ascending order
       c_miss: cost of miss
       c_fa: cost of false alarm

    """

    def __init__(
        self, key, scores, attack_scores, attack_stats, p_tar, c_miss=None, c_fa=None
    ):
        super(VerificationAdvAttackEvaluator, self).__init__(
            key, scores, p_tar, c_miss, c_fa
        )
        if not isinstance(attack_scores, list):
            attack_scores = [attack_scores]
        if not isinstance(attack_stats, list):
            attack_stats = [attack_stats]

        assert len(attack_scores) == len(
            attack_stats
        ), "num_attack_scores({}) != num_attack_stats({})".format(
            len(attack_scores), len(attack_stats)
        )

        if isinstance(attack_scores[0], str):
            l = []
            for file_path in attack_scores:
                logging.info("Load attack scores: %s" % file_path)
                scores = TrialScores.load(file_path)
                l.append(scores)
            attack_scores = l

        # align attack scores to key
        attack_scores_mat = np.zeros(
            (len(attack_scores), self.key.num_models, self.key.num_tests),
            dtype=float_cpu(),
        )

        for i, s in enumerate(attack_scores):
            s = s.align_with_ndx(self.key)
            attack_scores_mat[i] = s.scores

        if isinstance(attack_stats[0], str):
            l = []
            for file_path in attack_stats:
                logging.info("Load attack stats: %s" % file_path)
                scores = TrialStats.load(file_path)
                l.append(scores)
            attack_stats = l

        self.attack_scores = attack_scores_mat
        self.attack_stats = attack_stats

        self._last_stat_name = None
        self._last_stats_mat = None

    @property
    def num_attacks(self):
        return self.attack_scores.shape[0]

    @staticmethod
    def _sort_stats_bins(stat_bins, higher_better):
        """Sorts the statistics from best to worst,
           e.g., for snr higher is better,
                 for perturbation linf, lower is better

        Args:
           stats_bins: statistics bins
           higher_better: True for snr, False for Linf, L2, ...

        Returns:
           Sorted bins
        """
        stat_bins = np.sort(stat_bins)
        if higher_better:
            stat_bins = stat_bins[::-1]
        return stat_bins

    def _get_stats_mat(self, stat_name):
        """Gets the statistics (SNR, Linf) in matrix format aligned with
           the score matrix.

        Args:
           stat_name: name of the statistic matching the column name in pandas DataFrame.

        Returns:
           Stats matrix aligned with TrialKey
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
        stat_name,
        stat_bins,
        attacked_trials="all",
        higher_better=False,
        return_df=False,
    ):
        """
        Computes DCF/EER versus SNR/Linf/etc curves

        Args:
           stat_name: stat name for x-axis matching pandas DataFrame column name.
           stat_bins: bins to sweep to generate the curve
           attacked_trials: str in ['all', 'tar', 'non'] indicating if we want curves where
                            we attack all trials, only targets or only nontargets
           higher_better: Indicates if the stat_name (x-axis) is better if is high.
                          True for SNR, false for Linf,L2,...
           return_df: if True, it returns the result in a pandas DataFrame object.

        Returns:
           stat_bins, min_dcf, act_dcf, eer arrays or pandas DataFrame
        """

        # sort stats bins from best to worse
        stat_bins = self._sort_stats_bins(stat_bins, higher_better)

        if attacked_trials == "all":
            mask = np.logical_or(self.key.tar, self.key.non)
        elif attacked_trials == "tar":
            mask = self.key.tar
        else:
            mask = self.key.non

        # extract the stats and align with the score matrices
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

        scores_attack = copy.deepcopy(self.scores)
        print(np.max(stats_mat, axis=(1, 2)))
        for b in range(num_bins):
            # we initialize the score matrix with non-attack scores
            scores = copy.copy(self.scores.scores)
            # find attack scores that meet the bin criteria
            score_mask = cmp_func(stats_mat, stat_bins[b])
            print(b, np.sum(score_mask, axis=(1, 2)))

            if self.num_attacks == 1:
                scores[score_mask[0]] = self.attack_scores[score_mask]
            else:
                for i in range(scores.shape[0]):
                    for j in range(scores.shape[1]):
                        mask_ij = score_mask[:, i, j]
                        if np.any(mask_ij):
                            k = sort_func(stats_mat[mask_ij, i, j])
                            scores[i, j] = self.attack_scores[k, i, j]

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
        stat_name,
        attacked_trials,
        num_best=10,
        min_delta=1,
        attack_idx=0,
        threshold=None,
        prior_idx=0,
        higher_better=False,
        return_df=False,
    ):
        """
        Find the best attacks from the point of view of some of the stats. E.g.,
        Attacks with best SNR or with lowest Linf.

        Args:
           stat_name: stat name for x-axis matching pandas DataFrame column name.
           attacked_trials: str in ['all', 'tar', 'non'] indicating if we want curves where
                            we attack all trials, only targets or only nontargets
           num_best: number of best trials
           min_delta: adv. trials need to have score<threshold-min_delta for targets or
                      score>threshold+min_delta for non-targets
           threshold: decision threshold, if None, it uses -logit(p_tar)
           prior_idx: indicates whichi of the priors in the p_tar array to use to compute threshold.
           higher_better: Indicates if the stat_name (x-axis) is better if is high.
                          True for SNR, false for Linf,L2,...
           return_df: if True, it returns the result in a pandas DataFrame object.

        Returns:
             modelid, test_segmentid, original_scores, adversarial_scores, stat_values arrays
             or pandas DataFrame
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
        else:
            success_mask = np.logical_and(
                np.logical_and(self.key.non, scores < threshold),
                np.logical_and(
                    attack_scores > threshold, attack_scores - scores > min_delta
                ),
            )

        if not np.any(success_mask):
            return None

        stats_mat = self._get_stats_mat(stat_name)[attack_idx]
        sorted_stats = np.sort(stats_mat[success_mask])
        if higher_better:
            sorted_stats = sorted_stats[::-1]

        num_best = min(len(sorted_stats), num_best)
        stats_threshold = sorted_stats[num_best - 1]
        if higher_better:
            success_mask = np.logical_and(success_mask, stats_mat >= stats_threshold)
        else:
            success_mask = np.logical_and(success_mask, stats_mat <= stats_threshold)

        rmodelid = []
        rsegmentid = []
        rscores = np.zeros((num_best,), dtype=float_cpu())
        rascores = np.zeros((num_best,), dtype=float_cpu())
        rstat = np.zeros((num_best,), dtype=float_cpu())
        k = 0
        nz = success_mask.nonzero()
        for i, j in zip(nz[0], nz[1]):
            rmodelid.append(self.key.model_set[i])
            rsegmentid.append(self.key.seg_set[j])
            rscores[k] = scores[i, j]
            rascores[k] = attack_scores[i, j]
            rstat[k] = stats_mat[i, j]
            k += 1
            if k == num_best:
                break

        if not return_df:
            return rmodelid, rsegmentid, rscores, rascores, rstat

        print(rmodelid, rsegmentid, rscores, rascores, rstat)
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
        file_path,
        stat_name,
        attacked_trials,
        num_best=10,
        min_delta=1,
        attack_idx=0,
        threshold=None,
        prior_idx=0,
        higher_better=False,
    ):
        """
        Find the best attacks from the point of view of some of the stats. E.g.,
        Attacks with best SNR or with lowest Linf and saves to csv file

        Args:
           stat_name: stat name for x-axis matching pandas DataFrame column name.
           attacked_trials: str in ['all', 'tar', 'non'] indicating if we want curves where
                            we attack all trials, only targets or only nontargets
           num_best: number of best trials
           min_delta: adv. trials need to have score<threshold-min_delta for targets or
                      score>threshold+min_delta for non-targets
           threshold: decision threshold, if None, it uses -logit(p_tar)
           prior_idx: indicates whichi of the priors in the p_tar array to use to compute threshold.
           higher_better: Indicates if the stat_name (x-axis) is better if is high.
                          True for SNR, false for Linf,L2,...
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
    def _process_perf_name(name):

        m = re.match(r"eer", name)
        if m is not None:
            return 0, "EER(\%)"

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

    @staticmethod
    def plot_dcf_eer_vs_stat_v1(
        df,
        stat_name,
        output_path,
        eer_max=50.0,
        min_dcf_max=1.0,
        act_dcf_max=1.0,
        log_x=False,
        clean_ref=None,
        file_format="pdf",
        xlabel="",
        higher_better=False,
        legends=None,
        title=None,
        fmt=["b", "r", "g", "m", "c", "y"],
        legend_loc="upper left",
        legend_font="medium",
        font_size=10,
        colors=None,
    ):
        """Plot EER/MinDCF/ActDCF versus stat (SNR, Linf) with matplotlib and save figs to file.

        Args:
           df: DataFrame from compute_dcf_eer_vs_stats method
           stat_name: stat name as in dataframe column
           output_path: output file path without extension
           eer_max: y axis maximum value for EER
           min_dcf: y axis maximum value for MinDCF
           act_dcf: y axis maximum value for ActDCF
           log_x: if True x-axis is plot in log scale
           clean_ref: row number containing the values for non-attack result,
                      if None, if won't plot the non-attack result
           file_format: format of the mage file default: pdf
           xlabel: label for x-axis
           higher_better: True for SNR, False for LInf, Lx
           legends: legend strings
           title: figure title
           fmt: format strings to indicate colors for each line.
           legend_loc: legend location
           legend_font: font size for legend
           font_size: global font size
           colors: colors in string format
        """
        matplotlib.rc("font", size=font_size)
        matplotlib.rc("legend", fontsize=legend_font)
        matplotlib.rc("legend", loc=legend_loc)

        if not isinstance(df, list):
            df = [df]

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
            for i in range(len(df)):
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
                plt.plot(x, y, fmt[i], linewidth=1.5, label=label)

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
        df,
        stat_name,
        output_path,
        eer_max=50.0,
        dcf_max=1.0,
        log_x=False,
        clean_ref=None,
        file_format="pdf",
        xlabel="",
        higher_better=False,
        legends=None,
        title=None,
        fmt=["b", "r", "g", "m", "c", "y"],
        legend_loc="upper left",
        legend_font="medium",
        font_size=10,
        colors=None,
    ):
        """Plot EER/MinDCF/ActDCF versus stat (SNR, Linf) with matplotlib and save figs to file.
           In this version minimum and actual DCF are plotted in the same figure.

        Args:
           df: DataFrame from compute_dcf_eer_vs_stats method
           stat_name: stat name as in dataframe column
           output_path: output file path without extension
           eer_max: y axis maximum value for EER
           min_dcf: y axis maximum value for DCF
           log_x: if True x-axis is plot in log scale
           clean_ref: row number containing the values for non-attack result,
                      if None, if won't plot the non-attack result
           file_format: format of the mage file default: pdf
           xlabel: label for x-axis
           higher_better: True for SNR, False for LInf, Lx
           legends: legend strings
           title: figure title
           fmt: format strings to indicate colors for each line.
           legend_loc: legend location
           legend_font: font size for legend
           font_size: global font size
           colors: colors in string format
        """

        matplotlib.rc("font", size=font_size)
        matplotlib.rc("legend", fontsize=legend_font)
        matplotlib.rc("legend", loc=legend_loc)

        if not isinstance(df, list):
            df = [df]

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
                for i in range(len(df)):
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
                        if colors is None:
                            plt.plot(
                                x,
                                y,
                                fmt[i],
                                linestyle="solid",
                                linewidth=1.5,
                                label=label,
                            )
                        else:
                            plt.plot(
                                x,
                                y,
                                fmt[i],
                                linestyle="solid",
                                linewidth=1.5,
                                label=label,
                                color=colors[i],
                            )
                    else:
                        if colors is None:
                            plt.plot(x, y, fmt[i], linestyle="dashed", linewidth=1.5)
                        else:
                            plt.plot(
                                x,
                                y,
                                fmt[i],
                                linestyle="dashed",
                                linewidth=1.5,
                                color=colors[i],
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
