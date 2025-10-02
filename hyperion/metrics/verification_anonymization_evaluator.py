"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats

matplotlib.use("Agg")
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"]})
matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt

from ..hyp_defs import float_cpu
from ..np.classifiers import BinaryLogisticRegression as LR
from ..np.metrics.dcf import fast_eval_dcf_eer, fast_eval_equalized_dcf_eer
from ..np.metrics.gain_distinctness import compute_gain_distinctness
from ..np.metrics.utils import effective_prior
from ..utils import (
    EnrollmentMap,
    SegmentSet,
    SparseTrialKey,
    SparseTrialScores,
    TrialKey,
    TrialScores,
)
from ..utils.math_funcs import sigmoid
from ..utils.misc import PathLike, check_and_disable_latex


class VerificationAnonymizationEvaluator:
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
        key: Union[PathLike, TrialKey, SparseTrialKey],
        scores_orig_orig: Union[str, TrialScores, SparseTrialScores],
        scores_orig_anon: Union[str, TrialScores, SparseTrialScores],
        scores_anon_anon: Union[str, TrialScores, SparseTrialScores],
        enroll_map: Union[PathLike, EnrollmentMap, None] = None,
        anon_enroll_segments: Union[PathLike, SegmentSet, None] = None,
        anon_test_segments: Union[PathLike, SegmentSet, None] = None,
        p_tar: Union[float, List[float], np.ndarray] = 0.01,
        c_miss: Union[float, List[float], np.ndarray] = None,
        c_fa: Union[float, List[float], np.ndarray] = None,
        key_name: Optional[str] = None,
        score_name: Optional[str] = None,
        calibrate_on_orig: bool = False,
        sparse: bool = False,
        class_column: str = "speaker",
        anon_class_column: str = "pseudo_speaker",
        ref_key: Optional[Union[PathLike, TrialKey, SparseTrialKey]] = None,
        scores_ref_anon: Optional[Union[str, TrialScores, SparseTrialScores]] = None,
    ):
        if isinstance(key, str):
            logging.info("Load key: %s", key)
            if sparse:
                key = SparseTrialKey.load(key)
            else:
                key = TrialKey.load(key)

        if isinstance(scores_orig_orig, str):
            logging.info("Load scores orig. vs orig.: %s", scores_orig_orig)
            if sparse:
                scores_orig_orig = SparseTrialScores.load(scores_orig_orig)
            else:
                scores_orig_orig = TrialScores.load(scores_orig_orig)

        if isinstance(scores_orig_anon, str):
            logging.info("Load scores orig. vs anon.: %s", scores_orig_anon)
            if sparse:
                scores_orig_anon = SparseTrialScores.load(scores_orig_anon)
            else:
                scores_orig_anon = TrialScores.load(scores_orig_anon)

        if isinstance(scores_anon_anon, str):
            logging.info("Load scores anon. vs anon.: %s", scores_anon_anon)
            if sparse:
                scores_anon_anon = SparseTrialScores.load(scores_anon_anon)
            else:
                scores_anon_anon = TrialScores.load(scores_anon_anon)

        if enroll_map is not None and isinstance(enroll_map, str):
            logging.info("Load enrollment map: %s", enroll_map)
            enroll_map = EnrollmentMap.load(enroll_map)

        if anon_enroll_segments is not None and isinstance(anon_enroll_segments, str):
            logging.info("Load enroll anon meta: %s", anon_enroll_segments)
            anon_enroll_segments = SegmentSet.load(anon_enroll_segments)

        if anon_test_segments is not None and isinstance(anon_test_segments, str):
            logging.info("Load test anon meta: %s", anon_test_segments)
            anon_test_segments = SegmentSet.load(anon_test_segments)

        if ref_key is not None and isinstance(ref_key, str):
            logging.info("Load ref key: %s", ref_key)
            ref_key = TrialKey.load(ref_key)
            if scores_ref_anon is not None and isinstance(scores_ref_anon, str):
                logging.info("Load scores ref. vs anon.: %s", scores_ref_anon)
                scores_ref_anon = TrialScores.load(scores_ref_anon)

        assert ref_key is None or scores_ref_anon is not None

        self.key = key
        self.scores_orig_orig = scores_orig_orig.align_with_ndx(key)
        self.scores_orig_anon = scores_orig_anon.align_with_ndx(key)
        self.scores_anon_anon = scores_anon_anon.align_with_ndx(key)
        self.enroll_map = enroll_map
        self.anon_enroll_segments = anon_enroll_segments
        self.anon_test_segments = anon_test_segments
        self.key_name = key_name
        self.score_name = score_name
        self.calibrate_on_orig = calibrate_on_orig
        self.class_column = class_column
        self.anon_class_column = anon_class_column
        self.sparse = sparse
        self.ref_key = ref_key
        if ref_key is not None and scores_ref_anon is not None:
            self.scores_ref_anon = scores_ref_anon.align_with_ndx(ref_key)
        else:
            self.scores_ref_anon = None

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

        self.prepare_enroll_meta()
        self.make_anon_keys()
        if self.calibrate_on_orig:
            self.calibrate_scores()

    def calibrate_scores(self):
        lr = LR(prior=np.max(self.p_tar), bias_scaling=1, lambda_reg=1e-5)
        tar, non = self.scores_orig_orig.get_tar_non(self.key)
        x = np.concatenate((tar, non))
        y = np.concatenate(
            (np.ones_like(tar, dtype="int32"), np.zeros_like(non, dtype="int32"))
        )
        lr.fit(x, y)
        if self.sparse:
            pass
        else:
            self.scores_orig_orig.scores = lr(
                self.scores_orig_orig.scores.ravel()
            ).reshape(self.scores_orig_orig.scores.shape)
            self.scores_orig_anon.scores = lr(
                self.scores_orig_anon.scores.ravel()
            ).reshape(self.scores_orig_anon.scores.shape)
            self.scores_anon_anon.scores = lr(
                self.scores_anon_anon.scores.ravel()
            ).reshape(self.scores_anon_anon.scores.shape)
            if self.scores_ref_anon is not None:
                self.scores_ref_anon.scores = lr(
                    self.scores_ref_anon.scores.ravel()
                ).reshape(self.scores_ref_anon.scores.shape)

        self.lr = lr

    def prepare_enroll_meta(self):
        if self.anon_enroll_segments is None or self.enroll_map is None:
            return

        if self.class_column in self.anon_enroll_segments:
            column_names = ["id", self.class_column, self.anon_class_column]
        else:
            column_names = ["id", self.anon_class_column]

        self.enroll_map.add_columns(
            right_table=self.anon_enroll_segments,
            column_names=column_names,
            on="segmentid",
            right_on="id",
        )

        for model_id in self.enroll_map["id"].unique():
            idx = self.enroll_map["id"] == model_id
            pseudo_ids = self.enroll_map.loc[idx, self.anon_class_column].values
            uniq_pseudo_ids = np.unique(pseudo_ids)
            if len(uniq_pseudo_ids) > 1:
                self.enroll_map.loc[idx, self.anon_class_column] = "mixed"

        self.enroll_map = self.enroll_map.get_unique_modelid_df()

    def make_anon_keys(self):
        if self.enroll_map is None or self.anon_test_segments is None:
            logging.info(
                "anon_enroll_segments and anon_test_segments were not provided"
            )
            self.key_anon_anon = self.key
            self.key_cons_div = None
            self.key_cons_div_intra = None
            self.key_cons_div_inter = None
            return

        key_anon_anon = self.key.copy()
        key_cons_div = self.key.copy()
        key_cons_div_intra = self.key.copy()
        key_cons_div_inter = self.key.copy()
        enroll_pseudo = self.enroll_map.loc[
            self.key.model_set, self.anon_class_column
        ].values
        test_pseudo = self.anon_test_segments.loc[
            self.key.seg_set, self.anon_class_column
        ].values
        if self.sparse:
            pseudo_tar = sparse.lil_matrix(self.key.tar.shape, dtype="bool")
            for i, e_id in enumerate(enroll_pseudo):
                for j, t_id in enumerate(test_pseudo):
                    if e_id == t_id:
                        pseudo_tar[i, j] = True

            # Ensure pseudo_tar is in CSR format
            pseudo_tar = pseudo_tar.tocsr()
            # Logical NOT of pseudo_tar
            not_pseudo_tar = ~pseudo_tar
            # key_anon_anon
            key_anon_anon.tar = self.key.tar & not_pseudo_tar
            key_anon_anon.non = self.key.non & not_pseudo_tar
            # key_cons_div
            key_cons_div.tar = (self.key.tar & pseudo_tar) | (self.key.non & pseudo_tar)
            key_cons_div.non = (self.key.tar & not_pseudo_tar) | (
                self.key.non & not_pseudo_tar
            )
            # key_cons_div_intra
            key_cons_div_intra.tar = self.key.tar & pseudo_tar
            key_cons_div_intra.non = self.key.tar & not_pseudo_tar
            # key_cons_div_inter
            key_cons_div_inter.tar = self.key.tar & pseudo_tar
            key_cons_div_inter.non = self.key.non & not_pseudo_tar
        else:
            pseudo_tar = np.zeros_like(self.key.tar)
            for i, e_id in enumerate(enroll_pseudo):
                for j, t_id in enumerate(test_pseudo):
                    if e_id == t_id:
                        pseudo_tar[i, j] = True

            not_pseudo_tar = np.logical_not(pseudo_tar)
            key_anon_anon.tar = np.logical_and(self.key.tar, not_pseudo_tar)
            key_anon_anon.non = np.logical_and(self.key.non, not_pseudo_tar)
            key_cons_div.tar = np.logical_or(
                np.logical_and(self.key.tar, pseudo_tar),
                np.logical_and(self.key.non, pseudo_tar),
            )
            key_cons_div.non = np.logical_or(
                np.logical_and(self.key.tar, not_pseudo_tar),
                np.logical_and(self.key.non, not_pseudo_tar),
            )
            key_cons_div_intra.tar = np.logical_and(self.key.tar, pseudo_tar)
            key_cons_div_intra.non = np.logical_and(self.key.tar, not_pseudo_tar)
            key_cons_div_inter.tar = np.logical_and(self.key.tar, pseudo_tar)
            key_cons_div_inter.non = np.logical_and(self.key.non, not_pseudo_tar)

        self.key_anon_anon = key_anon_anon
        self.key_cons_div = key_cons_div
        self.key_cons_div_intra = key_cons_div_intra
        self.key_cons_div_inter = key_cons_div_inter

    def __call__(self):
        return self.compute_dcf_eer()

    def _compute_dcf_eer(self, tar, non, key_name, score_name, anon_case):
        """
        This function computes the DCF and EER metrics for the given target and non-target scores.
        Args:
            tar: target scores (np.array)
            non: non-target scores (np.array)
            key_name: name describing the key
            score_name: name describing the score
            anon_case: anonimization condition name  (e.g., 'orig vs orig', 'orig vs anon', 'anon vs anon')
        Returns:
            pandas DataFrame
        """
        logging.info("separating tar/non")

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

        if len(self.p_tar) == 1:
            eer = np.asarray([eer])
            min_dcf = np.asarray([min_dcf])
            act_dcf = np.asarray([act_dcf])

        df = pd.DataFrame(
            {
                "scores": [score_name],
                "key": [key_name],
                "anon_case": [anon_case],
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

    def compute_privacy_dcf_eer(self):
        """
        Computes DCF/EER for all privacy cases

        Returns:
           pandas DataFrame with the results
        """
        logging.info("Computing DCF/EER for orig vs orig")
        tar_orig_orig, non_orig_orig = self.scores_orig_orig.get_tar_non(self.key)
        tar_orig_anon, non_orig_anon = self.scores_orig_anon.get_tar_non(self.key)
        tar_anon_anon, non_anon_anon = self.scores_anon_anon.get_tar_non(
            self.key_anon_anon
        )
        df_oo = self._compute_dcf_eer(
            tar_orig_orig,
            non_orig_orig,
            self.key_name,
            self.score_name,
            "Privacy T(orig-orig) N(orig-orig)",
        )

        logging.info("Computing DCF/EER for orig vs anon")
        df_tar_oa_non_oo = self._compute_dcf_eer(
            tar_orig_anon,
            non_orig_orig,
            self.key_name,
            self.score_name,
            "Privacy T(orig-anon) N(orig-orig)",
        )
        df_tar_oa_non_oa = self._compute_dcf_eer(
            tar_orig_anon,
            non_orig_anon,
            self.key_name,
            self.score_name,
            "Privacy T(orig-anon) N(orig-anon)",
        )
        df_tar_oa_non_aa = self._compute_dcf_eer(
            tar_orig_anon,
            non_anon_anon,
            self.key_name,
            self.score_name,
            "Privacy T(orig-anon) N(anon-anon)",
        )

        logging.info("Computing DCF/EER for anon vs anon")
        df_tar_aa_non_oo = self._compute_dcf_eer(
            tar_anon_anon,
            non_orig_orig,
            self.key_name,
            self.score_name,
            "Privacy T(anon-anon) N(orig-orig)",
        )
        df_tar_aa_non_oa = self._compute_dcf_eer(
            tar_anon_anon,
            non_orig_anon,
            self.key_name,
            self.score_name,
            "Privacy T(anon-anon) N(orig-anon)",
        )
        df_tar_aa_non_aa = self._compute_dcf_eer(
            tar_anon_anon,
            non_anon_anon,
            self.key_name,
            self.score_name,
            "Privacy T(anon-anon) N(anon-anon)",
        )

        dfs = [
            df_oo,
            df_tar_oa_non_oo,
            df_tar_oa_non_oa,
            df_tar_oa_non_aa,
            df_tar_aa_non_oo,
            df_tar_aa_non_oa,
            df_tar_aa_non_aa,
        ]

        return pd.concat(dfs, ignore_index=True)

    def compute_cons_div_dcf_eer(self):
        """
        Computes DCF/EER for consistency vs diversity cases
        Returns:
           pandas DataFrame with the results
        """
        if self.key_cons_div is None:
            logging.warning(
                "Key for consistency/diversity is not available so we cannot calculate DCF/EER for Cons/Div cases."
            )
            return None

        logging.info("Computing DCF/EER for consistency vs diversity cases")

        tar, non = self.scores_anon_anon.get_tar_non(self.key_cons_div)
        df_cons_div = self._compute_dcf_eer(
            tar, non, self.key_name, self.score_name, "Cons/Div Intra+Inter"
        )
        tar, non = self.scores_anon_anon.get_tar_non(self.key_cons_div_intra)
        df_cons_div_intra = self._compute_dcf_eer(
            tar, non, self.key_name, self.score_name, "Cons/Div Intra"
        )
        tar, non = self.scores_anon_anon.get_tar_non(self.key_cons_div_inter)
        df_cons_div_inter = self._compute_dcf_eer(
            tar, non, self.key_name, self.score_name, "Cons/Div Inter"
        )

        dfs = [df_cons_div, df_cons_div_intra, df_cons_div_inter]
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            logging.warning("No Cons/Div cases found, returning None")
            return None
        return pd.concat(dfs, ignore_index=True)

    def compute_voice_cloning_dcf_eer(self):
        """
        Computes DCF/EER for reference vs anonymized
        Returns:
           pandas DataFrame with the results
        """
        if self.ref_key is None or self.scores_ref_anon is None:
            return None

        logging.info("Computing DCF/EER for ref. orig vs anon")
        tar_ref_anon, non_ref_anon = self.scores_ref_anon.get_tar_non(self.ref_key)
        df_tar_ra_non_ra = self._compute_dcf_eer(
            tar_ref_anon,
            non_ref_anon,
            self.key_name,
            self.score_name,
            "Voice Cloning T(ref-anon) N(ref-anon)",
        )
        tar_mean = np.mean(tar_ref_anon)
        tar_std = np.std(tar_ref_anon)
        sem = stats.sem(tar_ref_anon)  # standard error of the mean
        ci = stats.t.interval(0.95, len(tar_ref_anon) - 1, loc=0, scale=sem)
        df_tar_ra_non_ra["tar_mean"] = tar_mean
        df_tar_ra_non_ra["tar_std"] = tar_std
        df_tar_ra_non_ra["tar_ci_95"] = ci[0]
        if self.calibrate_on_orig:
            df_tar_ra_non_ra["tar_mean_non_cal"] = (
                tar_mean - self.lr.b[0]
            ) / self.lr.A[0, 0]
            df_tar_ra_non_ra["tar_std_non_cal"] = tar_std / self.lr.A[0, 0]
            df_tar_ra_non_ra["tar_ci_95_non_cal"] = ci[0] / self.lr.A[0, 0]

        return df_tar_ra_non_ra

    def compute_dcf_eer(self):
        """
        Computes DCF/EER for all cases
        Returns:
           pandas DataFrame with the results
        """
        df_priv = self.compute_privacy_dcf_eer()
        df_ref = self.compute_voice_cloning_dcf_eer()
        if df_ref is not None:
            df_priv = pd.concat([df_priv, df_ref], ignore_index=True)

        df_cons_div = self.compute_cons_div_dcf_eer()
        if df_cons_div is not None:
            df_gain = self.compute_gain_distinctness()
            df_cons_div = df_cons_div.merge(
                df_gain, on=["scores", "key", "anon_case"], how="left"
            )
            df = pd.concat([df_priv, df_cons_div], ignore_index=True)
        else:
            df = df_priv

        return df

    def compute_gain_distinctness(self):
        """
        Computes the gain in voice distinctness for the anonymization process.
        This is a placeholder for the actual implementation.

        Returns:
            pandas DataFrame with the results
        """
        if self.key_cons_div is None:
            logging.warning(
                "Key for consistency/diversity is not available so we cannot calculate Gain Voice Distinctness."
            )
            return None

        logging.info("Computing gain in distinctness")
        if not self.calibrate_on_orig:
            logging.warning(
                "Scores need to be calibrated for Gain Voice Distinctness to make sense (e.g., using --calibrate-on-orig option or calibrating the scores beforehand)"
            )

        if (
            self.class_column in self.enroll_map
            and self.class_column in self.anon_test_segments
        ):
            model_classes = self.enroll_map.loc[
                self.key.model_set, self.class_column
            ].values
            test_classes = self.anon_test_segments.loc[
                self.key.seg_set, self.class_column
            ].values
        else:
            model_classes = test_classes = None

        logging.info("Computing M_orig_orig")
        M_orig_orig, model_classes, test_classes = self.scores_orig_orig.get_class_sim(
            self.key, model_classes=model_classes, seg_classes=test_classes
        )

        common_classes = list(set(model_classes) & set(test_classes))
        M_orig_orig = M_orig_orig[np.isin(model_classes, common_classes), :][
            :, np.isin(test_classes, common_classes)
        ]
        print(
            M_orig_orig[:5, :5],
            "\n",
            M_orig_orig[5:10, 5:10],
            "\n",
            M_orig_orig[10:15, 5:15],
            "\n",
            flush=True,
        )

        if (
            self.anon_class_column in self.enroll_map
            and self.anon_class_column in self.anon_test_segments
        ):
            model_classes = self.enroll_map.loc[
                self.key.model_set, self.anon_class_column
            ].values
            test_classes = self.anon_test_segments.loc[
                self.key.seg_set, self.anon_class_column
            ].values
        else:
            model_classes = test_classes = None

        logging.info("Computing M_anon_anon")
        M_anon_anon, model_classes, test_classes = self.scores_anon_anon.get_class_sim(
            self.key_cons_div, model_classes=model_classes, seg_classes=test_classes
        )
        common_classes = list(set(model_classes) & set(test_classes))
        M_anon_anon = M_anon_anon[np.isin(model_classes, common_classes), :][
            :, np.isin(test_classes, common_classes)
        ]
        print(
            M_anon_anon[:5, :5],
            "\n",
            M_anon_anon[5:10, 5:10],
            "\n",
            M_anon_anon[10:15, 5:15],
            "\n",
            flush=True,
        )

        M_orig_orig = sigmoid(M_orig_orig)
        M_anon_anon = sigmoid(M_anon_anon)
        logging.info("Computing gain distinctness")
        gain = compute_gain_distinctness(M_orig_orig, M_anon_anon)
        df_gain = pd.DataFrame(
            {
                "scores": [self.score_name],
                "key": [self.key_name],
                "anon_case": ["Cons/Div Intra+Inter"],
                "gain_distinctness": [gain],
            }
        )
        return df_gain

    def plot_privacy_score_hist(
        self, output_path: PathLike, plot_thresholds: bool = False
    ):
        check_and_disable_latex()
        tar_orig_orig, non_orig_orig = self.scores_orig_orig.get_tar_non(self.key)
        tar_orig_anon, non_orig_anon = self.scores_orig_anon.get_tar_non(self.key)
        tar_anon_anon, non_anon_anon = self.scores_anon_anon.get_tar_non(
            self.key_anon_anon
        )

        nbins_tar = max(min(len(tar_orig_orig) // 100, 100), 5)
        nbins_non = max(min(len(non_orig_orig) // 100, 100), 5)
        plt.hist(
            tar_orig_orig,
            nbins_tar,
            histtype="step",
            density=True,
            color="b",
            linestyle="solid",
            linewidth=1.5,
            label="T(O-O)",
        )
        plt.hist(
            non_orig_orig,
            nbins_non,
            histtype="step",
            density=True,
            color="r",
            linestyle="solid",
            linewidth=1.5,
            label="N(O-O)",
        )

        plt.hist(
            tar_orig_anon,
            nbins_tar,
            histtype="step",
            density=True,
            color="c",
            linestyle="solid",
            linewidth=1.5,
            label="T(O-A)",
        )
        plt.hist(
            non_orig_anon,
            nbins_non,
            histtype="step",
            density=True,
            color="m",
            linestyle="solid",
            linewidth=1.5,
            label="N(O-A)",
        )

        plt.hist(
            tar_anon_anon,
            nbins_tar,
            histtype="step",
            density=True,
            color="g",
            linestyle="solid",
            linewidth=1.5,
            label="T(A-A)",
        )
        plt.hist(
            non_anon_anon,
            nbins_non,
            histtype="step",
            density=True,
            color="C1",
            linestyle="solid",
            linewidth=1.5,
            label="N(A-A)",
        )
        name = f"{self.score_name} {self.key_name}"
        title = f"Privacy {name}"
        if plot_thresholds:
            for p in self.p_tar:
                thr = -np.log(p / (1 - p))
                plt.axvline(
                    x=thr,
                    color="k",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"thr({p:.3f})={thr:.3f}",
                )

        plt.title(title)
        plt.xlabel("LLR score")
        plt.grid(True)
        plt.legend()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        name = re.sub(r" ", "_", name)
        file_path = output_path / f"privacy_score_hist_{name}.png"
        plt.savefig(file_path)
        plt.close()

    def plot_cons_div_score_hist(
        self, output_path: PathLike, plot_thresholds: bool = False
    ):
        check_and_disable_latex()
        tar_orig, non_orig = self.scores_orig_orig.get_tar_non(self.key)
        tar_anon_intra, non_anon_intra = self.scores_anon_anon.get_tar_non(
            self.key_cons_div_intra
        )
        _, non_anon_inter = self.scores_anon_anon.get_tar_non(self.key_cons_div_inter)

        nbins_tar = max(min(len(tar_orig) // 100, 100), 5)
        nbins_non = max(min(len(non_orig) // 100, 100), 5)
        plt.hist(
            tar_orig,
            nbins_tar,
            histtype="step",
            density=True,
            color="b",
            linestyle="solid",
            linewidth=1.5,
            label="T(O-O)",
        )
        plt.hist(
            non_orig,
            nbins_non,
            histtype="step",
            density=True,
            color="r",
            linestyle="solid",
            linewidth=1.5,
            label="N(O-O)",
        )

        plt.hist(
            tar_anon_intra,
            nbins_tar,
            histtype="step",
            density=True,
            color="c",
            linestyle="solid",
            linewidth=1.5,
            label="T(A-A)",
        )
        plt.hist(
            non_anon_intra,
            nbins_non,
            histtype="step",
            density=True,
            color="m",
            linestyle="solid",
            linewidth=1.5,
            label="N(A-A-Intra)",
        )

        plt.hist(
            non_anon_inter,
            nbins_non,
            histtype="step",
            density=True,
            color="C1",
            linestyle="solid",
            linewidth=1.5,
            label="N(A-A-Inter)",
        )
        name = f"{self.score_name} {self.key_name}"
        title = f"Cons/Div {name}"
        if plot_thresholds:
            for p in self.p_tar:
                thr = -np.log(p / (1 - p))
                plt.axvline(
                    x=thr,
                    color="k",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"thr({p:.3f})={thr:.2f}",
                )

        plt.title(title)
        plt.xlabel("LLR score")
        plt.grid(True)
        plt.legend()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        name = re.sub(r" ", "_", name)
        file_path = output_path / f"cons_div_score_hist_{name}.png"
        plt.savefig(file_path)
        plt.close()
