"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import numpy as np
import h5py

from .score_norm import ScoreNorm


class AdaptSNorm(ScoreNorm):
    """Class for adaptive S-Norm.

    Attributes:
      nbest: number of samples selected to compute the statistics for each trial
        by the adaptive algorith
      nbest_discard: discard the nbest trials with higher scores, which could
        be actual target trials.
      std_floor: floor for standard deviations.
    """

    def __init__(
        self,
        nbest=100,
        nbest_discard=0,
        nbest_sel_method="highest-other-side",
        **kwargs,
    ):
        super().__init__(*kwargs)
        self.nbest = nbest
        self.nbest_discard = nbest_discard
        self.nbest_sel_method = nbest_sel_method

    def predict(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test=None,
        mask_enr_coh=None,
        return_stats=False,
    ):
        """Normalizes the scores.

        Args:
          scores: score matrix enroll vs. test.
          scores_coh_test: score matrix cohort vs. test.
          scores_enr_coh: score matrix enroll vs cohort.
          mask_coh_test: binary matrix to mask out target trials
            from cohort vs test matrix.
          mask_enr_coh: binary matrix to mask out target trials
            from enroll vs. cohort matrix.

        """

        assert scores_enr_coh.shape[1] == scores_coh_test.shape[0]
        assert self.nbest_discard < scores_enr_coh.shape[1]
        if self.nbest > scores_enr_coh.shape[1] - self.nbest_discard:
            nbest = scores_enr_coh.shape[1] - self.nbest_discard
        else:
            nbest = self.nbest

        if mask_coh_test is not None:
            scores_coh_test[mask_coh_test == False] = 0
        if mask_enr_coh is not None:
            scores_enr_coh[mask_enr_coh == False] = 0

        if self.nbest_sel_method == "highest-other-side":
            return self._norm_highest_other_side(
                scores,
                scores_coh_test,
                scores_enr_coh,
                mask_coh_test,
                mask_enr_coh,
                return_stats,
                nbest,
            )
        elif self.nbest_sel_method == "highest-same-side":
            return self._norm_highest_same_side(
                scores,
                scores_coh_test,
                scores_enr_coh,
                mask_coh_test,
                mask_enr_coh,
                return_stats,
                nbest,
            )
        #     best_idx = np.flipud(np.argsort(scores_coh_test, axis=0))[
        #         self.nbest_discard : self.nbest_discard + nbest
        #     ]
        # elif self.nbest_sel_method == "highest-same-side":
        #     best_idx = np.fliplr(np.argsort(scores_enr_coh, axis=1))[
        #         :, self.nbest_discard : self.nbest_discard + nbest
        #     ].T
        else:
            raise Exception(f"invalid cohort selection method {self.nbest_sel_method}")

        # scores_z_norm = np.zeros_like(scores)
        # for i in range(scores.shape[1]):
        #     best_idx_i = best_idx[:, i]

        #     best_scores_i = scores_enr_coh[:, best_idx_i]
        #     mu_z = np.mean(best_scores_i, axis=1, keepdims=True)

        #     if mask_enr_coh is None:
        #         s_z = np.std(best_scores_i, axis=1, keepdims=True)
        #     else:
        #         norm = np.mean(mask_enr_coh[:, best_idx_i], axis=1, keepdims=True)
        #         mu_z /= norm
        #         s_z = np.sqrt(
        #             np.mean(best_scores_i ** 2, axis=1, keepdims=True) / norm
        #             - mu_z ** 2
        #         )

        #     s_z = np.clip(s_z, a_min=1e-5, a_max=None)
        #     if not self.norm_var:
        #         s_z = 1.0

        #     scores_z_norm[:, i] = (scores[:, i] - mu_z.T) / s_z.T

        # if self.nbest_sel_method == "highest-other-side":
        #     best_idx = np.fliplr(np.argsort(scores_enr_coh, axis=1))[
        #         :, self.nbest_discard : self.nbest_discard + nbest
        #     ]
        # elif self.nbest_sel_method == "highest-same-side":
        #     best_idx = np.flipud(np.argsort(scores_coh_test, axis=0))[
        #         self.nbest_discard : self.nbest_discard + nbest
        #     ].T
        # else:
        #     raise Exception(f"invalid cohort selection method {self.nbest_sel_method}")

        # scores_t_norm = np.zeros_like(scores)
        # for i in range(scores.shape[0]):
        #     best_idx_i = best_idx[i]
        #     best_scores_i = scores_coh_test[best_idx_i, :]
        #     mu_t = np.mean(best_scores_i, axis=0, keepdims=True)

        #     if mask_coh_test is None:
        #         s_t = np.std(best_scores_i[best_idx_i, :], axis=0, keepdims=True)
        #     else:
        #         norm = np.mean(mask_coh_test[best_idx_i, :], axis=0, keepdims=True)
        #         mu_t /= norm
        #         s_t = np.sqrt(
        #             np.mean(best_scores_i[best_idx_i, :] ** 2, axis=0, keepdims=True)
        #             / norm
        #             - mu_z ** 2
        #         )

        #     s_t = np.clip(s_t, a_min=1e-5, a_max=None)
        #     if not self.norm_var:
        #         s_t = 1.0

        #     scores_t_norm[i, :] = (scores[i, :] - mu_t) / s_t

        # scores_norm = (scores_z_norm + scores_t_norm) / np.sqrt(2)

        # if return_stats:
        #     return scores_norm, mu_z, s_z, mu_t, s_t
        # else:
        #     return scores_norm

    def _norm_highest_other_side(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test,
        mask_enr_coh,
        return_stats,
        nbest,
    ):

        if return_stats:
            mu_z = np.zeros_like(scores)
            mu_t = np.zeros_like(scores)
            if self.norm_var:
                s_z = np.zeros_like(scores)
                s_t = np.zeros_like(scores)
            else:
                s_z = s_t = 1.0

        scores_z_norm = np.zeros_like(scores)
        best_idx = np.flipud(np.argsort(scores_coh_test, axis=0))[
            self.nbest_discard : self.nbest_discard + nbest
        ]
        for i in range(scores.shape[1]):
            best_idx_i = best_idx[:, i]

            best_scores_i = scores_enr_coh[:, best_idx_i]
            mu_z_i = np.mean(best_scores_i, axis=1, keepdims=False)

            if mask_enr_coh is None:
                s_z_i = np.std(best_scores_i, axis=1, keepdims=False)
            else:
                norm = np.mean(mask_enr_coh[:, best_idx_i], axis=1, keepdims=False)
                mu_z_i /= norm
                s_z_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=1, keepdims=False) / norm
                    - mu_z_i ** 2
                )

            s_z_i = np.clip(s_z_i, a_min=1e-5, a_max=None)
            if not self.norm_var:
                s_z_i = 1.0

            scores_z_norm[:, i] = (scores[:, i] - mu_z_i) / s_z_i
            if return_stats:
                mu_z[:, i] = mu_z_i
                if self.norm_var:
                    s_z[:, i] = s_z_i

        scores_t_norm = np.zeros_like(scores)
        best_idx = np.fliplr(np.argsort(scores_enr_coh, axis=1))[
            :, self.nbest_discard : self.nbest_discard + nbest
        ]
        for i in range(scores.shape[0]):
            best_idx_i = best_idx[i]
            best_scores_i = scores_coh_test[best_idx_i, :]
            mu_t_i = np.mean(best_scores_i, axis=0, keepdims=False)

            if mask_coh_test is None:
                s_t_i = np.std(best_scores_i, axis=0, keepdims=False)
            else:
                norm = np.mean(mask_coh_test[best_idx_i, :], axis=0, keepdims=False)
                mu_t_i /= norm
                s_t_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=0, keepdims=False) / norm
                    - mu_t_i ** 2
                )

            s_t_i = np.clip(s_t_i, a_min=1e-5, a_max=None)
            if not self.norm_var:
                s_t_i = 1.0

            scores_t_norm[i, :] = (scores[i, :] - mu_t_i) / s_t_i
            if return_stats:
                mu_t[i, :] = mu_t_i
                if self.norm_var:
                    s_t[i, :] = s_t_i

        scores_norm = (scores_z_norm + scores_t_norm) / np.sqrt(2)
        if return_stats:
            return scores_norm, mu_z, s_z, mu_t, s_t
        else:
            return scores_norm

    def _norm_highest_same_side(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test,
        mask_enr_coh,
        return_stats,
        nbest,
    ):

        if return_stats:
            mu_z = np.zeros_like(scores)
            mu_t = np.zeros_like(scores)
            if self.norm_var:
                s_z = np.zeros_like(scores)
                s_t = np.zeros_like(scores)
            else:
                s_z = s_t = 1.0

        best_idx = np.fliplr(np.argsort(scores_enr_coh, axis=1))[
            :, self.nbest_discard : self.nbest_discard + nbest
        ]

        scores_z_norm = np.zeros_like(scores)
        for i in range(scores.shape[0]):
            best_idx_i = best_idx[i]
            best_scores_i = scores_enr_coh[:, best_idx_i]
            mu_z_i = np.mean(best_scores_i, axis=1, keepdims=False)

            if mask_coh_test is None:
                s_z_i = np.std(best_scores_i, axis=1, keepdims=False)
            else:
                norm = np.mean(mask_enr_coh[:, best_idx_i], axis=1, keepdims=False)
                mu_z_i /= norm
                s_z_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=1, keepdims=False) / norm
                    - mu_z_i ** 2
                )

            s_z_i = np.clip(s_z_i, a_min=1e-5, a_max=None)
            if not self.norm_var:
                s_z_i = 1.0

            scores_z_norm[:, i] = (scores[:, i] - mu_z_i) / s_z_i
            if return_stats:
                mu_z[:, i] = mu_z_i
                if self.norm_var:
                    s_z[:, i] = s_z_i

        best_idx = np.flipud(np.argsort(scores_coh_test, axis=0))[
            self.nbest_discard : self.nbest_discard + nbest
        ]
        scores_t_norm = np.zeros_like(scores)
        for i in range(scores.shape[1]):
            best_idx_i = best_idx[:, i]

            best_scores_i = scores_coh_test[best_idx_i, :]
            mu_t_i = np.mean(best_scores_i, axis=0, keepdims=False)

            if mask_enr_coh is None:
                s_t_i = np.std(best_scores_i, axis=0, keepdims=False)
            else:
                norm = np.mean(mask_coh_test[best_idx_i, :], axis=0, keepdims=False)
                mu_t_i /= norm
                s_t_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=0, keepdims=False) / norm
                    - mu_t_i ** 2
                )

            s_t_i = np.clip(s_t_i, a_min=1e-5, a_max=None)
            if not self.norm_var:
                s_t_i = 1.0

            scores_t_norm[i, :] = (scores[i, :] - mu_t_i) / s_t_i
            if return_stats:
                mu_t[i, :] = mu_t_i
                if self.norm_var:
                    s_t[i, :] = s_t_i

        scores_norm = (scores_z_norm + scores_t_norm) / np.sqrt(2)
        if return_stats:
            return scores_norm, mu_z, s_z, mu_t, s_t
        else:
            return scores_norm
