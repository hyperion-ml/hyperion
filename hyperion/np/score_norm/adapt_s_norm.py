"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
import h5py
import numpy as np

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

    def __call__(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test=None,
        mask_enr_coh=None,
        return_stats=False,
    ):
        return self.predict(
            scores,
            scores_coh_test,
            scores_enr_coh,
            mask_coh_test,
            mask_enr_coh,
            return_stats,
        )

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
            scores_coh_test[~mask_coh_test] = 0
        if mask_enr_coh is not None:
            scores_enr_coh[~mask_enr_coh] = 0

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
        else:
            raise Exception(f"invalid cohort selection method {self.nbest_sel_method}")

    def _norm_highest_other_side0(
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

        # this is very memory intensive, so we pass to f32
        scores_coh_test = scores_coh_test.astype("float32", copy=False)
        scores_enr_coh = scores_enr_coh.astype("float32", copy=False)

        best_idx = np.argsort(-scores_coh_test, axis=0)[
            self.nbest_discard : self.nbest_discard + nbest
        ].T  # (n_test, n_best)

        mem = nbest * scores_enr_coh.shape[0] * scores.shape[1] * 4 / 2 ** 30
        # limit mem to 10 GB
        num_groups = math.ceil(mem / 10)
        num_el_group = int(math.ceil(scores.shape[1] / num_groups))
        scores_enr_coh = np.expand_dims(scores_enr_coh, 0)
        if mask_enr_coh is not None:
            mask_enr_coh = np.expand_dims(scores_enr_coh, 0)

        mu_z = []
        s_z = []
        for start in range(0, scores.shape[1], num_el_group):
            stop = min(start + num_el_group, scores.shape[1])
            best_idx_i = np.expand_dims(best_idx[start:stop], 1)
            best_scores_i = np.take_along_axis(scores_enr_coh, best_idx_i, axis=-1)
            mu_z_i = best_scores_i.mean(axis=-1)

            if mask_enr_coh is None:
                s_z_i = np.std(best_scores_i, axis=-1)
            else:
                mask_i = np.take_along_axis(mask_enr_coh, best_idx_i, axis=-1)
                norm = mask_i.mean(axis=-1)
                mu_z_i /= norm
                s_z_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=-1) / norm - mu_z_i ** 2
                )

            del best_scores_i
            mu_z.append(mu_z_i.T)
            s_z.append(s_z_i.T)

        mu_z = np.concatenate(mu_z, axis=-1)
        s_z = np.concatenate(s_z, axis=-1)

        s_z = np.clip(s_z, a_min=1e-5, a_max=None)
        if not self.norm_var:
            s_z = 1.0

        scores_z_norm = (scores - mu_z) / s_z

        scores_enr_coh = scores_enr_coh[0]  # unsqueeze
        best_idx = np.argsort(-scores_enr_coh, axis=1)[
            :, self.nbest_discard : self.nbest_discard + nbest
        ].T

        mem = nbest * scores.shape[0] * scores_coh_test.shape[1] * 4 / 2 ** 30
        # limit mem to 10 GB
        num_groups = math.ceil(mem / 10)
        num_el_group = int(math.ceil(scores.shape[0] / num_groups))
        scores_coh_test = np.expand_dims(scores_coh_test, -1)
        if mask_coh_test is not None:
            mask_coh_test = np.expand_dims(mask_coh_test, -1)

        mu_t = []
        s_t = []
        for start in range(0, scores.shape[0], num_el_group):
            stop = min(start + num_el_group, scores.shape[0])
            best_idx_i = np.expand_dims(best_idx[:, start:stop], 1)
            # print(scores_coh_test.shape, best_idx_i.shape)
            best_scores_i = np.take_along_axis(scores_coh_test, best_idx_i, axis=0)
            # print(best_scores_i.shape)
            mu_t_i = best_scores_i.mean(axis=0)
            if mask_enr_coh is None:
                s_t_i = np.std(best_scores_i, axis=0)
            else:
                mask_i = np.take_along_axis(mask_coh_test, best_idx_i, axis=0)
                norm = mask_i.mean(axis=0)
                mu_t_i /= norm
                s_t_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=0) / norm - mu_t_i ** 2
                )

            # print(best_scores_i.shape, mu_t_i.shape)
            del best_scores_i
            mu_t.append(mu_t_i.T)
            s_t.append(s_t_i.T)

        mu_t = np.concatenate(mu_t, axis=0)
        s_t = np.concatenate(s_t, axis=0)

        s_t = np.clip(s_t, a_min=1e-5, a_max=None)
        if not self.norm_var:
            s_t = 1.0

        scores_t_norm = (scores - mu_t) / s_t

        scores_norm = (scores_z_norm + scores_t_norm) / np.sqrt(2)
        if return_stats:
            return scores_norm, mu_z, s_z, mu_t, s_t
        else:
            return scores_norm

    def _norm_highest_same_side0(
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

        # this is very memory intensive, so we pass to f32
        scores_coh_test = scores_coh_test.astype("float32", copy=False)
        scores_enr_coh = scores_enr_coh.astype("float32", copy=False)

        best_idx = np.argsort(-scores_enr_coh, axis=1)[
            :, self.nbest_discard : self.nbest_discard + nbest
        ]

        mem = nbest * scores_enr_coh.shape[0] * scores.shape[0] * 4 / 2 ** 30
        # limit mem to 10 GB
        num_groups = math.ceil(mem / 10)
        num_el_group = int(math.ceil(scores.shape[0] / num_groups))
        scores_enr_coh = np.expand_dims(scores_enr_coh, 0)
        if mask_enr_coh is not None:
            mask_enr_coh = np.expand_dims(scores_enr_coh, 0)

        mu_z = []
        s_z = []
        for start in range(0, scores.shape[0], num_el_group):
            stop = min(start + num_el_group, scores.shape[0])
            best_idx_i = np.expand_dims(best_idx[start:stop], 1)
            best_scores_i = np.take_along_axis(scores_enr_coh, best_idx_i, axis=-1)
            mu_z_i = best_scores_i.mean(axis=-1)

            if mask_enr_coh is None:
                s_z_i = np.std(best_scores_i, axis=-1)
            else:
                mask_i = np.take_along_axis(mask_enr_coh, best_idx_i, axis=-1)
                norm = mask_i.mean(axis=-1)
                mu_z_i /= norm
                s_z_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=-1) / norm - mu_z_i ** 2
                )

            del best_scores_i
            mu_z.append(mu_z_i.T)
            s_z.append(s_z_i.T)

        mu_z = np.concatenate(mu_z, axis=-1)
        s_z = np.concatenate(s_z, axis=-1)

        s_z = np.clip(s_z, a_min=1e-5, a_max=None)
        if not self.norm_var:
            s_z = 1.0

        scores_z_norm = (scores - mu_z) / s_z

        best_idx = np.argsort(-scores_coh_test, axis=0)[
            self.nbest_discard : self.nbest_discard + nbest
        ]  # (n_best, n_test)

        mem = nbest * scores.shape[1] * scores_coh_test.shape[1] * 4 / 2 ** 30
        # limit mem to 10 GB
        num_groups = math.ceil(mem / 10)
        num_el_group = int(math.ceil(scores.shape[1] / num_groups))
        scores_coh_test = np.expand_dims(scores_coh_test, -1)
        if mask_coh_test is not None:
            mask_coh_test = np.expand_dims(mask_coh_test, -1)

        mu_t = []
        s_t = []
        for start in range(0, scores.shape[1], num_el_group):
            stop = min(start + num_el_group, scores.shape[1])
            best_idx_i = np.expand_dims(best_idx[:, start:stop], 1)
            # print(scores_coh_test.shape, best_idx_i.shape)
            best_scores_i = np.take_along_axis(scores_coh_test, best_idx_i, axis=0)
            # print(best_scores_i.shape)
            mu_t_i = best_scores_i.mean(axis=0)
            if mask_enr_coh is None:
                s_t_i = np.std(best_scores_i, axis=0)
            else:
                mask_i = np.take_along_axis(mask_coh_test, best_idx_i, axis=0)
                norm = mask_i.mean(axis=0)
                mu_t_i /= norm
                s_t_i = np.sqrt(
                    np.mean(best_scores_i ** 2, axis=0) / norm - mu_t_i ** 2
                )

            # print(best_scores_i.shape, mu_t_i.shape)
            del best_scores_i
            mu_t.append(mu_t_i.T)
            s_t.append(s_t_i.T)

        mu_t = np.concatenate(mu_t, axis=0)
        s_t = np.concatenate(s_t, axis=0)

        s_t = np.clip(s_t, a_min=1e-5, a_max=None)
        if not self.norm_var:
            s_t = 1.0

        scores_t_norm = (scores - mu_t) / s_t

        scores_norm = (scores_z_norm + scores_t_norm) / np.sqrt(2)
        if return_stats:
            return scores_norm, mu_z, s_z, mu_t, s_t
        else:
            return scores_norm
