"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import numpy as np
import h5py

from .score_norm import ScoreNorm


class AdaptSNorm(ScoreNorm):
    """Class for adaptive S-Norm"""

    def __init__(self, nbest=100, nbest_discard=0, **kwargs):
        super(AdaptSNorm, self).__init__(*kwargs)
        self.nbest = nbest
        self.nbest_discard = nbest_discard

    def predict(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test=None,
        mask_enr_coh=None,
    ):

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

        best_idx = np.flipud(np.argsort(scores_coh_test, axis=0))[
            self.nbest_discard : self.nbest_discard + nbest
        ]
        scores_z_norm = np.zeros_like(scores)
        for i in range(scores.shape[1]):
            best_idx_i = best_idx[:, i]

            mu_z = np.mean(scores_enr_coh[:, best_idx_i], axis=1, keepdims=True)

            if mask_enr_coh is None:
                s_z = np.std(scores_enr_coh[:, best_idx_i], axis=1, keepdims=True)
            else:
                norm = np.mean(mask_enr_coh[:, best_idx_i], axis=1, keepdims=True)
                mu_z /= norm
                s_z = np.sqrt(
                    np.mean(scores_enr_coh[:, best_idx_i] ** 2, axis=1, keepdims=True)
                    / norm
                    - mu_z ** 2
                )

            s_z = np.clip(s_z, a_min=1e-5, a_max=None)
            scores_z_norm[:, i] = (scores[:, i] - mu_z.T) / s_z.T

        best_idx = np.fliplr(np.argsort(scores_enr_coh, axis=1))[
            :, self.nbest_discard : self.nbest_discard + nbest
        ]
        scores_t_norm = np.zeros_like(scores)
        for i in range(scores.shape[0]):
            best_idx_i = best_idx[i]

            mu_z = np.mean(scores_coh_test[best_idx_i, :], axis=0, keepdims=True)

            if mask_coh_test is None:
                s_z = np.std(scores_coh_test[best_idx_i, :], axis=0, keepdims=True)
            else:
                norm = np.mean(mask_coh_test[best_idx_i, :], axis=0, keepdims=True)
                mu_z /= norm
                s_z = np.sqrt(
                    np.mean(scores_coh_test[best_idx_i, :] ** 2, axis=0, keepdims=True)
                    / norm
                    - mu_z ** 2
                )

            s_z = np.clip(s_z, a_min=1e-5, a_max=None)
            scores_t_norm[i, :] = (scores[i, :] - mu_z) / s_z

        return (scores_z_norm + scores_t_norm) / np.sqrt(2)
