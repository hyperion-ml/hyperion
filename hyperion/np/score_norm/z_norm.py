"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .score_norm import ScoreNorm


class ZNorm(ScoreNorm):
    """
    Class for Z-Norm score normalization.
    """

    def predict(self, scores, scores_enr_coh, mask=None):
        """Normalizes the scores.

        Args:
          scores: score matrix enroll vs. test.
          scores_enr_coh: score matrix enroll vs cohort.
          mask: binary matrix to mask out target trials
            from enroll vs. cohort matrix.

        """
        if mask is None:
            mu_z = np.mean(scores_enr_coh, axis=1, keepdims=True)
            s_z = np.std(scores_enr_coh, axis=1, keepdims=True)
        else:
            scores_enr_coh[mask == False] = 0
            n_z = np.mean(mask, axis=1, keepdims=True)
            mu_z = np.mean(scores_enr_coh, axis=1, keepdims=True) / n_z
            s_z = np.sqrt(
                np.mean(scores_enr_coh ** 2, axis=1, keepdims=True) / n_z - mu_z ** 2
            )

        s_z[s_z < self.std_floor] = self.std_floor

        scores_norm = (scores - mu_z) / s_z
        return scores_norm
