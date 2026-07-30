"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional

import numpy as np

from .score_norm import ScoreNorm


class ZNorm(ScoreNorm):
    """Class for Z-Norm score normalization.

    Example:
      ```python
      import numpy as np
      from hyperion.np.score_norm import ZNorm

      n_enr, n_test, n_coh = 3, 5, 20
      scores = np.random.randn(n_enr, n_test)
      scores_enr_coh = np.random.randn(n_enr, n_coh)

      z_norm = ZNorm(norm_var=True, std_floor=1e-5)
      scores_z = z_norm.predict(scores, scores_enr_coh)
      ```
    """

    def predict(
        self,
        scores: np.ndarray,
        scores_enr_coh: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalizes the scores.

        Args:
          scores: score matrix enroll vs. test.
          scores_enr_coh: score matrix enroll vs cohort.
          mask: binary matrix to mask out target trials
            from enroll vs. cohort matrix.

        """
        if mask is None:
            mu_z = np.mean(scores_enr_coh, axis=1, keepdims=True)
            if self.norm_var:
                s_z = np.std(scores_enr_coh, axis=1, keepdims=True)
        else:
            scores_enr_coh[mask == False] = 0
            n_z = np.maximum(np.mean(mask, axis=1, keepdims=True), 1e-10)
            mu_z = np.mean(scores_enr_coh, axis=1, keepdims=True) / n_z
            if self.norm_var:
                s_z = np.sqrt(
                    np.mean(scores_enr_coh**2, axis=1, keepdims=True) / n_z - mu_z**2
                )

        if self.norm_var:
            s_z[s_z < self.std_floor] = self.std_floor
        else:
            s_z = 1.0

        scores_norm = (scores - mu_z) / s_z
        return scores_norm
