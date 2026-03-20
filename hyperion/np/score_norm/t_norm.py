"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import h5py
import numpy as np
from typing import Optional

from .score_norm import ScoreNorm


class TNorm(ScoreNorm):
    """Class for T-Norm score normalization.

    Example:
      ```python
      import numpy as np
      from hyperion.np.score_norm import TNorm

      n_enr, n_test, n_coh = 3, 5, 20
      scores = np.random.randn(n_enr, n_test)
      scores_coh_test = np.random.randn(n_coh, n_test)

      t_norm = TNorm(norm_var=True, std_floor=1e-5)
      scores_t = t_norm.predict(scores, scores_coh_test)
      ```
    """

    def predict(
        self,
        scores: np.ndarray,
        scores_coh_test: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalizes the scores.

        Args:
          scores: score matrix enroll vs. test.
          scores_coh_test: score matrix cohort vs. test.
          mask: binary matrix to mask out target trials
            from cohort vs test matrix.

        """
        if mask is None:
            mu_t = np.mean(scores_coh_test, axis=0, keepdims=True)
            if self.norm_var:
                s_t = np.std(scores_coh_test, axis=0, keepdims=True)
        else:
            scores_coh_test[mask == False] = 0
            n_t = np.maximum(np.mean(mask, axis=0, keepdims=True), 1e-10)
            mu_t = np.mean(scores_coh_test, axis=0, keepdims=True) / n_t
            if self.norm_var:
                s_t = np.sqrt(
                    np.mean(scores_coh_test ** 2, axis=0, keepdims=True) / n_t
                    - mu_t ** 2
                )

        if self.norm_var:
            s_t[s_t < self.std_floor] = self.std_floor
        else:
            s_t = 1.0

        scores_norm = (scores - mu_t) / s_t
        return scores_norm
