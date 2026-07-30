"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Optional

import h5py
import numpy as np

from .score_norm import ScoreNorm
from .t_norm import TNorm
from .z_norm import ZNorm


class ZTNorm(ScoreNorm):
    """Class ZT-Norm score-normalization.

    Example:
      ```python
      import numpy as np
      from hyperion.np.score_norm import ZTNorm

      n_enr, n_test, n_coh = 3, 5, 20
      scores = np.random.randn(n_enr, n_test)
      scores_coh_test = np.random.randn(n_coh, n_test)
      scores_enr_coh = np.random.randn(n_enr, n_coh)
      scores_coh_coh = np.random.randn(n_coh, n_coh)

      zt_norm = ZTNorm(norm_var=True, std_floor=1e-5)
      scores_zt = zt_norm.predict(
          scores, scores_coh_test, scores_enr_coh, scores_coh_coh
      )
      ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Builds the internal T-Norm and Z-Norm normalizers.

        Args:
          **kwargs: Parameters forwarded to `ScoreNorm`, `TNorm`, and `ZNorm`.
        """
        super().__init__(**kwargs)
        self.t_norm = TNorm(**kwargs)
        self.z_norm = ZNorm(**kwargs)

    def predict(
        self,
        scores: np.ndarray,
        scores_coh_test: np.ndarray,
        scores_enr_coh: np.ndarray,
        scores_coh_coh: np.ndarray,
        mask_coh_test: Optional[np.ndarray] = None,
        mask_enr_coh: Optional[np.ndarray] = None,
        mask_coh_coh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalizes the scores.

        Args:
          scores: score matrix enroll vs. test.
          scores_coh_test: score matrix cohort vs. test.
          scores_enr_coh: score matrix enroll vs cohort.
          scores_coh_coh: score matrix cohort vs cohort.
          mask_coh_test: binary matrix to mask out target trials
            from cohort vs test matrix.
          mask_enr_coh: binary matrix to mask out target trials
            from enroll vs. cohort matrix.
          mask_coh_coh: binary matrix to mask out target trials
            from cohort vs. cohort matrix.
        """

        scores_z_norm = self.z_norm.predict(scores, scores_enr_coh, mask_enr_coh)
        scores_coh_test_z_norm = self.z_norm.predict(
            scores_coh_test, scores_coh_coh, mask_coh_coh
        )
        scores_zt_norm = self.t_norm.predict(
            scores_z_norm, scores_coh_test_z_norm, mask_coh_test
        )

        return scores_zt_norm
