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


class SNorm(ScoreNorm):
    """Class for S-Norm, symmetric score normalization.

    Example:
      ```python
      import numpy as np
      from hyperion.np.score_norm import SNorm

      n_enr, n_test, n_coh = 3, 5, 20
      scores = np.random.randn(n_enr, n_test)
      scores_coh_test = np.random.randn(n_coh, n_test)
      scores_enr_coh = np.random.randn(n_enr, n_coh)

      s_norm = SNorm(norm_var=True, std_floor=1e-5)
      scores_s = s_norm.predict(scores, scores_coh_test, scores_enr_coh)
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
        mask_coh_test: Optional[np.ndarray] = None,
        mask_enr_coh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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

        scores_z_norm = self.z_norm.predict(scores, scores_enr_coh, mask_enr_coh)
        scores_t_norm = self.t_norm.predict(scores, scores_coh_test, mask_coh_test)

        return (scores_z_norm + scores_t_norm) / np.sqrt(2)
