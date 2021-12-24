"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from .score_norm import ScoreNorm
from .t_norm import TNorm
from .z_norm import ZNorm


class TZNorm(ScoreNorm):
    """Class for TZ-Norm score normalization."""

    def __init__(self, **kwargs):
        super(SNorm, self).__init__(*kwargs)
        self.t_norm = TNorm(**kwargs)
        self.z_norm = ZNorm(**kwargs)

    def predict(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        scores_coh_coh,
        mask_coh_test=None,
        mask_enr_coh=None,
        mask_coh_coh=None,
    ):

        scores_t_norm = self.t_norm.predict(scores, scores_coh_test, mask_coh_test)
        scores_enr_coh_t_norm = self.t_norm.predict(
            scores_enr_coh, scores_coh_coh, mask_coh_coh
        )
        scores_tz_norm = self.z_norm.predict(
            scores_t_norm, scores_enr_coh_t_norm, mask_enr_coh
        )

        return scores_tz_norm
