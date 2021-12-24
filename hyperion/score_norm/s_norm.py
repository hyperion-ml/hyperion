"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import h5py

from .score_norm import ScoreNorm
from .t_norm import TNorm
from .z_norm import ZNorm


class SNorm(ScoreNorm):
    """Class for S-Norm, symmetric score normalization."""

    def __init__(self, **kwargs):
        super(SNorm, self).__init__(*kwargs)
        self.t_norm = TNorm(**kwargs)
        self.z_norm = ZNorm(**kwargs)

    def predict(
        self,
        scores,
        scores_coh_test,
        scores_enr_coh,
        mask_coh_test=None,
        mask_enr_coh=None,
    ):

        scores_z_norm = self.z_norm.predict(scores, scores_enr_coh, mask_enr_coh)
        scores_t_norm = self.t_norm.predict(scores, scores_coh_test, mask_coh_test)

        return (scores_z_norm + scores_t_norm) / np.sqrt(2)
