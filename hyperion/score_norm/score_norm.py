"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..hyp_model import HypModel


class ScoreNorm(HypModel):
    """
    Base class for score normalization
    """

    def __init__(self, std_floor=1e-5, **kwargs):
        super(ScoreNorm, self).__init__(*kwargs)
        self.std_floor = std_floor
