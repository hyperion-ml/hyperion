"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..hyper_np_model import HyperNPModel


class ScoreNorm(HyperNPModel):
    """Base class for score normalization

    Attributes:
      std_floor: floor for standard deviations.
    """

    def __init__(self, norm_var=True, std_floor=1e-5, **kwargs):
        super().__init__(*kwargs)
        self.norm_var = norm_var
        self.std_floor = std_floor

    def forward(self, **kwargs):
        """Overloads predict function."""
        return self.predict(**kwargs)

    def __call__(self, *args, **kwargs):
        """Overloads predict function."""
        return self.predict(*args, **kwargs)
