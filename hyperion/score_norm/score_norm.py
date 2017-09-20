"""
Base class for score normalization
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel


class ScoreNorm(HypModel):

    def __init__(self, std_floor=1e-5, **kwargs):
        super(ScoreNorm, self).__init__(*kwargs)
        self.std_floor = std_floor

    
