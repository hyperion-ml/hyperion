"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .feature_set import FeatureSet
from .info_table import InfoTable
from .misc import PathLike


class VADSet(FeatureSet):

    def __init__(self, df):
        super().__init__(df)
