"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import pytest
import os
import numpy as np
from numpy.testing import assert_allclose

from hyperion.metrics.eer import *


def test_eer():

    tar = np.linspace(-2, 10, 1000)
    non = np.linspace(-10, 2, 1000)

    eer = compute_eer(tar, non)
    assert eer > 0.166 and eer < 0.167


def test_prbep():

    tar = np.linspace(-2, 10, 1200)
    non = np.linspace(-10, 2, 1200)

    p = compute_prbep(tar, non)
    assert p == 200
