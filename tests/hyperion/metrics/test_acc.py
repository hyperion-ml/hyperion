"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hyperion.np.metrics.acc import compute_accuracy


def test_compute_accuracy():

    y_true = np.arange(10, dtype="int32")
    y_pred = np.arange(10, dtype="int32")
    y_pred[:3] = 5

    acc = compute_accuracy(y_true, y_pred)
    assert acc == 0.7
