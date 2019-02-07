"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import os
import numpy as np
from numpy.testing import assert_allclose

from hyperion.metrics.acc import compute_accuracy

def test_compute_accuracy():

    y_true = np.arange(10, dtype='int32')
    y_pred = np.arange(10, dtype='int32')
    y_pred[:3] = 5

    acc = compute_accuracy(y_true, y_pred)
    assert acc == 0.7

    
