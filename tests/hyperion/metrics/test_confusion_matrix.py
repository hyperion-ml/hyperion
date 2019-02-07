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

from hyperion.metrics.confusion_matrix import *


def test_confusion_matrix():

    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0])

    C = compute_confusion_matrix(y_true, y_pred)
    C_true = np.array([[0.6, 0.4], [0.4, 0.6]])

    assert_allclose(C, C_true)


def test_xlabel_confusion_matrix():

    y_true = np.array([0, 0, 0, 0, 0, 2, 2, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0])

    C = compute_xlabel_confusion_matrix(y_true, y_pred)
    C_true = np.array([[3/5, 2/5], [2/3, 1/3], [0, 1]])

    assert_allclose(C, C_true)



    
