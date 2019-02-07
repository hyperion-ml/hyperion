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

from hyperion.metrics.cllr import *


def test_cllr():

    s = np.zeros((100,))
    c = compute_cllr(s, s)
    assert c == 1


def test_min_cllr():

    s = 10*np.ones((100,))
    c = compute_min_cllr(s, s)
    assert c <= 1 and c >0.99

    
