"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import time
import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import float_cpu
from hyperion.feats.frame_selector import *


def generate_features():

    rng = np.random.RandomState(seed = 1024)
    x = rng.randn(10,2).astype(float_cpu(), copy=False)
    vad = np.zeros((10,), dtype='bool')
    vad[4:8]=1
    return x,vad

x,vad = generate_features()



def test_select():

    fs = FrameSelector(tol_num_frames=3)

    y = fs.select(x,vad)
    assert_allclose(x[4:8], y)


def test_select_missmatch_num_frames():

    fs = FrameSelector(tol_num_frames=3)

    y = fs.select(x[:8],vad)
    assert_allclose(x[4:8], y)

    y = fs.select(x,vad[:8])
    assert_allclose(x[4:8], y)

