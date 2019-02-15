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
from hyperion.feats.feature_normalization import *


def generate_features():

    rng = np.random.RandomState(seed = 1024)
    x = rng.randn(60*100,2).astype(float_cpu(), copy=False)
    x *= rng.rand(60*100,1)
    
    return x

x = generate_features()


def test_mvn_global():

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=False)
    x_norm = mvn.normalize(x)
    x_ref = x - np.mean(x, axis=0)
    assert_allclose(x_norm, x_ref)

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=True)
    x_norm = mvn.normalize(x)
    x_ref /= np.std(x, axis=0)
    assert_allclose(x_norm, x_ref)


def test_stmvn():

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=False,
                           left_context=150, right_context=50)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)
    # idx=np.argmax(np.abs(x_norm-x_ref))
    # print(x_norm.ravel()[idx], x_ref.ravel()[idx], np.abs(x_norm-x_ref).ravel()[idx])
    # print(x_norm[:10])
    # print(x_ref[:10])
    # print(x_norm[1000:1010])
    # print(x_ref[1000:1010])
    # print(x_norm[-10:])
    # print(x_ref[-10:])
    assert_allclose(x_norm, x_ref, atol=1e-4)

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=True,
                           left_context=150, right_context=50)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)
    assert_allclose(x_norm, x_ref, atol=1e-4)


    
def test_mvn_cum_forward():

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=False,
                           left_context=None, right_context=0)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)

    assert_allclose(x_norm, x_ref, atol=1e-4)

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=True,
                           left_context=None, right_context=0)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)
    assert_allclose(x_norm, x_ref, atol=1e-4)


    
def test_mvn_cum_backward():

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=False,
                           left_context=0, right_context=None)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)
    # idx=np.argmax(np.abs(x_norm-x_ref))
    # print(x_norm.ravel()[idx], x_ref.ravel()[idx], np.abs(x_norm-x_ref).ravel()[idx])
    # print(x_norm[:10])
    # print(x_ref[:10])
    # print(x_norm[1000:1010])
    # print(x_ref[1000:1010])
    # print(x_norm[-10:])
    # print(x_ref[-10:])
    assert_allclose(x_norm, x_ref, atol=1e-4)

    mvn = MeanVarianceNorm(norm_mean=True, norm_var=True,
                           left_context=0, right_context=None)
    x_norm = mvn.normalize(x)
    x_ref = mvn.normalize_slow(x)
    assert_allclose(x_norm, x_ref, atol=1e-4)


    
