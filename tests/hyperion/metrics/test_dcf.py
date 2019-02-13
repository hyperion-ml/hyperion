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

from hyperion.metrics.dcf import *


def test_dcf():

    p_miss = np.asarray([0.01, 0.02, 0.03])
    p_fa = np.asarray([0.03, 0.02, 0.01])
    p_tar = 0.5
    dcf = compute_dcf(p_miss, p_fa, p_tar)

    dcf_ref = np.array([0.04, 0.04, 0.04])
    assert_allclose(dcf_ref, dcf)

    p_tar = [0.1, 0.5]
    dcf = compute_dcf(p_miss, p_fa, p_tar)
    
    dcf_ref = np.array([
        [0.01+9*0.03, 0.02+9*0.02, 0.03+9*0.01],
        [0.04, 0.04, 0.04]])
    assert_allclose(dcf_ref, dcf)

        
def test_min_dcf():

    tar = np.linspace(-2,10,1000)+3
    non = np.linspace(-10,2,1000)+3

    p=0.5
    dcf, _, _ = compute_min_dcf(tar, non, p)
    assert dcf > 0.332 and dcf < 0.334

    p = [0.1, 0.5, 0.9]
    dcf, _, _ = compute_min_dcf(tar, non, p)
    assert dcf[1] > 0.332 and dcf[1] < 0.334
    


def test_act_dcf():

    tar = np.linspace(-2,10,1000)
    non = np.linspace(-10,2,1000)

    p=0.5
    dcf, _, _ = compute_act_dcf(tar, non, p)
    assert dcf == 2*167/1000

    p = [0.1, 0.5, 0.9]
    dcf, _, _ = compute_act_dcf(tar, non, p)
    print(dcf)
    assert dcf[1] == 2*167/1000


def test_fast_eval():

    tar = np.linspace(-2,10,1000)
    non = np.linspace(-10,2,1000)

    p=0.5
    min_dcf, act_dcf, eer, _ = fast_eval_dcf_eer(tar, non, p)

    assert min_dcf > 0.332 and min_dcf < 0.334
    assert act_dcf == 2*167/1000
    assert eer > 0.166 and eer < 0.167

    p = [0.1, 0.5, 0.9]
    min_dcf, act_dcf, eer, _ = fast_eval_dcf_eer(tar, non, p)
    
    assert min_dcf[1] > 0.332 and min_dcf[1] < 0.334
    assert act_dcf[1] == 2*167/1000
    assert eer > 0.166 and eer < 0.167
