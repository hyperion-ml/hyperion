"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hyperion.hyp_defs import float_cpu
from hyperion.np.feats.mfcc import MFCC

fs = 16000
window_type = "povey"


def generate_signal():

    rng = np.random.default_rng(seed=1024)
    s = (2**10) * rng.standard_normal((fs * 10,)).astype(float_cpu(), copy=False)
    return s


s = generate_signal()


def test_mfcc():

    mfcc = MFCC(window_type=window_type)
    P = mfcc.compute(s)


def test_mfcc_return_all():

    mfcc = MFCC(window_type=window_type)
    P, X, F, B = mfcc.compute(s, return_fft=True, return_spec=True, return_logfb=True)


def test_mfcc_etsi():

    mfcc = MFCC(window_type=window_type, fb_type="mel_etsi")
    P = mfcc.compute(s)


def test_mfcc_linear():

    mfcc = MFCC(window_type=window_type, fb_type="linear")
    P = mfcc.compute(s)


def test_mfcc_from_fft():

    mfcc = MFCC(window_type=window_type)
    P = mfcc.compute(s)

    mfcc_1 = MFCC(window_type=window_type, output_step="fft")
    mfcc_2 = MFCC(window_type=window_type, input_step="fft")

    X = mfcc_1.compute(s)
    P2 = mfcc_2.compute(X)

    assert_allclose(P, P2, rtol=1e-5, atol=1e-5)


def test_mfcc_from_fft_mag():

    mfcc = MFCC(window_type=window_type)
    P = mfcc.compute(s)

    mfcc_1 = MFCC(window_type=window_type, output_step="spec")
    mfcc_2 = MFCC(window_type=window_type, input_step="spec")

    F = mfcc_1.compute(s)
    P2 = mfcc_2.compute(F)

    assert_allclose(P, P2, rtol=1e-5)


def test_mfcc_from_logfb():

    mfcc = MFCC(window_type=window_type)
    P = mfcc.compute(s)

    mfcc_1 = MFCC(window_type=window_type, output_step="logfb")
    mfcc_2 = MFCC(window_type=window_type, input_step="logfb")

    B = mfcc_1.compute(s)
    P2 = mfcc_2.compute(B)

    assert_allclose(P, P2, rtol=1e-5)
