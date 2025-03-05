"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hyperion.hyp_defs import float_cpu
from hyperion.np.feats.feature_windows import FeatureWindowFactory as FWF
from hyperion.np.feats.stft import *

margin = 10


def generate_signal():

    fs = 16000
    rng = np.random.default_rng(seed=1024)
    s = (2**10) * rng.standard_normal((fs * 10,)).astype(float_cpu(), copy=False)
    return s


s = generate_signal()


def test_stft_hanning_half():

    w = FWF.create("hanning", 512)

    X = stft(s, frame_length=512, frame_shift=256, fft_length=512, window=w)
    shat = np.real(istft(X, frame_length=512, frame_shift=256, window=w))

    s_ref = s[margin : shat.shape[0] - margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-3, atol=1e-1)


def test_strft_hanning_half():

    w = FWF.create("hanning", 512)

    X = strft(s, frame_length=512, frame_shift=256, fft_length=512, window=w)
    shat = istrft(X, frame_length=512, frame_shift=256, window=w)

    s_ref = s[margin : shat.shape[0] - margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-3, atol=1e-1)


def test_stft_povey_10hz():

    w = FWF.create("povey", 400)

    X = stft(s, frame_length=400, frame_shift=160, fft_length=512, window=w)
    shat = np.real(istft(X, frame_length=400, frame_shift=160, window=w))

    s_ref = s[margin : shat.shape[0] - margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-4, atol=1e-2)


def test_strft_povey_10hz():

    w = FWF.create("povey", 400)

    X = strft(s, frame_length=400, frame_shift=160, fft_length=512, window=w)
    shat = istrft(X, frame_length=400, frame_shift=160, window=w)

    s_ref = s[margin : shat.shape[0] - margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-4, atol=1e-2)
