"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import float_cpu
from hyperion.feats.stft import *
from hyperion.feats.feature_windows import FeatureWindowFactory as FWF

margin=10

def generate_signal():

    fs=16000
    rng = np.random.RandomState(seed = 1024)
    s = (2**10)*rng.randn(fs*10).astype(float_cpu(), copy=False)
    return s

s = generate_signal()


def test_stft_hanning_half():

    w = FWF.create('hanning', 512)
    
    X = stft(s, frame_length=512, frame_shift=256, fft_length=512, window=w)
    shat = np.real(istft(X, frame_length=512, frame_shift=256, window=w))

    s_ref = s[margin:shat.shape[0]-margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-3, atol=1e-1)


def test_strft_hanning_half():

    w = FWF.create('hanning', 512)
    
    X = strft(s, frame_length=512, frame_shift=256, fft_length=512, window=w)
    shat = istrft(X, frame_length=512, frame_shift=256, window=w)

    s_ref = s[margin:shat.shape[0]-margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-3, atol=1e-1)


def test_stft_povey_10hz():

    w = FWF.create('povey', 400)
    
    X = stft(s, frame_length=400, frame_shift=160, fft_length=512, window=w)
    shat = np.real(istft(X, frame_length=400, frame_shift=160, window=w))

    s_ref = s[margin:shat.shape[0]-margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-4, atol=1e-2)

    

def test_strft_povey_10hz():

    w = FWF.create('povey', 400)
    
    X = strft(s, frame_length=400, frame_shift=160, fft_length=512, window=w)
    shat = istrft(X, frame_length=400, frame_shift=160, window=w)

    s_ref = s[margin:shat.shape[0]-margin]
    shat = shat[margin:-margin]
    assert_allclose(s_ref, shat, rtol=1e-4, atol=1e-2)



    
