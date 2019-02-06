"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
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
from hyperion.feats.energy_vad import EnergyVAD

fs=16000

def generate_signal():

    rng = np.random.RandomState(seed = 1024)
    s = (2**3)*rng.randn(fs*10).astype(float_cpu(), copy=False)
    vad = np.zeros((len(s),), dtype=bool)
    vad[2*fs:8*fs] = True
    s += (2**12)*vad.astype(dtype=float_cpu())*np.sign(s)
    vad = vad[::160]
    #s = rng.randn(fs*10).astype(float_cpu(), copy=False)
    return s, vad


s, vad = generate_signal()


def test_vad():
    e_vad = EnergyVAD()
    vad_est = e_vad.compute(s)
    print(np.max(s[2*fs:3*fs]), np.min(s[2*fs:3*fs]))

    assert np.mean(vad[:len(vad_est)]==vad_est) > 0.9
    
