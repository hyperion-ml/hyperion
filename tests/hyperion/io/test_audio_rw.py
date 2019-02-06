"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.io import AudioWriter as AW, SequentialAudioReader as SAR, RandomAccessAudioReader as RAR

audio_path = './tests/data_out/io/audio'
wav_scp_file = audio_path + '/wav.scp'
flac_scp_file = audio_path + '/flac.scp'
pipe_scp_file = audio_path + '/pipe.scp'
fs = 16000

def gen_signals(num_signals=3):
    rng = np.random.RandomState(seed=1)
    s = []
    keys = []
    for i in xrange(num_signals):
        s_i = rng.randn(16000)
        s_i = ((2**15-1)/np.max(np.abs(s_i))*s_i).astype('int32').astype(float_cpu())
        s.append(s_i)
        keys.append('s%d' % i)

    return keys, s


keys, s = gen_signals()


def test_write_audio_files_wav():

    with AW(audio_path, wav_scp_file, 'wav') as w:
        w.write(keys, s, fs)


def test_write_audio_files_flac():

    with AW(audio_path, flac_scp_file, 'flac') as w:
        w.write(keys, s, fs)


def test_read_sar_wav():

    with SAR(wav_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys,keys1):
        assert k_i == k1_i
        
    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)
        
        
def test_read_sar_flac():

    with SAR(flac_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys,keys1):
        assert k_i == k1_i
        
    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)

        
        
def test_read_sar_pipe():

    with open(pipe_scp_file,'w') as f:
        for i, k in enumerate(keys):
            f.write('%s sox %s/%s.flac -t wav - |\n' %(k, audio_path, k))

    with SAR(pipe_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys,keys1):
        assert k_i == k1_i
        
    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)


def test_read_sar_iter():

    with SAR(wav_scp_file) as r:
        for i, (k_i, s_i, fs_i) in enumerate(r):
            assert k_i == keys[i]
            assert_allclose(s_i, s[i], atol=1)
            assert fs_i == fs


def test_read_rar():

    with RAR(wav_scp_file) as r:
        s1, fs1 = r.read(keys)

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)

