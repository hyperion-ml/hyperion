"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.io import (
    AudioWriter as AW,
    SequentialAudioReader as SAR,
    RandomAccessAudioReader as RAR,
)

audio_path = "./tests/data_out/io/audio"
wav_scp_file = audio_path + "/wav.scp"
flac_scp_file = audio_path + "/flac.scp"
pipe_scp_file = audio_path + "/pipe.scp"
segments_file = audio_path + "/segments"
fs = 16000


def gen_signals(num_signals=3):
    rng = np.random.RandomState(seed=1)
    s = []
    keys = []
    for i in range(num_signals):
        s_i = rng.randn(fs)
        s_i = (
            ((2 ** 15 - 1) / np.max(np.abs(s_i)) * s_i)
            .astype("int32")
            .astype(float_cpu())
        )
        s.append(s_i)
        keys.append("s%d" % i)

    return keys, s


keys, s = gen_signals()


def gen_segments(num_signals=3, num_segs=2):

    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    keys_seg = []
    s_seg = []
    with open(segments_file, "w") as f:
        for i in range(num_signals):
            file_i = "s%d" % (i)
            for j in range(num_segs):
                seg_ij = "%s-%d" % (file_i, j)
                tbeg = j * 0.1
                tend = (j + 1) * 0.1
                f.write("%s %s %.2f %.2f\n" % (seg_ij, file_i, tbeg, tend))
                keys_seg.append(seg_ij)
                s_seg.append(s[i][int(tbeg * fs) : int(tend * fs)])

    return keys_seg, s_seg


keys_seg, s_seg = gen_segments()


def test_write_audio_files_wav():

    with AW(audio_path, wav_scp_file, "wav") as w:
        w.write(keys, s, fs)


def test_write_audio_files_flac():

    with AW(audio_path, flac_scp_file, "flac") as w:
        w.write(keys, s, fs)


def test_read_sar_wav():

    with SAR(wav_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)


def test_read_sar_flac():

    with SAR(flac_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i, s1_i, atol=1)


def test_read_sar_pipe():

    with open(pipe_scp_file, "w") as f:
        for i, k in enumerate(keys):
            f.write("%s sox %s/%s.flac -t wav - |\n" % (k, audio_path, k))

    with SAR(pipe_scp_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys, keys1):
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


def test_read_sar_wav_with_segments():

    with SAR(wav_scp_file, segments_file) as r:
        keys1, s1, fs1 = r.read()

    for k_i, k1_i in zip(keys_seg, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s_seg, s1):
        assert_allclose(s_i, s1_i, atol=1)


def test_read_rar_with_segments():

    with RAR(wav_scp_file, segments_file) as r:
        s1, fs1 = r.read(keys_seg)

    for s_i, s1_i in zip(s_seg, s1):
        assert_allclose(s_i, s1_i, atol=1)
