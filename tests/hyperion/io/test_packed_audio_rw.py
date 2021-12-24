"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.io import (
    PackedAudioWriter as AW,
    SequentialPackedAudioReader as SAR,
    RandomAccessPackedAudioReader as RAR,
)

audio_path = "./tests/data_out/io/packed_audio"
wav_scp_file = audio_path + "/wav.scp"
flac_scp_file = audio_path + "/flac.scp"
wav_file = audio_path + "/audio.wav"
flac_file = audio_path + "/audio.flac"
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
            .astype("int16")
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

    with AW(wav_file, wav_scp_file, "wav", fs=fs) as w:
        w.write(keys, s)


def test_write_audio_files_flac():

    with AW(flac_file, flac_scp_file, "flac", fs=fs) as w:
        w.write(keys, s)


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


def test_read_sar_wav_intervals():

    with SAR(wav_scp_file) as r:
        keys1, s1, fs1 = r.read(time_offset=0.2, time_durs=0.5)

    n_start = int(0.2 * fs)
    n = int(0.5 * fs)

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_sar_flac_intervals():

    with SAR(flac_scp_file) as r:
        keys1, s1, fs1 = r.read(time_offset=0.2, time_durs=0.5)

    n_start = int(0.2 * fs)
    n = int(0.5 * fs)

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_rar_wav_intervals():

    with RAR(wav_scp_file) as r:
        s1, fs1 = r.read(keys, time_offset=0.2, time_durs=0.5)

    n_start = int(0.2 * fs)
    n = int(0.5 * fs)

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_rar_flac_intervals():

    with RAR(flac_scp_file) as r:
        s1, fs1 = r.read(keys, time_offset=0.2, time_durs=0.5)

    n_start = int(0.2 * fs)
    n = int(0.5 * fs)

    for s_i, s1_i in zip(s, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_sar_wav_with_segments_and_intervals():

    with SAR(wav_scp_file, segments_file) as r:
        keys1, s1, fs1 = r.read(time_offset=0.02, time_durs=0.05)

    n_start = int(0.02 * fs)
    n = int(0.05 * fs)

    for k_i, k1_i in zip(keys_seg, keys1):
        assert k_i == k1_i

    for s_i, s1_i in zip(s_seg, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_rar_with_segments_and_intervals():

    with RAR(flac_scp_file, segments_file) as r:
        s1, fs1 = r.read(keys_seg, time_offset=0.02, time_durs=0.05)

    n_start = int(0.02 * fs)
    n = int(0.05 * fs)

    for s_i, s1_i in zip(s_seg, s1):
        assert_allclose(s_i[n_start : n_start + n], s1_i, atol=1)


def test_read_sar_num_samples():

    with SAR(wav_scp_file) as r:
        keys1, ns1 = r.read_num_samples()

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, ns1_i in zip(s, ns1):
        assert_allclose(len(s_i), ns1_i)


def test_read_rar_num_samples():

    with RAR(wav_scp_file) as r:
        ns1 = r.read_num_samples(keys)

    for s_i, ns1_i in zip(s, ns1):
        assert_allclose(len(s_i), ns1_i)


def test_read_sar_num_samples_segments():

    with SAR(wav_scp_file, segments_file) as r:
        keys1, ns1 = r.read_num_samples()

    for k_i, k1_i in zip(keys_seg, keys1):
        assert k_i == k1_i

    for s_i, ns1_i in zip(s_seg, ns1):
        assert_allclose(len(s_i), ns1_i)


def test_read_rar_num_samples_segments():

    with RAR(wav_scp_file, segments_file) as r:
        ns1 = r.read_num_samples(keys_seg)

    for s_i, ns1_i in zip(s_seg, ns1):
        assert_allclose(len(s_i), ns1_i)


def test_read_sar_time_duration():

    with SAR(wav_scp_file) as r:
        keys1, ts1 = r.read_time_duration()

    for k_i, k1_i in zip(keys, keys1):
        assert k_i == k1_i

    for s_i, ts1_i in zip(s, ts1):
        assert_allclose(1.0, ts1_i)


def test_read_rar_time_duration():

    with RAR(wav_scp_file) as r:
        ts1 = r.read_time_duration(keys)

    for s_i, ts1_i in zip(s, ts1):
        assert_allclose(1.0, ts1_i)


def test_read_sar_time_duration_segments():

    with SAR(wav_scp_file, segments_file) as r:
        keys1, ts1 = r.read_time_duration()

    for k_i, k1_i in zip(keys_seg, keys1):
        assert k_i == k1_i

    for s_i, ts1_i in zip(s_seg, ts1):
        assert_allclose(0.1, ts1_i)


def test_read_rar_time_duration_segments():

    with RAR(wav_scp_file, segments_file) as r:
        ts1 = r.read_time_duration(keys_seg)

    for s_i, ts1_i in zip(s_seg, ts1):
        assert_allclose(0.1, ts1_i)
