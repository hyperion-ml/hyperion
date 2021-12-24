"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.utils.vad_utils import *


def test_merge_vad_timestamps():

    t_in = np.asarray(
        [[0.01, 3.4], [1.01, 2.3], [2.50, 3.7], [5.1, 6.3], [7, 8], [7.5, 9]]
    )

    t_target = np.asarray([[0.01, 3.7], [5.1, 6.3], [7, 9]])

    t_out = merge_vad_timestamps(t_in)

    assert_allclose(t_out, t_target)


def test_bin_vad_to_timestamps():

    vad = np.asarray([1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=np.bool)
    t_target = (
        np.asarray(
            [[0, 25], [3 * 10, 8 * 10 + 25], [12 * 10, 13 * 10 + 25]], dtype=np.float
        )
        - (25.0 - 10.0) / 2
    )
    t_target[0, 0] = 0

    t_out = bin_vad_to_timestamps(vad, 25, 10)
    assert_allclose(t_out, t_target)


def test_bin_vad_to_timestamps_snipedges():

    vad = np.asarray([0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=np.bool)
    t_target = np.asarray(
        [[3 * 10, 8 * 10 + 25], [12 * 10, 13 * 10 + 25]], dtype=np.float
    )

    t_out = bin_vad_to_timestamps(vad, 25, 10, snip_edges=True)
    assert_allclose(t_out, t_target)


def test_vad_timestamps_to_bin():

    vad_target = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.bool
    )
    t_in = (
        np.asarray(
            [[0, 25], [3 * 10, 8 * 10 + 25], [12 * 10, 13 * 10 + 25]], dtype=np.float
        )
        - (25.0 - 10.0) / 2
    )
    t_in[0, 0] = 0
    vad_out = vad_timestamps_to_bin(t_in, 25, 10)
    assert_allclose(vad_out, vad_target)


def test_vad_timestamps_to_bin_snipedges():

    vad_target = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], dtype=np.bool)
    t_in = np.asarray([[3 * 10, 7 * 10 + 25], [12 * 10, 12 * 10 + 25]], dtype=np.float)

    vad_out = vad_timestamps_to_bin(t_in, 25, 10, snip_edges=True)
    assert_allclose(vad_out, vad_target)

    vad_target = np.asarray([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], dtype=np.bool)
    t_in = np.asarray(
        [[0, 13], [3 * 10, 7 * 10 + 25], [12 * 10, 12 * 10 + 25]], dtype=np.float
    )

    vad_out = vad_timestamps_to_bin(t_in, 25, 10, snip_edges=True)
    assert_allclose(vad_out, vad_target)


def test_timestamps_wrt_vad_to_absolute_timestamps():

    vad = np.asarray([[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]])

    t_in = np.asarray([[0.5, 1.0], [1.25, 1.5], [1.75, 2.75]])

    t_target = np.asarray([[1.5, 2.0], [4.25, 4.50], [4.75, 5.0], [6.0, 6.75]])

    t_out = timestamps_wrt_vad_to_absolute_timestamps(t_in, vad)

    assert_allclose(t_out, t_target)


def test_timestamps_wrt_bin_vad_to_absolute_timestamps():

    vad = np.asarray([[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]])

    vad = vad_timestamps_to_bin(vad, 0.025, 0.010)
    t_in = np.asarray([[0.5, 1.0], [1.25, 1.5], [1.75, 2.75]])

    t_target = np.asarray([[1.5, 2.0], [4.25, 4.50], [4.75, 5.0], [6.0, 6.75]])

    t_out = timestamps_wrt_bin_vad_to_absolute_timestamps(t_in, vad, 0.025, 0.010)

    assert_allclose(t_out, t_target, atol=0.05)


def test_intersect_segment_timestamps_with_vad():

    t_in = np.asarray(
        [
            [0, 0.75],
            [0, 1.0],
            [0, 1.25],
            [0, 1.5],
            [0.25, 1.75],
            [0.5, 2.0],
            [0.75, 2.25],
            [1.0, 2.5],
            [1.25, 2.75],
            [1.5, 3.0],
            [1.75, 3.25],
            [2.0, 3.5],
            [2.25, 3.75],
            [2.5, 4.0],
            [2.75, 4.25],
            [3.0, 4.5],
            [3.25, 4.75],
            [3.5, 5.0],
            [3.75, 5.25],
            [4.0, 5.5],
            [4.25, 5.75],
            [4.5, 6.0],
            [4.75, 6.25],
            [5.0, 6.5],
        ]
    )

    vad = np.asarray([[1.3, 2.0], [2.1, 2.7], [5.0, 6.0]])

    speech_target = np.ones((t_in.shape[0],), dtype=np.bool)
    speech_target[:3] = False
    speech_target[14:18] = False

    t_target = np.asarray(
        [
            [1.3, 1.5],
            [1.3, 1.75],
            [1.3, 2.0],
            [1.3, 2],
            [2.1, 2.25],
            [1.3, 2.0],
            [2.1, 2.5],
            [1.3, 2.0],
            [2.1, 2.7],
            [1.5, 2.0],
            [2.1, 2.7],
            [1.75, 2.0],
            [2.1, 2.7],
            [2.1, 2.7],
            [2.25, 2.7],
            [2.5, 2.7],
            [5.0, 5.25],
            [5.0, 5.5],
            [5.0, 5.75],
            [5.0, 6.0],
            [5.0, 6.0],
            [5.0, 6.0],
        ]
    )

    out2speech_target = np.asarray(
        [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        dtype=np.int,
    )

    speech_idx, t_out, out2speech_idx = intersect_segment_timestamps_with_vad(t_in, vad)

    assert_allclose(speech_idx, speech_target)
    assert_allclose(t_out, t_target)
    assert_allclose(out2speech_idx, out2speech_idx)


if __name__ == "__main__":
    pytest.main([__file__])
