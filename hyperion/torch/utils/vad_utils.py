"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from .collation import collate_seq_nd


def remove_silence(x, vad, x_lengths=None, time_dim=1, tol=0):
    """Remove silence samples/frames.

    Args:
        x: input signal/spectrogram of shape=(batch,...,time,...).
        vad: binary voice activity detection mask of shape=(batch, time).
        x_lenghts: lengths of each sequence in x.
        time_dim: which dimension in x is time.
        tol: tolerance for the difference between time dimensions in x and vad.

    Returns:
        x without silence samples/frames.
    """

    # we make x and vad time dimensions of the same size.
    assert x.size(0) == vad.size(0), "batch-size is different for x and vad"
    x_max_length = x.size(time_dim)
    vad_max_length = vad.size(-1)
    length_err = x_max_length - vad_max_length
    assert abs(length_err) <= tol, (
        f"Difference between x_length({x_max_length}) and "
        f"vad_length({vad_max_length}) > tol ({tol})"
    )
    if length_err > 0:
        vad = nn.functional.pad(vad, (0, length_err), model="constant", value=0)
    elif length_err < 0:
        vad = vad[:, :x_max_length]

    # if x_lengths is passed, we make sure that vad is 0 for time steps larger
    # than x_length
    if x_lengths is not None:
        for i in range(x.size(0)):
            vad[i, x_lengths[i] :] = 0

    trans = False
    if time_dim != 1 or time_dim != 1 - x.dim():
        x = x.transpose(1, time_dim)
        trans = True

    y = []
    for i in range(x.size(0)):
        y.append(x[i, vad[i]])

    y, y_lengths = collate_seq_nd(y, pad_dim=0)
    if trans:
        y = y.transpose(1, time_dim).contigous()

    return y, y_lengths
