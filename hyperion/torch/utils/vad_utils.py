"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .collation import collate_seqs_nd


def remove_silence(
    x: torch.Tensor,
    vad: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    time_dim: int = 1,
    tol: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove silence samples/frames.

    Args:
        x: Input tensor of shape ``(batch, ..., time, ...)``.
        vad: Binary VAD mask of shape ``(batch, time)``.
        x_lengths: Optional valid lengths for each sequence in ``x``.
        time_dim: Time dimension index in ``x``.
        tol: Allowed difference between ``x`` and ``vad`` time lengths.

    Returns:
        Tuple ``(y, y_lengths)`` where ``y`` is ``x`` with silent frames removed
        and padded across the batch, and ``y_lengths`` contains the resulting
        per-sample lengths.
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
        vad = nn.functional.pad(vad, (0, length_err), mode="constant", value=0)
    elif length_err < 0:
        vad = vad[:, :x_max_length]

    # if x_lengths is passed, we make sure that vad is 0 for time steps larger
    # than x_length
    if x_lengths is not None:
        for i in range(x.size(0)):
            vad[i, x_lengths[i] :] = 0

    if time_dim < 0:
        time_dim = x.dim() + time_dim

    trans = False
    if time_dim != 1:
        x = x.transpose(1, time_dim)
        trans = True

    y = []
    for i in range(x.size(0)):
        y.append(x[i, vad[i]])

    y, y_lengths = collate_seqs_nd(y, pad_dim=0)
    if trans:
        y = y.transpose(1, time_dim).contiguous()

    return y, y_lengths
