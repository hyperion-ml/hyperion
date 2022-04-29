"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


def scale_seq_lengths(lengths, max_out_length, max_in_length=None):
    if lengths is None:
        return None

    if max_in_length is None:
        max_in_length = lengths.max()

    return torch.div(lengths * max_out_length, max_in_length, rounding_mode="floor")


def seq_lengths_to_mask(lengths, max_length=None, dtype=None, time_dim=1):
    """Creates a binary masks indicating the valid values in a sequence.

    Args:
      lengths: sequence lengths with shape=(batch,). If None, it returns None
      max_length: maximum length of the sequence.
      dtype: dtype for the mask.
      time_dim: dimension corresponding to time in the mask. This will
                return a view of the mask which will adapt to the shape
                of the tensor where we want to apply the mask.
                This has to be a positive integer.

    Returns:
      Binary mask with shape=(batch,...,max_length) or None
    """
    if lengths is None:
        return None

    assert lengths.dim() == 1

    if max_length is None:
        max_length = lengths.max()
    idx = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    # compute mask shape=(batch, max_length)
    mask = idx.unsqueeze(0) < lengths.unsqueeze(1)

    # view to match the tensor where we want to apply the mask
    if time_dim > 1:
        shape = [1] * (time_dim + 1)
        shape[0] = lengths.size(0)
        shape[time_dim] = -1
        mask = mask.view(*shape)

    # change dtype if needed
    if dtype is not None:
        mask = mask.to(dtype)

    return mask
