"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

Batch = Union[List[Any], Tuple[Any, ...]]
ListOfDicts = Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...]]


def list_of_dicts_to_list(list_of_dicts: ListOfDicts, key: str) -> List[Any]:
    """Returns values for ``key`` from each dict in ``list_of_dicts``."""
    output: List[Any] = []
    for item in list_of_dicts:
        output.append(item[key])

    return output


def collate_seqs_1d(
    x: Batch, pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of 1-D sequences to the same length.

    Args:
        x: List/tuple of 1-D tensors (or numpy arrays) with variable lengths.
        pad_value: Padding value used by ``torch.nn.utils.rnn.pad_sequence``.

    Returns:
        Tensor of shape ``(batch, max_length)`` containing padded sequences.
        Tensor of shape ``(batch,)`` with the original sequence lengths.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]

    assert x[0].dim() == 1
    x_lengths: List[int] = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths, device=x[0].device)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    return x, x_lengths


def collate_seqs_2d(
    x: Batch, pad_value: float = 0.0, pad_dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of 2-D tensors along a selected dimension.

    Args:
        x: List/tuple of 2-D tensors (or numpy arrays).
        pad_value: Padding value used by ``torch.nn.utils.rnn.pad_sequence``.
        pad_dim: Dimension to pad. Supports negative values.

    Returns:
        Tensor with shape ``(batch, max_length, feat_dim)`` when ``pad_dim=0``
        or ``(batch, feat_dim, max_length)`` when ``pad_dim=1``.
        Tensor of shape ``(batch,)`` with lengths along the padded dimension.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]
    assert x[0].dim() == 2
    if pad_dim < 0:
        pad_dim = 2 + pad_dim

    if pad_dim != 0:
        x = [x_i.transpose(pad_dim, 0) for x_i in x]

    x_lengths: List[int] = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths, device=x[0].device)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if pad_dim != 0:
        x = x.transpose(1, pad_dim + 1)

    return x, x_lengths


def collate_seqs_nd(
    x: Batch, pad_value: float = 0.0, pad_dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of N-D tensors along a selected dimension.

    Args:
        x: List/tuple of N-D tensors (or numpy arrays).
        pad_value: Padding value used by ``torch.nn.utils.rnn.pad_sequence``.
        pad_dim: Dimension to pad. Supports negative values.

    Returns:
        Padded tensor with one extra batch dimension at axis 0.
        Tensor of shape ``(batch,)`` with lengths along the padded dimension.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]

    if x[0].dim() == 1:
        return collate_seqs_1d(x, pad_value=pad_value)

    if pad_dim < 0:
        pad_dim = x[0].dim() + pad_dim

    if pad_dim != 0:
        x = [x_i.transpose(pad_dim, 0) for x_i in x]

    x_lengths: List[int] = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths, device=x[0].device)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if pad_dim != 0:
        x = x.transpose(1, pad_dim + 1)

    return x, x_lengths
