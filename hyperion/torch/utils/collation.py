"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def list_of_dicts_to_list(list_of_dicts, key):
    """Takes a list of dictionaries and a key,
    and returns a list of the items corresponding to the key
    """
    output = []
    for item in list_of_dicts:
        output.append(item[key])

    return output


def collate_seqs_1d(x, pad_value=0):
    """Combines a list/tuple of vectors with different lengths
       into a single tensor.

    Args:
        x: input lits/tuple of vectors.

    Returns:
      2D tensor with shape (num_vectors, max_vector_length).
      1D long tensor containing the vector lengths.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]

    assert x[0].dim() == 1
    x_lengths = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    return x, x_lengths


def collate_seqs_2d(x, pad_value=0, pad_dim=0):
    """Combines a list/tuple of matrices with different sizes in one of
       the dimensions into a single 3d tensor.
       Combines performing padding on the dimension which is not constant.

    Args:
        x: input lits/tuple of matrices.
        pad_dim: padding dimension.

    Returns:
      3D tensor with shape (num_vectors, max_length, feat_dim) or (num_vectors, feat_dim, length).
      1D long tensor containing the dimensions lengths.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]
    assert x[0].dim() == 2
    if pad_dim < 0:
        pad_dim = 2 + pad_dim

    if pad_dim != 0:
        x = [x_i.transpose(pad_dim, 0) for x_i in x]

    x_lengths = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if pad_dim != 0:
        x = x.transpose(1, pad_dim + 1)

    return x, x_lengths


def collate_seqs_nd(x, pad_value=0, pad_dim=0):
    """Combines a list/tuple of N-d tensors with different sizes in one of
       the dimensions into a single (N+1)-d tensor.
       Combines performing padding on the dimension which is not constant.

    Args:
        x: input lits/tuple of matrices.
        pad_dim: padding dimension.

    Returns:
      (N+1)-d combined tensor.
      1D long tensor containing the dimensions lengths.
    """
    if not isinstance(x[0], torch.Tensor):
        x = [torch.from_numpy(x_i) for x_i in x]

    if x[0].dim() == 1:
        return collate_seqs_1d(x, pad_value=pad_value)

    if pad_dim < 0:
        pad_dim = x[0].dim() + pad_dim

    if pad_dim != 0:
        x = [x_i.transpose(pad_dim, 0) for x_i in x]

    x_lengths = []
    for x_i in x:
        x_lengths.append(x_i.size(0))

    x_lengths = torch.as_tensor(x_lengths)
    x = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if pad_dim != 0:
        x = x.transpose(1, pad_dim + 1)

    return x, x_lengths


# def collate_seq_1d(x, pad_value=0):
#     """Combines a list/tuple of vectors with different lengths
#        into a single tensor.

#     Args:
#         x: input lits/tuple of vectors.

#     Returns:
#       2D tensor with shape (num_vectors, max_vector_length).
#       1D long tensor containing the vector lengths.
#     """
#     max_length = max([x_i.size(0) for x_i in x])
#     y = pad_value * torch.ones(len(x), max_length, dtype=x[0].dtype, device=x[0].device)
#     y_lengths = torch.empty(len(x), dtype=torch.long, device=x[0].device)
#     for i, x_i in enumerate(x):
#         y[i, : x_i.size(0)] = x_i
#         y_lengths[i] = x_i.size(0)

#     return y, y_lengths


# def collate_seq_2d(x, pad_value=0, pad_dim=-1):
#     """Combines a list/tuple of matrices with different sizes in one of
#        the dimensions into a single 3d tensor.
#        Combines performing padding on the dimension which is not constant.

#     Args:
#         x: input lits/tuple of matrices.
#         pad_dim: padding dimension.

#     Returns:
#       3D tensor with shape (num_vectors, max_length, feat_dim) or (num_vectors, feat_dim, length).
#       1D long tensor containing the dimensions lengths.
#     """
#     max_length = max([x_i.size(pad_dim) for x_i in x])
#     y_size = list(x[0].size())
#     y_size[pad_dim] = max_length
#     y = pad_value * torch.ones(*y_size, dtype=x[0].dtype, device=x[0].device)
#     y_lengths = torch.empty(len(x), dtype=torch.long, device=x[0].device)
#     if pad_dim == -1 or pad_dim == 1:
#         for i, x_i in enumerate(x):
#             y[i, :, : x_i.size(pad_dim)] = x_i
#             y_lengths[i] = x_i.size(pad_dim)
#     else:
#         for i, x_i in enumerate(x):
#             y[i, : x_i.size(pad_dim)] = x_i
#             y_lengths[i] = x_i.size(pad_dim)

#     return y, y_lengths


# def collate_seq_nd(x, pad_value=0, pad_dim=-1):
#     """Combines a list/tuple of N-d tensors with different sizes in one of
#        the dimensions into a single (N+1)-d tensor.
#        Combines performing padding on the dimension which is not constant.

#     Args:
#         x: input lits/tuple of matrices.
#         pad_dim: padding dimension.

#     Returns:
#       (N+1)-D combined tensor.
#       1D long tensor containing the dimensions lengths.
#     """
#     if x[0].dim() == 1:
#         return collate_seq_1d(x)

#     if x[0].dim() == 2:
#         return collate_seq_2d(x)

#     # here the general case
#     max_length = max([x_i.size(pad_dim) for x_i in x])
#     y_trans_size = list(x[0].transpose(0, pad_dim).size())
#     y = pad_value * torch.ones(*y_trans_size, dtype=x[0].dtype, device=x[0].device)
#     y_lengths = torch.empty(len(x), dtype=torch.long, device=x[0].device)
#     for i, x_i in enumerate(x):
#         y[i, : x_i.size(pad_dim)] = x_i.transpose(0, pad_dim)
#         y_lengths[i] = x_i.size(pad_dim)

#     if pad_dim > 0:
#         pad_dim = pad_dim + 1
#     y = y.transpose(1, pad_dim).contiguous()
#     return y, y_lengths
