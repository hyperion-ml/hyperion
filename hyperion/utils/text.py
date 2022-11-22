"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from pathlib import Path

import k2
import k2.version
import numpy as np
import pandas as pd
import torch



# Copied and modified from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/mask.py
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret



def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim

    max_len = lengths.max()
    n = lengths.size(0)

    expaned_lengths = torch.arange(max_len).expand(n, max_len).to(lengths)

    return expaned_lengths >= lengths.unsqueeze(1)


def concat(
    ragged: k2.RaggedTensor, value: int, direction: str
) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def read_text(text_file: str):
    # assert check_argument_types()
    text_file = Path(text_file)

    data = {"id":[],"text":[]}
    with Path(text_file).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            # if k in data:
            #     raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data["id"].append(k)
            data["text"].append(v)
    return pd.DataFrame(data=data, index=data["id"])

