"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..hyper_torch_model import HyperTorchModel


def eval_nnet_by_chunks(
    x: torch.Tensor,
    nnet: "HyperTorchModel",
    chunk_length: int = 0,
    detach_chunks: bool = True,
    time_dim: int = -1,
) -> torch.Tensor:
    """Evaluates a network on long sequences by splitting them into chunks.

    The function handles optional left/right model context (when available via
    ``nnet.in_context()``), stitches chunk outputs back together, and can detach
    per-chunk outputs from the graph.

    Args:
        x: Input tensor containing a time dimension.
        nnet: Network model exposing ``device`` and optionally
            ``in_context()``/``out_shape(in_shape)`` helpers.
        chunk_length: Input chunk size along ``time_dim``. If 0, evaluate in one pass.
        detach_chunks: If True, detaches chunk outputs before aggregation.
        time_dim: Tensor dimension corresponding to time.

    Returns:
        Network output tensor for the full sequence.
    """
    device = None if nnet.device == x.device else nnet.device
    T = x.shape[time_dim]
    if T <= chunk_length or chunk_length == 0:
        if device is not None:
            x = x.to(device)

        y = nnet(x)
        if isinstance(y, tuple):
            y = y[0]
        if detach_chunks:
            y = y.detach()
        return y

    try:
        left_context, right_context = nnet.in_context()
    except AttributeError:
        left_context = right_context = 0

    in_shape = x.shape
    chunk_shift_in = chunk_length - left_context - right_context
    if chunk_shift_in <= 0:
        raise ValueError(
            "chunk_length must be greater than left_context + right_context, "
            f"got chunk_length={chunk_length}, left_context={left_context}, "
            f"right_context={right_context}"
        )

    try:
        out_shape = nnet.out_shape(in_shape)
        T_out = out_shape[time_dim]
        r = float(T_out) / T
    except AttributeError:
        out_shape = None

    num_chunks = int(math.ceil((T - chunk_length) / chunk_shift_in + 1))
    # move time dimension to dim 0
    x = x.transpose(0, time_dim)
    y = None
    tbeg_in = 0
    tbeg_out = 0
    last_covered = 0
    for i in range(num_chunks):
        tend_in = min(tbeg_in + chunk_length, x.shape[0])
        # get slice and move back time dimension to last dim
        x_i = x[tbeg_in:tend_in].transpose(0, time_dim)
        if device is not None:
            x_i = x_i.to(device)

        y_i = nnet(x_i)
        if isinstance(y_i, tuple):
            y_i = y_i[0]
        if detach_chunks:
            y_i = y_i.detach()

        chunk_length_out = y_i.shape[time_dim]
        if out_shape is None:
            # infer chunk_shift in the output
            r = float(chunk_length_out) / chunk_length

            # infer total output length
            T_out = int(r * T)
            out_shape = list(y_i.shape)
            out_shape[time_dim] = T_out

        if y is None:
            right_context_out = int(math.floor(r * right_context))
            left_context_out = int(math.floor(r * left_context))
            chunk_shift_out = chunk_length_out - right_context_out - left_context_out
            # create output tensor
            y = torch.zeros(out_shape, device=y_i.device, dtype=y_i.dtype)
            # move time dimension to dim 0
            y = y.transpose(0, time_dim)

        y_i = y_i.transpose(0, time_dim)

        if i == 0:
            tend_out = min(tbeg_out + chunk_length_out, T_out)
            y[tbeg_out:tend_out] = y_i
            last_covered = max(last_covered, tend_out)
            tbeg_out = +(chunk_length_out - right_context_out)
        else:
            tend_out = min(
                int(round(tbeg_out)) + chunk_length_out - left_context_out, T_out
            )
            dt = tend_out - tbeg_out
            if dt > 0:
                # print('eu', tbeg_out, tend_out, left_context_out,left_context_out+dt, T_out, chunk_length, chunk_length_out, tbeg_in, tend_in)
                y[tbeg_out:tend_out] = y_i[left_context_out : left_context_out + dt]
                last_covered = max(last_covered, tend_out)
                tbeg_out += chunk_shift_out

        tbeg_in += chunk_shift_in

    if last_covered < T_out:
        if last_covered == 0:
            raise RuntimeError("Unable to cover any output positions in chunked eval")
        # Fill uncovered tail with the last valid frame.
        y[last_covered:T_out] = y[last_covered - 1 : last_covered]

    # put time dimension back in its place
    y = y.transpose(0, time_dim)

    return y


def eval_nnet_overlap_add(
    x: torch.Tensor,
    nnet: "HyperTorchModel",
    chunk_length: int = 0,
    chunk_overlap: Optional[int] = None,
    detach_chunks: bool = True,
    time_dim: int = -1,
) -> torch.Tensor:
    """Evaluates a network by overlap-add aggregation of chunked outputs.

    Chunks are extracted with a fixed overlap and their outputs are accumulated.
    Overlapped output positions are normalized by the number of contributing
    chunks.

    Args:
        x: Input tensor containing a time dimension.
        nnet: Network model exposing ``device`` and optionally
            ``in_context()``/``out_shape(in_shape)`` helpers.
        chunk_length: Input chunk size along ``time_dim``. If 0, evaluate in one pass.
        chunk_overlap: Overlap size in input samples. If None, inferred from
            ``nnet.in_context()`` when available.
        detach_chunks: If True, detaches chunk outputs before aggregation.
        time_dim: Tensor dimension corresponding to time.

    Returns:
        Network output tensor for the full sequence.
    """
    device = None if nnet.device == x.device else nnet.device

    # assume time is the last dimension
    T = x.shape[time_dim]
    if T <= chunk_length or chunk_length == 0:
        if device is not None:
            x = x.to(device)
        y = nnet(x)
        if isinstance(y, tuple):
            y = y[0]
        if detach_chunks:
            y = y.detach()
        return y

    if chunk_overlap is None:
        # infer chunk overlap from network input context
        try:
            left_context, right_context = nnet.in_context()
        except AttributeError:
            left_context = right_context = 0

        chunk_overlap = left_context + right_context

    in_shape = x.shape
    chunk_shift_in = chunk_length - chunk_overlap
    if chunk_shift_in <= 0:
        raise ValueError(
            "chunk_length must be greater than chunk_overlap, "
            f"got chunk_length={chunk_length}, chunk_overlap={chunk_overlap}"
        )

    try:
        out_shape = nnet.out_shape(in_shape)
        T_out = out_shape[time_dim]
        r = float(T_out) / T
    except AttributeError:
        out_shape = None

    num_chunks = int(math.ceil((T - chunk_length) / chunk_shift_in + 1))
    # move time dimension to dim 0
    x = x.transpose(0, time_dim)
    y = None
    tbeg_in = 0
    tbeg_out = 0
    prev_end = 0
    for i in range(num_chunks):
        tend_in = min(tbeg_in + chunk_length, x.shape[0])
        # get slice and move back time dimension to last dim
        x_i = x[tbeg_in:tend_in].transpose(0, time_dim)
        if device is not None:
            x_i = x_i.to(device)

        y_i = nnet(x_i)
        if isinstance(y_i, tuple):
            y_i = y_i[0]
        if detach_chunks:
            y_i = y_i.detach()

        chunk_length_out = y_i.shape[time_dim]
        if out_shape is None:
            # infer chunk_shift in the output
            r = float(chunk_length_out) / chunk_length

            # infer total output length
            T_out = int(r * T)
            out_shape = list(y_i.shape)
            out_shape[time_dim] = T_out

        if y is None:
            chunk_shift_out = r * chunk_shift_in
            # create output tensor
            y = torch.zeros(out_shape, device=y_i.device, dtype=y_i.dtype)
            # move time dimension to dim 0
            y = y.transpose(0, time_dim)
            count = torch.zeros(T_out, device=y_i.device, dtype=y_i.dtype)

        y_i = y_i.transpose(0, time_dim)

        tbeg_out_i = int(round(tbeg_out))
        tbeg_out_i = min(tbeg_out_i, prev_end)
        tbeg_out_i = max(0, tbeg_out_i)
        tend_out = min(tbeg_out_i + chunk_length_out, T_out)

        dt = tend_out - tbeg_out_i
        if dt > 0:
            y[tbeg_out_i:tend_out] += y_i[:dt]
            count[tbeg_out_i:tend_out] += 1
            prev_end = max(prev_end, tend_out)
        tbeg_out += chunk_shift_out
        tbeg_in += chunk_shift_in

    # put time dimension back in his place and normalize
    if prev_end < T_out:
        if prev_end == 0:
            raise RuntimeError("Unable to cover any output positions in overlap-add eval")
        # Fill uncovered tail with the last valid frame.
        y[prev_end:T_out] = y[prev_end - 1 : prev_end]
        count[prev_end:T_out] = count[prev_end - 1 : prev_end]

    if torch.any(count[:prev_end] == 0):
        raise RuntimeError("Uncovered middle positions found in overlap-add aggregation")

    y = y.transpose(0, time_dim) / count

    return y
