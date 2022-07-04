"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
import torch


def eval_nnet_by_chunks(x, nnet, chunk_length=0, detach_chunks=True, time_dim=-1):

    device = None if nnet.device == x.device else nnet.device
    T = x.shape[time_dim]
    if T <= chunk_length or chunk_length == 0:
        if device is not None:
            x = x.to(device)

        y = nnet(x)
        if detach_chunks:
            y = y.detach()
        return y

    try:
        left_context, right_context = nnet.in_context()
    except:
        left_context = right_context = 0

    in_shape = x.shape
    chunk_shift_in = chunk_length - left_context - right_context

    try:
        out_shape = nnet.out_shape(in_shape)
        T_out = out_shape[time_dim]
        r = float(T_out) / T
    except:
        out_shape = None

    num_chunks = int(math.ceil((T - chunk_length) / chunk_shift_in + 1))
    # move time dimension to dim 0
    x = x.transpose(0, time_dim)
    y = None
    tbeg_in = 0
    tbeg_out = 0
    for i in range(num_chunks):
        tend_in = min(tbeg_in + chunk_length, x.shape[0])
        # get slice and move back time dimension to last dim
        x_i = x[tbeg_in:tend_in].transpose(0, time_dim)
        if device is not None:
            x_i = x_i.to(device)

        y_i = nnet(x_i)
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
            y = torch.zeros(out_shape)
            # move time dimension to dim 0
            y = y.transpose(0, time_dim)

        y_i = y_i.transpose(0, time_dim)

        if i == 0:
            tend_out = min(tbeg_out + chunk_length_out, T_out)
            y[tbeg_out:tend_out] = y_i
            tbeg_out = +(chunk_length_out - right_context_out)
        else:
            tend_out = min(
                int(round(tbeg_out)) + chunk_length_out - left_context_out, T_out
            )
            dt = tend_out - tbeg_out
            if dt > 0:
                # print('eu', tbeg_out, tend_out, left_context_out,left_context_out+dt, T_out, chunk_length, chunk_length_out, tbeg_in, tend_in)
                y[tbeg_out:tend_out] = y_i[left_context_out : left_context_out + dt]
                tbeg_out += chunk_shift_out

        tbeg_in += chunk_shift_in

    # put time dimension back in its place
    y = y.transpose(0, time_dim)

    return y


def eval_nnet_overlap_add(
    x, nnet, chunk_length=0, chunk_overlap=None, detach_chunks=True, time_dim=-1
):

    device = None if nnet.device == x.device else nnet.device

    # assume time is the last dimension
    T = x.shape[time_dim]
    if T <= chunk_length or chunk_length == 0:
        if device is not None:
            x = x.to(device)
        y = nnet(x)
        if detach_chunks:
            y = y.detach()
        return y

    if chunk_overlap is None:
        # infer chunk overlap from network input context
        try:
            left_context, right_context = nnet.in_context()
        except:
            left_context = right_context = 0

        chunk_overlap = left_context + right_context

    in_shape = x.shape
    chunk_shift_in = chunk_length - chunk_overlap

    try:
        out_shape = nnet.out_shape(in_shape)
        T_out = out_shape[time_dim]
        r = float(T_out) / T
    except:
        out_shape = None

    num_chunks = int(math.ceil((T - chunk_length) / chunk_shift_in + 1))
    # move time dimension to dim 0
    x = x.transpose(0, time_dim)
    y = None
    N = None
    tbeg_in = 0
    tbeg_out = 0
    for i in range(num_chunks):
        tend_in = min(tbeg_in + chunk_length, x.shape[0])
        # get slice and move back time dimension to last dim
        x_i = x[tbeg_in:tend_in].transpose(0, time_dim)
        if device is not None:
            x_i = x_i.to(device)

        y_i = nnet(x_i)
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
            y = torch.zeros(out_shape)
            # move time dimension to dim 0
            y = y.transpose(0, time_dim)
            count = torch.zeros(T_out)

        y_i = y_i.transpose(0, time_dim)

        tend_out = min(int(round(tbeg_out)) + chunk_length_out, T_out)
        dt = tend_out - tbeg_out
        y[tbeg_out:tend_out] += y_i[:dt]
        count[tbeg_out:tend_out] += 1
        tbeg_out += chunk_shift_out
        tbeg_in += chunk_shift_in

    # put time dimension back in his place and normalize
    y = y.transpose(0, time_dim) / count

    return y
