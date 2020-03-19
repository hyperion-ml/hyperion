"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import math

#import numpy
import torch
from torch import nn


class ScaledDotProdAttV1(nn.Module):
    """Inner product Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, in_feats, out_feats, num_heads, d_k, d_v, dropout_rate=0, time_dim=1):
        """Construct an MultiHeadedAttention object."""
        super(ScaledDotProdAtt, self).__init__()
        # We assume d_v always equals d_k
        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.droput_rate = droput_rate
        self.time_dim = time_dim
        self.linear_q = nn.Linear(in_feats, num_heads*d_k)
        self.linear_k = nn.Linear(in_feats, num_heads*d_k)
        self.linear_v = nn.Linear(in_feats, num_heads*d_v)
        self.linear_out = nn.Linear(num_heads*d_v, out_feats)
        self.attn = None
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)


    @property
    def in_feats(self):
        return self.linear_v.in_features

    @property
    def out_feats(self):
        return self.linear_out.out_features

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{}(in_feats={}, out_feats={}, num_heads={}, d_k={}, d_v={}, dropout_rate={}, time_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats, self.num_heads,
            self.d_k, self.d_v, self.dropout_rate, self.time_dim)
        return s


    def forward(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        batch_size = value.size(0)
        if self.time_dim != 1:
            q = q.transpose(1, time_dim)
            k = k.transpose(1, time_dim)
            v = v.transpose(1, time_dim)

        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_v)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            #min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            min_value = -1e200
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        if self.dropout_rate > 0:
            p_attn = self.dropout(self.attn)
        else:
            p_att = self.attn

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)



class LocalScaledDotProdAttV1(nn.Module):
    """Inner product Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, in_feats, out_feats, num_heads, d_k, d_v, context=25, dropout_rate=0, time_dim=1):
        """Construct an MultiHeadedAttention object."""
        super(LocalScaledDotProdAtt, self).__init__()
        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.context = context
        self.droput_rate = droput_rate
        self.time_dim = time_dim
        self.linear_q = nn.Linear(in_feats, num_heads*d_k)
        self.linear_k = nn.Linear(in_feats, num_heads*d_k)
        self.linear_v = nn.Linear(in_feats, num_heads*d_v)
        self.linear_out = nn.Linear(num_heads*d_v, out_feats)

        self.attn = None
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        batch_size = query.size(0)
        if self.time_dim != 1:
            q = q.transpose(1, time_dim)
            k = k.transpose(1, time_dim)
            v = v.transpose(1, time_dim)

        t1 = q.size(1)
        t2 = k.size(1)
        num_blocks = (t1 + self.context//2)//self.context
        num_blocks2 = (t2 + self.context//2)//self.context
        assert num_blocks == num_blocks2
        pad1 = self.context * num_blocks - t1
        pad2 = self.context * num_blocks - t2
        if pad1 > 0:
            q = nn.functional.pad(q, (0, pad1, 0, 0))

        if pad2 > 0:
            k = nn.functional.pad(k, (0, pad2, 0, 0))
            v = nn.functional.pad(v, (0, pad2, 0, 0))

        q0 = self.linear_q(query)
        k0 = self.linear_k(key)
        v0 = self.linear_v(value)

        # compute block diagonal attention
        q = q0.view(
            batch_size, -1, num_blocks, self.num_heads, self.d_k).transpose(
                1, 3)  # (batch, blocks, head, time1, d_k)
        k = k0.view(
            batch_size, -1, num_blocks, self.num_heads, self.d_k).transpose(
                1, 3)  # (batch, blocks, head, d_k)
        v = v0.view(
            batch_size, -1, num_blocks, self.num_heads, self.d_v).transpose(
                1, 3)  # (batch, blocks, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, blocks, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsquezee(1).eq(0)  # (batch, 1, 1, time1, time2)
            #min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            min_value = -1e200
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, blocks, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, blocks, head, time1, time2)

        if self.dropout_rate > 0:
            p_attn = self.dropout(self.attn)
        else:
            p_attn = self.attn

        x = torch.matmul(p_attn, v)  # (batch, blocks, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  # (batch, time1, d_model)

        # compute shifted block diagonal attention
        left_shift = self.context // 2
        right_shift = self.context - shift
        q = q0[:,left_shift:-right_shift].view(
            batch_size, -1, num_blocks-1, self.num_heads, self.d_k).transpose(
                1, 3)  # (batch, blocks, head, time1, d_k)
        k = k0[:,left_shift:-right_shift].view(
            batch_size, -1, num_blocks-1, self.num_heads, self.d_k).transpose(
                1, 3)  # (batch, blocks, head, d_k)
        v = v0[:,left_shift:-right_shift].view(
            batch_size, -1, num_blocks-1, self.num_heads, self.d_v).transpose(
                1, 3)  # (batch, blocks, head, time2, d_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, blocks, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsquezee(1).eq(0)  # (batch, 1, 1, time1, time2)
            #min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            min_value = -1e200
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, blocks, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, blocks, head, time1, time2)

        if self.dropout_rate > 0:
            p_attn = self.dropout(self.attn)

        x2 = torch.matmul(p_attn, v)  # (batch, blocks, head, time1, d_k)
        x2 = x2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  # (batch, time1, d_model)
        x2 = nn.functional.pad(x2, (left_shift, right_shfit, 0, 0))
        x = x + x2
        x = x[:,:t1]
        return self.linear_out(x)  # (batch, time1, d_model)
