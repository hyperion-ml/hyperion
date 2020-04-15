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
        super(ScaledDotProdAttV1, self).__init__()
        # We assume d_v always equals d_k
        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
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
            query = query.transpose(1, self.time_dim)
            key = key.transpose(1, self.time_dim)
            value = value.transpose(1, self.time_dim)

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
            min_value = -1e20
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        if self.dropout_rate > 0:
            p_attn = self.dropout(self.attn)
        else:
            p_attn = self.attn

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)



class LocalScaledDotProdAttV1(ScaledDotProdAttV1):
    """Inner product Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, in_feats, out_feats, num_heads, d_k, d_v, 
                 context=25, dropout_rate=0, time_dim=1):
        """Construct an MultiHeadedAttention object."""
        super(LocalScaledDotProdAttV1, self).__init__(
            in_feats, out_feats, num_heads, d_k, d_v, 
            dropout_rate, time_dim)
        self.context = context

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = ('{}(in_feats={}, out_feats={}, num_heads={}, d_k={}, d_v={}, '
             'context={}, dropout_rate={}, time_dim={})'.format(
            self.__class__.__name__, self.in_feats, self.out_feats, self.num_heads,
            self.d_k, self.d_v, self.context, self.dropout_rate, self.time_dim))
        return s

    @staticmethod
    def _softmax(scores1, scores2, shift1, shift2, t1, t2):
        
        # scores1 = (batch, heads, blocks, t1, t2)
        # scores2 = (batch, heads, blocks-1 , t1, t2)
        batch_size = scores1.size(0)
        num_heads = scores1.size(1)
        num_blocks = scores1.size(2)
        context1 = scores1.size(3)
        context2 = scores1.size(4)

        # set elements in scores2 that overlap with elements in scores1 to -inf
        scores2[:,:,:,:context1-shift1,:context2-shift2] = -1e20
        scores2[:,:,:,shift1:,shift2:] = -1e20

        #set the padding time steps that we had to add to make integer block-number to -inf
        # in scores1
        dt1 = max(0, scores1.size(2)*scores1.size(3) - t1)
        dt2 = max(0, scores1.size(2)*scores1.size(4) - t2)
        if dt1 > 0  or dt2 > 0:
            scores1[:,:,-1,-dt1:,-dt2:] = -1e20

            # in scores2
            dt1 = max(0, dt1 - shift1)
            dt2 = max(0, dt2 - shift2)
            if dt1 > 0  or dt2 > 0:
                scores2[:,:,-1,-dt1:,-dt2:] = -1e20
                
        #flatten blocks and time1 dimensions
        scores1 = scores1.view(batch_size, num_heads, -1, context2)
        scores2 = scores2.view(batch_size, num_heads, -1, context2)
        #print('aa', scores1.shape, scores2.shape)
        #pad scores2  to have the same size as scores1
        scores2 = nn.functional.pad(scores2, (0, 0, shift1, context1-shift1),
                                    mode='constant', value=-1e20)
        #print('bb', scores1.shape, scores2.shape)
        #concat scores1, scores2 and do softmax in time2 dimension
        # (batch, heads, blocks*time1, 2*time2)
        probs = torch.softmax(torch.cat((scores1, scores2), dim=-1), dim=-1)
        
        #now we separate back probs into probs1, and probs2
        #probs1
        probs1 = probs[:,:,:,:context2].contiguous().view(
            batch_size, num_heads, num_blocks, -1, context2)
        #probs2
        probs2 = probs[:,:,shift1:-(context1-shift1),context2:].contiguous().view(
            batch_size, num_heads, num_blocks-1, -1, context2)
       
        return probs1, probs2
            


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
        t1 = query.size(self.time_dim)
        t2 = key.size(self.time_dim)
        if t2 <= self.context:
            return super(LocalScaledDotProdAttV1, self).forward(
                query, key, value, mask)

        if self.time_dim != 1:
            query = query.transpose(1, self.time_dim)
            key = key.transpose(1, self.time_dim)
            value = value.transpose(1, self.time_dim)

        context_k = self.context
        num_blocks = math.ceil(t2/context_k)    #(t2 + context_k//2)//context_k
        context_q = math.ceil(t1/num_blocks)
        num_blocks_q = math.ceil(t1/context_q) #(t1 + context_q//2)//context_q
        assert num_blocks == num_blocks_q, (
            'num_blocks_k({})!=num_blocks_q({}), context_k={}, context_q={}, t1={}, t2={}'.format(
                num_blocks, num_blocks_q, context_k, context_q, t1, t2))
        pad1 = context_q * num_blocks - t1
        pad2 = context_k * num_blocks - t2
        # print('1',query.shape,key.shape,value.shape,pad1,pad2, context_q, context_k)
        if pad1 > 0:
            query = nn.functional.pad(query, (0, 0, 0, pad1))

        if pad2 > 0:
            key = nn.functional.pad(key, (0, 0, 0, pad2))
            value = nn.functional.pad(value, (0, 0, 0, pad2))

        # print('2',query.shape,key.shape,value.shape)
        q0 = self.linear_q(query)
        k0 = self.linear_k(key)
        v0 = self.linear_v(value)

        # print('3',q0.shape,k0.shape,v0.shape)
        # compute block diagonal affinity matrix
        q = q0.view(
            batch_size, -1, self.num_heads, self.d_k).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks, -1, self.d_k)  
            # (batch, head, blocks, time1, d_k)
        k = k0.view(
            batch_size, -1, self.num_heads, self.d_k).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks, -1, self.d_k)  
            # (batch, head, blocks time2, d_k)
        # print('4',q.shape,k.shape)

        scores1 = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  
        # (batch, blocks, head, time1, time2)
        # print('5',scores1.shape)

        # compute shifted block diagonal affinity matrix
        q_left_shift = context_q // 2
        q_right_shift = context_q - q_left_shift
        k_left_shift = context_k // 2
        k_right_shift = context_k - k_left_shift
        q = q0[:,q_left_shift:-q_right_shift].view(
            batch_size, -1, self.num_heads, self.d_k).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks-1, -1, self.d_k)  
            # (batch, blocks-1, head, time1, d_k)
        k = k0[:,k_left_shift:-k_right_shift].view(
            batch_size, -1, self.num_heads, self.d_k).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks-1, -1, self.d_k)  
            # (batch, blocks-1, head, d_k)
        # print('6',q.shape,k.shape)

        scores2 = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  
        # (batch, blocks-1, head, time1, time2)
        # print('7',scores2.shape)

        #combine both block diagonal affinity matrix to do the softmax
        if mask is not None:
            raise NotImplementedError()
        else:
            self.attn1, self.attn2 = self._softmax(
                scores1, scores2, q_left_shift, k_left_shift, t1, t2)

        if self.dropout_rate > 0:
            p_attn1 = self.dropout(self.attn1)
            p_attn2 = self.dropout(self.attn2)
        else:
            p_attn1 = self.attn1
            p_attn2 = self.attn2

        v = v0.view(
            batch_size, -1, self.num_heads, self.d_v).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks, -1, self.d_k)  
        # (batch, heads, blocks, time2, d_v)
        # print('8',p_attn1.shape,p_attn2.shape, v.shape)
        # (batch, blocks, head, time1, time2) x (batch, blocks, head, time2, d_v)
        x = torch.matmul(p_attn1, v)  # (batch, heads, blocks, time1, d_k)
        # print('9',x.shape)
        x = x.view(batch_size, self.num_heads, -1, self.d_k).transpose(
            1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  
        # (batch, time1, d_model)
        # print('10',x.shape)

        v = v0[:,k_left_shift:-k_right_shift].view(
            batch_size, -1, self.num_heads, self.d_v).transpose(
                1, 2).contiguous().view(
                    batch_size, self.num_heads, num_blocks-1, -1, self.d_v)  
        # (batch, blocks-1, head, time2, d_v)
        # print('11',p_attn1.shape,p_attn2.shape, v.shape)
        # (batch, blocks-1, head, time1, time2) x (batch, blocks-1, head, time2, d_v)
        x2 = torch.matmul(p_attn2, v)  # (batch, heads, blocks-1, time1, d_k)
        # print('12',x2.shape)
        x2 = x2.view(batch_size, self.num_heads, -1, self.d_k).transpose(
            1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  
        # (batch, time1, d_model)
        # print('12',x2.shape)
        x[:,q_left_shift:-q_right_shift:] = x[:,q_left_shift:-q_right_shift:] + x2
        x = x[:,:t1]
        return self.linear_out(x)  # (batch, time1, d_model)
