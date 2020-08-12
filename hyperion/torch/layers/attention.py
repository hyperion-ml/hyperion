"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
# from __future__ import absolute_import

import math

#import numpy
import torch
from torch import nn


class ScaledDotProdAttV1(nn.Module):
    """Scaled dot product multihead attention layer

    Attributes:
       in_feats: input feature dimension
       out_feats: output feature dimension
       num_heads: number of heads
       d_k: key/query projection dimension
       d_v: value projection dimension
       dropout_rate: dropout rate
       time_dim: time dimension in the input, default=1 meaning input 
                 dimensions are (batch, time, in_feats)
    """

    def __init__(self, in_feats, out_feats, num_heads, d_k, d_v, dropout_rate=0, time_dim=1):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
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
        """Computes 'Scaled Dot Product Attention'.

        Args:
           query: query with size=(batch, time1, in_feats), 
                  where time1 is the output time dimension
           key: key with size=(batch, time2, in_feats)
                  where time1 is the input time dimension
           value: value with size=(batch, time2, in_feats)
           mask: optional mask with size=(batch, time1, time2), 
                  to zero attention between some time steps
                  or size=(batch, time) to make time1=time2
        Returns:
           Attention weigthed average of the value with size=(batch, time1, out_feats)
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
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2) or (batch, 1, time)
            #min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            min_value = -1e20
            if mask.dim() == 4:
                scores = scores.masked_fill(mask, min_value)
                self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
            else:
                mask1 = mask.unsqueze(2)
                mask2 = mask.unsqueeze(-1)
                scores = scores.masked_fill(mask1, min_value)
                scores = scores.masked_fill(mask2, min_value)
                self.attn = torch.softmax(scores, dim=-1) # (batch, head, time1, time2)
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
    """Local Scaled dot product multihead attention layer 
       I calculates self-attention between time steps within
       a window of 'context' frames.

    Attributes:
       in_feats: input feature dimension
       out_feats: output feature dimension
       num_heads: number of heads
       d_k: key/query projection dimension
       d_v: value projection dimension
       context: maximum attention temporal context.
       dropout_rate: dropout rate
       time_dim: time dimension in the input, default=1 meaning input 
                 dimensions are (batch, time, in_feats)
    """

    def __init__(self, in_feats, out_feats, num_heads, d_k, d_v, 
                 context=25, dropout_rate=0, time_dim=1):
        """Construct an MultiHeadedAttention object."""
        super().__init__(
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
        """Computes softmax for block diagonal attention maps

        Args:
          scores1: attention scores from block-diagonal score matrix
                   with size=(batch, heads, blocks, t1, t2)
          scores2: attention scores from a shifted block-diagonal score matrix
                   with size=(batch, heads, blocks-1, t1, t2)
          shift1: shift of diagonal blocks of scores2 wrt scores1 in time steps in the 
                  time dimension 1
          shift2: shift of diagonal blocks of scores2 wrt scores1 in time steps in the 
                  time dimension 2, with self-attention shift1=shift2
          t1: length of time dimension 1 (output time dimension)
          t2: length of time dimension 2 (input time dimension), with self-att t1=t2.

        Returns
          probs1: posterior attention scores for block-diagonal att. matrix
                   with size=(batch, heads, blocks, t1, t2)
          probs2: posterior attention scores for a shifted block-diagonal att. matrix
                   with size=(batch, heads, blocks-1, t1, t2)
           
        """
        if scores2.dtype == torch.half:
            min_val = -65504
        else:
            min_val = -1e20

        batch_size = scores1.size(0)
        num_heads = scores1.size(1)
        num_blocks = scores1.size(2)
        context1 = scores1.size(3)
        context2 = scores1.size(4)

        # set elements in scores2 that overlap with elements in scores1 to -inf
        scores2[:,:,:,:context1-shift1,:context2-shift2] = min_val
        scores2[:,:,:,shift1:,shift2:] = min_val

        #set the padding time steps that we had to add to make integer block-number to -inf
        # in scores1
        dt1 = max(0, scores1.size(2)*scores1.size(3) - t1)
        dt2 = max(0, scores1.size(2)*scores1.size(4) - t2)
        if dt1 > 0  or dt2 > 0:
            scores1[:,:,-1,-dt1:,-dt2:] = min_val

            # in scores2
            dt1 = max(0, dt1 - shift1)
            dt2 = max(0, dt2 - shift2)
            if dt1 > 0  or dt2 > 0:
                scores2[:,:,-1,-dt1:,-dt2:] = min_val
                
        #flatten blocks and time1 dimensions
        scores1 = scores1.view(batch_size, num_heads, -1, context2)
        scores2 = scores2.view(batch_size, num_heads, -1, context2)
        #print('aa', scores1.shape, scores2.shape)
        #pad scores2  to have the same size as scores1
        scores2 = nn.functional.pad(scores2, (0, 0, shift1, context1-shift1),
                                    mode='constant', value=min_val)
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
            


    def _mask_scores_1d(self, scores, mask, shift1, shift2):
        if scores.dtype == torch.half:
            min_value = -65504
        else:
            min_value = -1e20

        batch_size = scores.size(0)
        num_blocks = scores.size(2)
        context1 = scores.size(3)
        context2 = scores.size(4)
        mask_blocks = torch.ones_like(scores dtype=mask.dtype)
        mask_single_block = torch.zeros(
            (batch_size, context1, context2), dtype=mask.dtype)

        t1_start = shift1
        t2_start = shift2
        for block in range(num_blocks):
            t1_end = t1_start + context1
            t2_end = t2_start + context2
            mask_single_block.fill_(False)
            mask_single_block.masked_fill_(mask[:,0,t1_start:t1_end], True)
            mask_single_block.masked_fill_(mask[:,:,t2_start:t2_end], True)
            mask_blocks[:,block] = mask_single_block
            t1_start += context1
            t2_start += context2

        return scores.masked_fill(mask_blocks, min_value)
        

    def _mask_scores_2d(self, scores, mask, shift1, shift2):
        if scores.dtype == torch.half:
            min_value = -65504
        else:
            min_value = -1e20

        batch_size = scores.size(0)
        num_blocks = scores.size(2)
        context1 = scores.size(3)
        context2 = scores.size(4)
        mask_blocks = torch.ones_like(scores, dtype=mask.dtype)
        t1_start = shift1
        t2_start = shift2
        for block in range(num_blocks):
            t1_end = min(t1_start + context1, mask.size(1))
            t2_end = min(t2_start + context2, mask.size(2))
            mask_blocks[:,block,:(t1_end-t1_start),:(t2_end-t2_start)] = mask[
                :,t1_start:t1_end,t2_start:t2_end]
            t1_start += context1
            t2_start += context2

        return scores.masked_fill(mask_blocks, min_value)



    def forward(self, query, key, value, mask):
        """Computes 'Local Scaled Dot Product Attention'.

        Args:
           query: query with size=(batch, time1, in_feats), 
                  where time1 is the output time dimension
           key: key with size=(batch, time2, in_feats)
                  where time1 is the input time dimension
           value: value with size=(batch, time2, in_feats)
           mask: optional mask with size=(batch, time1, time2), 
                  to zero attention between some time steps.
                 or (batch, time) if time1=time2
        Returns:
           Attention weigthed average of the values with size=(batch, time1, out_feats)
        """
        batch_size = query.size(0)
        t1 = query.size(self.time_dim)
        t2 = key.size(self.time_dim)
        if t2 <= self.context:
            return super().forward(
                query, key, value, mask)

        if self.time_dim != 1:
            query = query.transpose(1, self.time_dim)
            key = key.transpose(1, self.time_dim)
            value = value.transpose(1, self.time_dim)

        context_k = self.context
        num_blocks = math.ceil(t2 / context_k)    #(t2 + context_k//2)//context_k
        context_q = math.ceil(t1 / num_blocks)
        num_blocks_q = math.ceil(t1 / context_q) #(t1 + context_q//2)//context_q
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
            # put to -inf scores in points where mask==0
            if mask.dim() == 4:
                # case when mask is 2d matrix per batch element
                mask = mask.eq(0)  # (batch, time1, time2) 

                # first, we mask block diagonal blocks
                scores1 = self._mask_scores_2d(scores1, mask, 0, 0)

                # second, we mask shifted block diagonal blocks
                scores2 = self._mask_scores_2d(scores2, mask, q_left_shift, k_left_shift)

            else:
                # case when mask is 1d vector per batch element,
                # meaning that time1 and time2 are the same, so mask is symmetric
                mask = nn.functional.pad(mask, (0, pad2))
                mask = mask.squeeze(1).eq(0)  # (batch, 1, time) 

                # first, we mask block diagonal blocks
                scores1 = self._mask_scores_1d(scores1, mask, 0, 0)

                # second, we mask shifted block diagonal blocks
                scores2 = self._mask_scores_1d(scores2, mask, q_left_shift, k_left_shift)

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
