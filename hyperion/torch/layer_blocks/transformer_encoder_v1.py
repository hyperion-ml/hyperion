"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

from ..layers.transformer_feedforward import *
from ..layers.attention import *

class TransformerEncoderBlockV1(nn.Module):
    """Encoder layer module.
    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, num_feats, self_attn, num_heads, feed_forward, d_ff, ff_kernel_size, ff_dropout_rate=0,
                 att_context=25, att_dropout_rate=0, norm_before=True, concat_after=False):

        """Construct an EncoderLayer object."""
        super(TransformerEncoderBlock, self).__init__()
        if isinstance(self_att, str):
            self.self_attn = _make_att(self_att, num_feats, num_heads, att_context, att_dropout_rate)
        else:
            self.self_attn = self_attn

        if isinstance(feed_forward, str):
            self.feed_forward = _make_ff(feed_forward, d_model, d_ff, ff_kernel_size, dropout_rate)
        else:
            self.feed_forward = feed_forward

        self.norm1 = nn.LayerNorm(num_feats)
        self.norm2 = nn.LayerNorm(num_feats)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.norm_before = norm_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(num_feats + num_feats, num_feats)


    @staticmethod
    def _make_att(att_type, num_feats, num_heads, context, dropout_rate):
        
        assert num_feats % num_heads == 0
        d_k = num_feats // num_heads

        if att_type == 'scaled-dot-v1':
            return ScaledDotProdAtt(num_feats, num_feats, num_heads, d_k, d_k, dropout_rate, time_dim=1)

        if att_type == 'local-scaled-dot-v1':
            return LocalScaledDotProdAtt(num_feats, num_feats, num_heads, d_k, d_k, context, dropout_rate, time_dim=1)

        
    @staticmethod
    def _make_ff(ff_type, num_feats, hid_feats, kernel_size, dropout_rate):

        if ff_type == 'linear':
            return PositionwiseFeedForward(
                num_feats, hid_feats, dropout_rate, time_dim=1)

        if ff_type == 'conv1dx2':
            return Conv1dx2(
                num_feats, hid_feats, kernel_size, dropout_rate, time_dim=1)

        if ff_type == 'conv1d-linear':
            return Conv1dLinear(
                num_feats, hid_feats, kernel_size, dropout_rate, time_dim=1)



    def forward(self, x, mask):
        """Compute encoded features.
        :param torch.Tensor x: encoded source features (batch, max_time_in, num_feats)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.norm_before:
            x = self.norm1(x)

        x_att = self.self_attn(x, x, x, mask)
        if self.concat_after:
            x = torch.cat((x, x_att), dim=-1)
            x = self.concat_linear(x)
        else:
            x = x_att

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        if not self.norm_before:
            x = self.norm1(x)

        residual = x
        if self.norm_before:
            x = self.norm2(x)

        x =self.feed_forward(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        if not self.norm_before:
            x = self.norm2(x)

        return x, mask
