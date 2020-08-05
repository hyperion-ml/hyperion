"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

from ..layers import PosEncoder
from ..layer_blocks import TransformerEncoderBlockV1 as EBlock
from ..layer_blocks import TransformerConv2dSubsampler as Conv2dSubsampler
from .net_arch import NetArch

class TransformerEncoderV1(NetArch):
    """Transformer encoder module.

    Attributes:
      in_feats: input features dimension
      d_model: encoder blocks feature dimension
      num_heads: number of heads
      num_blocks: number of self attn blocks
      att_type: string in ['scaled-dot-prod-att-v1', 'local-scaled-dot-prod-att-v1']
      att_context: maximum context range for local attention
      ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
      d_ff: dimension of middle layer in feed_forward block
      ff_kernel_size: kernel size for convolutional versions of ff block
      ff_dropout_rate: dropout rate for ff block
      pos_dropout_rate: dropout rate for positional encoder
      att_dropout_rate: dropout rate for attention block
      in_layer_type: input layer block type in ['linear','conv2d-sub', 'embed', None]
      norm_before: if True, use layer norm before layers, otherwise after
      concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)
      padding_idx: padding idx for embed layer
      in_time_dim: time dimension in the input Tensor
      out_time_dim: dimension that we want to be time in the output tensor

    """

    def __init__(self, in_feats,
                 d_model=256,
                 num_heads=4,
                 num_blocks=6,
                 att_type = 'scaled-dot-prod-v1',
                 att_context = 25,
                 ff_type='linear',
                 d_ff=2048,
                 ff_kernel_size=1,
                 ff_dropout_rate=0.1,
                 pos_dropout_rate=0.1,
                 att_dropout_rate=0.0,
                 in_layer_type='conv2d-sub',
                 norm_before=True,
                 concat_after=False,
                 padding_idx=-1, in_time_dim=-1, out_time_dim=1):

        super(TransformerEncoderV1, self).__init__()
        self.in_feats = in_feats
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.att_type = att_type
        self.att_context = att_context

        self.ff_type = ff_type
        self.d_ff = d_ff
        self.ff_kernel_size = ff_kernel_size
        self.ff_dropout_rate = ff_dropout_rate
        self.att_dropout_rate = att_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.in_layer_type = in_layer_type
        self.norm_before = norm_before
        self.concat_after = concat_after
        self.padding_idx = padding_idx
        self.in_time_dim = in_time_dim
        self.out_time_dim = out_time_dim

        self._make_in_layer(in_layer_type, in_feats, d_model, 
                            ff_dropout_rate, pos_dropout_rate, 
                            padding_idx, in_time_dim)

        blocks = []
        for i in range(num_blocks):
            blocks.append(EBlock(
                d_model, att_type, num_heads, 
                ff_type, d_ff, ff_kernel_size, ff_dropout_rate=ff_dropout_rate,
                att_context=att_context, att_dropout_rate=att_dropout_rate, 
                norm_before=norm_before, concat_after=concat_after))

        self.blocks = nn.ModuleList(blocks)
        
        if self.norm_before:
            self.norm = nn.LayerNorm(d_model)


    def _make_in_layer(self, in_layer_type, in_feats, d_model, 
                       dropout_rate, pos_dropout_rate, padding_idx, time_dim):

        if in_layer_type == "linear":
            self.in_layer = nn.Sequential(
                nn.Linear(in_feats, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout_rate),
                nn.ReLU6(),
                PosEncoder(d_model, pos_dropout_rate)
            )
        elif in_layer_type == "conv2d-sub":
            self.in_layer = Conv2dSubsampler(
                in_feats, d_model, pos_dropout_rate, time_dim=time_dim)
        elif in_layer_type == "embed":
            self.in_layer = nn.Sequential(
                nn.Embedding(in_feats, d_model, padding_idx=padding_idx),
                PosEncoder(d_model, pos_dropout_rate)
            )
        elif isinstance(in_layer_type, nn.Module):
            self.in_layer = nn.Sequential(
                in_layer_type,
                PosEncoder(d_model, pos_dropout_rate),
            )
        elif in_layer_type is None:
            self.in_layer = nn.Sequential(
                PosEncoder(d_model, pos_dropout_rate)
            )
        else:
            raise ValueError("unknown in_layer_type: " + in_layer_type)


    def forward(self, x, mask=None):
        """Forward pass function

        Args:
          x: input tensor with size=(batch, time, num_feats)
          mask: mask to indicate valid time steps for x (batch, time)

        Returns:
           Tensor with output features
           Tensor with mask
        """
        if isinstance(self.in_layer, Conv2dSubsampler):
            x, mask = self.in_layer(x, mask)
        else:
            if self.in_time_dim != 1:
                x = x.transpose(1, self.in_time_dim).contiguous()
            x = self.in_layer(x)

        for i in range(len(self.blocks)):
            x, mask = self.blocks[i](x, mask)

        if self.norm_before:
            x = self.norm(x)

        if self.out_time_dim != 1:
            x = x.transpose(1, self.out_time_dim)

        if mask is None:
            return x
        return x, mask


    def get_config(self):
        """ Gets network config
        Returns:
           dictionary with config params
        """
        config = {'in_feats': self.in_feats,
                  'd_model': self.d_model,
                  'num_heads': self.num_heads,
                  'num_blocks': self.num_blocks,
                  'att_type': self.att_type,
                  'att_context': self.att_context,
                  'ff_type': self.ff_type,
                  'd_ff': self.d_ff,
                  'ff_kernel_size': self.ff_kernel_size,
                  'ff_dropout_rate': self.ff_dropout_rate,
                  'att_dropout_rate': self.att_dropout_rate,
                  'pos_dropout_rate': self.pos_dropout_rate,
                  'in_layer_type': self.in_layer_type,
                  'norm_before': self.norm_before,
                  'concat_after': self.concat_after,
                  'padding_idx': self.padding_idx,
                  'in_time_dim': self.in_time_dim,
                  'out_time_dim': self.out_time_dim }
        
        base_config = super(TransformerEncoderV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def in_context(self):
        return (self.att_context, self.att_context)

    def in_shape(self):
        """Input shape for network
        
        Returns:
           Tuple describing input shape
        """
        if self.in_time_dim == 1:
            return (None, None, self.in_feats)
        else:
            return (None, self.in_feats, None)


    def out_shape(self, in_shape=None):
        """Infers the network output shape given the input shape

        Args:
          in_shape: input shape tuple

        Returns:
          Tuple with the output shape
        """
        if in_shape is None:
            out_t = None
        else:
            assert len(in_shape) == 3
            in_t = in_shape[self.in_time_dim]
            if in_t is None:
                out_t = None
            else:
                if isinstance(self.in_layer, Conv2dSubsampler):
                    #out_t = in_t//4
                    out_t = ((in_t - 1)//2 - 1)//2
                else:
                    out_t = in_t

        if self.out_time_dim == 1:
            return (out_t, out_t, self.d_model)
        else:
            return (out_t, self.d_model, out_t)


        
