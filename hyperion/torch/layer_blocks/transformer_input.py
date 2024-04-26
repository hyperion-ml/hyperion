"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF


class TransformerConv2dSubsampler(nn.Module):
    """Convolutional 2D subsampling (to 1//stride length) Tor transformer

    Attributes:
      in_feats: input feature dimension
      out_feats: Transformer d_model
      hid_act: activation layer object
      stride: total stride of the subsampler
      pos_enc: positional encoder layer
      time_dim: indicates which is the time dimension in the input tensor
    """

    def __init__(
        self, in_feats, out_feats, hid_act, stride=4, pos_enc=None, time_dim=1
    ):
        super().__init__()
        self.time_dim = time_dim
        hid_act = AF.create(hid_act)
        self.stride = stride
        if stride == 4:
            stride_1 = 2
            stride_2 = 2
            hid_feats = out_feats * (((in_feats - 1) // 2 - 1) // 2)
        elif stride == 2:
            stride_1 = 2
            stride_2 = 1
            hid_feats = out_feats * ((in_feats - 1) // 2 - 2)
        elif stride == 1:
            stride_1 = 1
            stride_2 = 1
            hid_feats = out_feats * (in_feats - 4)
        else:
            raise NotImplementedError(
                "Valid TransformerConv2dSubsampler stride==1,2,4 !={stride}"
            )

        self.conv = nn.Sequential(
            nn.Conv2d(1, out_feats, 3, stride_1, padding=(0, 1)),
            hid_act,
            nn.Conv2d(out_feats, out_feats, 3, stride_2, padding=(0, 1)),
            hid_act,
        )

        linear = nn.Linear(hid_feats, out_feats)
        if pos_enc is None:
            self.out = linear
        else:
            self.out = nn.Sequential(linear, pos_enc)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with size=(batch, time, in_feats)
          x_mask: mask to indicate valid time steps for x (batch, time1, time2)

        Returns:
           Tensor with output features with shape = (batch, time//stride, out_feats)
           Tensor with subsampled mask // stride.
        """
        if self.time_dim == 1:
            x = x.transpose(1, 2)

        x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = self.out(x.contiguous().view(b, c * f, t).transpose(1, 2))
        if x_mask is None:
            return x, None

        return x, x_mask[:, :, :: self.stride]


class TransformerConv1dSubsampler(nn.Module):
    """Convolutional 1D subsampling (to 1//stride length) Tor transformer

    Attributes:
      in_feats: input feature dimension
      out_feats: Transformer d_model
      hid_act: activation layer object
      stride: total stride of the subsampler
      pos_enc: positional encoder layer
      time_dim: indicates which is the time dimension in the input tensor
    """

    def __init__(
        self, in_feats, out_feats, hid_act, stride=4, pos_enc=None, time_dim=1
    ):
        super().__init__()
        self.time_dim = time_dim
        hid_act = AF.create(hid_act)
        self.stride = stride
        if stride == 4:
            stride_1 = 2
            stride_2 = 2
        elif stride == 2:
            stride_1 = 2
            stride_2 = 1
        elif stride == 1:
            stride_1 = 1
            stride_2 = 1
        else:
            raise NotImplementedError(
                "Valid TransformerConv1dSubsampler stride==1,2,4 !={stride}"
            )

        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, out_feats, 3, stride_1, padding=1),
            hid_act,
            nn.Conv1d(out_feats, out_feats, 3, stride_2, padding=1),
            hid_act,
        )

        linear = nn.Linear(out_feats, out_feats)
        if pos_enc is None:
            self.out = linear
        else:
            self.out = nn.Sequential(linear, pos_enc)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with size=(batch, time, in_feats)
          x_mask: mask to indicate valid time steps for x (batch, time1, time2)

        Returns:
           Tensor with output features with shape = (batch, time//stride, out_feats)
           Tensor with subsampled mask // stride.
        """
        if self.time_dim == 1:
            x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.out(x.transpose(1, 2))
        if x_mask is None:
            return x, None

        return x, x_mask[:, :, :: self.stride]
