"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d


class ETDNNBlock(nn.Module):
    """Building block for Extended-TDNN.

    Args:
      in_channels:   input channels.
      out_channels:  output channels.
      kernel_size:   kernels size for the convolution.
      dilation:      kernel dilation.
      activation:    non-linear activation function object, string or config dict.
      dropout_rate:  dropout rate.
      use_norm:      if True, if uses layer normalization.
      norm_layer:    Normalization Layer constructor, if None it used BatchNorm1d.
      norm_before:   if True, layer normalization is before the non-linearity, else
                     after the non-linearity.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        activation={"name": "relu", "inplace": True},
        dropout_rate=0,
        norm_layer=None,
        use_norm=True,
        norm_before=False,
    ):

        super().__init__()

        self.activation1 = AF.create(activation)
        self.activation2 = AF.create(activation)
        padding = int(dilation * (kernel_size - 1) / 2)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout1 = Dropout1d(dropout_rate)
            self.dropout2 = Dropout1d(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d

            self.bn1 = norm_layer(out_channels)
            self.bn2 = norm_layer(out_channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        bias = not self.norm_before
        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            bias=bias,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = Conv1d(out_channels, out_channels, bias=bias, kernel_size=1)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_time).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_time).
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.activation1(x)
        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        x = self.activation2(x)
        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout2(x)

        return x
