"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import numpy as np

import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d
from .etdnn_blocks import ETDNNBlock


class ResETDNNBlock(ETDNNBlock):
    """Building block for Residual Extended-TDNN.

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
        num_channels,
        kernel_size,
        dilation=1,
        activation={"name": "relu", "inplace": True},
        dropout_rate=0,
        norm_layer=None,
        use_norm=True,
        norm_before=False,
    ):

        super().__init__(
            num_channels,
            num_channels,
            kernel_size,
            dilation,
            activation,
            dropout_rate,
            norm_layer,
            use_norm,
            norm_before,
        )

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_time).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_time).
        """

        residual = x
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

        x += residual
        x = self.activation2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout2(x)

        return x
