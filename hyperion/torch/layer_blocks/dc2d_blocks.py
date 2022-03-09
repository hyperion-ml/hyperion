"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Dropout2d

from ..layers import ActivationFactory as AF
from ..layers.subpixel_convs import SubPixelConv2d


class DC2dEncBlock(nn.Module):
    """Build block for deep convolutional encoder 2d.

    Args:
      in_channels:   input channels.
      out_channels:  output channels.
      kernel_size:   kernels size for the convolution.
      stride:        downsampling stride.
      dilation:      kernel dilation.
      activation:    non-linear activation function object, string or config dict.
      dropout_rate:  dropout rate.
      use_norm:      if True, if uses layer normalization.
      norm_layer:    Normalization Layer constructor, if None it used BatchNorm2d.
      norm_before:   if True, layer normalization is before the non-linearity, else
                     after the non-linearity.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation="relu",
        dropout_rate=0,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.activation = AF.create(activation)
        padding = int(dilation * (kernel_size - 1) / 2)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout2d(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d

            self.bn1 = norm_layer(out_channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            bias=(not self.norm_before),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

        self.stride = stride
        self.context = dilation * (kernel_size - 1) // 2

    def freeze(self):
        """Freezes trainable parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreezes trainable parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class DC2dDecBlock(nn.Module):
    """Build block for deep convolutional decoder 2d.

    Args:
      in_channels:   input channels.
      out_channels:  output channels.
      kernel_size:   kernels size for the convolution.
      stride:        upsampling stride.
      dilation:      kernel dilation.
      activation:    non-linear activation function object, string or config dict.
      dropout_rate:  dropout rate.
      use_norm:      if True, if uses layer normalization.
      norm_layer:    Normalization Layer constructor, if None it used BatchNorm2d.
      norm_before:   if True, layer normalization is before the non-linearity, else
                     after the non-linearity.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation="relu",
        dropout_rate=0,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.activation = AF.create(activation)
        padding = int(dilation * (kernel_size - 1) / 2)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout2d(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d

            self.bn1 = norm_layer(out_channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        if stride == 1:
            self.conv1 = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                bias=(not self.norm_before),
                padding=padding,
            )  # pytorch > 1.0
        else:
            self.conv1 = SubPixelConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=(not self.norm_before),
                padding=padding,
            )

        self.stride = stride
        self.context = dilation * (kernel_size - 1) // 2

    def freeze(self):
        """Freezes trainable parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreezes trainable parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
