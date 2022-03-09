"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d, Dropout2d

from ..layers import ActivationFactory as AF
from .se_blocks import SEBlock2D, TSEBlock2D
from .resnet_blocks import ResNetBasicBlock, ResNetBNBlock


class SEResNetBasicBlock(ResNetBasicBlock):
    """Squeeze-excitation ResNet basic Block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r:              squeeze-excitation compression ratio.
      time_se:           If true, squeeze is done only in time dimension.
      num_feats:         Number of features in dimension 2, needed if time_se=True.
    """

    def __init__(
        self,
        in_channels,
        channels,
        activation={"name": "relu", "inplace": True},
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        norm_layer=None,
        norm_before=True,
        se_r=16,
        time_se=False,
        num_feats=None,
    ):

        super().__init__(
            in_channels,
            channels,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        if time_se:
            self.se_layer = TSEBlock2D(channels, num_feats, se_r, activation)
        else:
            self.se_layer = SEBlock2D(channels, se_r, activation)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if not self.norm_before:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.se_layer(x)
        x += residual
        x = self.act2(x)

        if not self.norm_before:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNetBNBlock(ResNetBNBlock):
    """Squeeze-excitation ResNet bottleneck Block.

    Attributes:
      in_channels:       input channels.
      channels:          channels in bottleneck layer when width_factor=1.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r=None:         squeeze-excitation compression ratio.
      time_se:           If true, squeeze is done only in time dimension.
      num_feats:         Number of features in dimension 2, needed if time_se=True.
    """

    def __init__(
        self,
        in_channels,
        channels,
        activation={"name": "relu", "inplace": True},
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        norm_layer=None,
        norm_before=True,
        se_r=16,
        time_se=False,
        num_feats=None,
    ):

        super().__init__(
            in_channels,
            channels,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        if time_se:
            self.se_layer = TSEBlock2D(
                channels * self.expansion, num_feats, se_r, activation
            )
        else:
            self.se_layer = SEBlock2D(channels * self.expansion, se_r, activation)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.
        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)
        if not self.norm_before:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)
        x = self.act2(x)
        if not self.norm_before:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.se_layer(x)
        x += residual
        x = self.act3(x)

        if not self.norm_before:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
