"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Dropout2d

from ..layers import ActivationFactory as AF
from .se_blocks import SEBlock2D, TSEBlock2D


def _conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def _conv1x1(in_channels, out_channels, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _make_downsample(in_channels, out_channels, stride, norm_layer, norm_before):

    if norm_before:
        return nn.Sequential(
            _conv1x1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _conv1x1(in_channels, out_channels, stride, bias=True)


class Res2NetBasicBlock(nn.Module):
    """Res2Net basic Block. This is a modified Res2Net block with
    two 3x3 convolutions, instead of the standard bottleneck block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.

      dropout_rate:      dropout rate.
      width_factor:      multiplication factor for the number of channels in the first layer
                         or the block.
      scale:             scale parameter of the Res2Net.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r:              squeeze-excitation compression ratio.
      time_se:           If true, squeeze is done only in time dimension.
      num_feats:         Number of features in dimension 2, needed if time_se=True.
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        activation={"name": "relu", "inplace": True},
        stride=1,
        dropout_rate=0,
        width_factor=1,
        scale=4,
        groups=1,
        dilation=1,
        norm_layer=None,
        norm_before=True,
        se_r=None,
        time_se=False,
        num_feats=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bias = not norm_before

        width_in = in_channels // scale
        width_mid = int(width_factor * channels) // scale
        self.width_in = width_in
        self.has_proj1 = width_in != width_mid and stride == 1
        self.scale = scale
        channels_mid = width_mid * scale
        if scale == 1:
            self.num_3x3 = 1
        else:
            self.num_3x3 = scale - 1

        if scale > 1:
            single_width = in_channels % width_in + width_in
            self.conv1x1 = _conv1x1(single_width, width_mid, stride, bias=bias)

        conv1s = []
        proj1s = []
        bn1s = []
        for i in range(self.num_3x3):
            conv1s.append(
                _conv3x3(width_in, width_mid, stride, groups, dilation, bias=bias)
            )
            bn1s.append(norm_layer(width_mid))
            if self.has_proj1 and i < self.num_3x3 - 1:
                proj1s.append(_conv1x1(width_mid, width_in, bias=False))

        self.conv1s = nn.ModuleList(conv1s)
        self.bn1s = nn.ModuleList(bn1s)
        if self.has_proj1:
            self.proj1s = nn.ModuleList(proj1s)

        self.conv2 = _conv3x3(channels_mid, channels, groups=groups, bias=bias)
        self.bn2 = norm_layer(channels)
        self.act1 = AF.create(activation)
        self.act2 = AF.create(activation)
        self.stride = stride

        self.norm_before = norm_before

        self.downsample = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = _make_downsample(
                in_channels, channels * self.expansion, stride, norm_layer, norm_before
            )

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout2d(dropout_rate)

        self.context = dilation
        self.downsample_factor = stride

        if se_r is not None:
            if time_se:
                self.se_layer = TSEBlock2D(channels, num_feats, se_r, activation)
            else:
                self.se_layer = SEBlock2D(channels, se_r, activation)
        else:
            self.se_layer = None

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        residual = x
        split_size = [self.width_in for i in range(self.scale - 1)]
        split_size.append(self.in_channels % self.width_in + self.width_in)
        split_x = torch.split(x, split_size, 1)

        x = []
        for i in range(self.num_3x3):
            if i == 0 or self.stride > 1:
                x_i = split_x[i]
            else:
                if self.has_proj1:
                    x_i = self.proj1s[i - 1](x_i)

                x_i = x_i + split_x[i]

            x_i = self.conv1s[i](x_i)
            if self.norm_before:
                x_i = self.bn1s[i](x_i)
            x_i = self.act1(x_i)
            if not self.norm_before:
                x_i = self.bn1s[i](x_i)
            x.append(x_i)

        if self.scale > 1:
            x.append(self.conv1x1(split_x[-1]))

        x = torch.cat(x, dim=1)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.se_layer:
            x = self.se_layer(x, x_mask=x_mask)

        x += residual
        x = self.act2(x)

        if not self.norm_before:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class Res2NetBNBlock(nn.Module):
    """Res2Net bottleneck Block.

    Attributes:
      in_channels:       input channels.
      channels:          channels in bottleneck layer when width_factor=1.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      width_factor:      multiplication factor for the number of channels in the bottleneck.
      scale:             scale parameter of the Res2Net.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r:              squeeze-excitation compression ratio.
      time_se:           If true, squeeze is done only in time dimension.
      num_feats:         Number of features in dimension 2, needed if time_se=True.
    """

    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        activation={"name": "relu", "inplace": True},
        stride=1,
        dropout_rate=0,
        width_factor=1,
        scale=4,
        groups=1,
        dilation=1,
        norm_layer=None,
        norm_before=True,
        se_r=None,
        time_se=False,
        num_feats=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bias = not norm_before

        width = int(width_factor * channels) // scale
        self.width = width
        self.scale = scale
        channels_bn = width * scale
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(in_channels, channels_bn, bias=bias)
        self.bn1 = norm_layer(channels_bn)

        if scale == 1:
            self.num_3x3 = 1
        else:
            self.num_3x3 = scale - 1

        if stride > 1 and scale > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        conv2s = []
        bn2s = []
        for i in range(self.num_3x3):
            conv2s.append(_conv3x3(width, width, stride, groups, dilation, bias=bias))
            bn2s.append(norm_layer(width))

        self.conv2s = nn.ModuleList(conv2s)
        self.bn2s = nn.ModuleList(bn2s)

        self.conv3 = _conv1x1(channels_bn, channels * self.expansion, bias=bias)
        self.bn3 = norm_layer(channels * self.expansion)
        self.act1 = AF.create(activation)
        self.act2 = AF.create(activation)
        self.act3 = AF.create(activation)
        self.stride = stride

        self.norm_before = norm_before

        self.downsample = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = _make_downsample(
                in_channels, channels * self.expansion, stride, norm_layer, norm_before
            )

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout2d(dropout_rate)

        self.context = dilation
        self.downsample_factor = stride

        if se_r is not None:
            if time_se:
                self.se_layer = TSEBlock2D(
                    channels * self.expansion, num_feats, se_r, activation
                )
            else:
                self.se_layer = SEBlock2D(channels * self.expansion, se_r, activation)
        else:
            self.se_layer = None

    @property
    def out_channels(self):
        return self.channels * self.expansion

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

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

        split_x = torch.split(x, self.width, 1)
        x = []
        for i in range(self.num_3x3):
            if i == 0 or self.stride > 1:
                x_i = split_x[i]
            else:
                x_i = x_i + split_x[i]
            x_i = self.conv2s[i](x_i)
            if self.norm_before:
                x_i = self.bn2s[i](x_i)
            x_i = self.act2(x_i)
            if not self.norm_before:
                x_i = self.bn2(x_i)
            x.append(x_i)

        if self.scale > 1:
            if self.stride == 1:
                x.append(split_x[-1])
            else:
                x.append(self.pool(split_x[-1]))

        x = torch.cat(x, dim=1)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.se_layer:
            x = self.se_layer(x, x_mask=x_mask)

        x += residual
        x = self.act3(x)

        if not self.norm_before:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
