"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Dropout2d
import torch.nn.functional as nnf

from ..layers import ActivationFactory as AF


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


class ResNetInputBlock(nn.Module):
    """Input block for ResNet architecture

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size for conv
      stride: stride for conv
      activation: str/dict indicationg activation type and arguments
      norm_layer: norm_layer object constructor, if None it uses BatchNorm2d
      norm_before: if True it applies the norm_layer before the activation,
                   if False, after the activation
      do_maxpool: apply maxpooling 2x2 at the output

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        activation={"name": "relu", "inplace": True},
        norm_layer=None,
        norm_before=True,
        do_maxpool=True,
    ):

        super().__init__()

        padding = int((kernel_size - 1) / 2)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        bias = not norm_before
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = norm_layer(out_channels)
        self.act = AF.create(activation)
        self.norm_before = norm_before
        self.do_maxpool = do_maxpool

        self.context = int((kernel_size - 1) / 2)
        self.downsample_factor = stride

        if do_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.downsample_factor *= 2

    def forward(self, x):

        x = self.conv(x)
        if self.norm_before:
            x = self.bn(x)

        x = self.act(x)
        if not self.norm_before:
            x = self.bn(x)

        if self.do_maxpool:
            x = self.maxpool(x)

        return x


class ResNetBasicBlock(nn.Module):
    expansion = 1

    # __constants__ = ['downsample']

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
    ):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _conv3x3(
            in_channels, channels, stride, groups, dilation, bias=bias
        )
        self.bn1 = norm_layer(channels)
        self.act1 = AF.create(activation)
        self.conv2 = _conv3x3(channels, channels, groups=groups, bias=bias)
        self.bn2 = norm_layer(channels)
        self.act2 = AF.create(activation)
        self.stride = stride
        self.norm_before = norm_before

        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = _make_downsample(
                in_channels, channels, stride, norm_layer, norm_before
            )

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout2d(dropout_rate)

        self.context = dilation + stride
        self.downsample_factor = stride

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
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

        x += residual
        x = self.act2(x)

        if not self.norm_before:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNetBNBlock(nn.Module):
    expansion = 4
    # __constants__ = ['downsample']

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
    ):

        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bias = not norm_before

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(in_channels, channels, bias=bias)
        self.bn1 = norm_layer(channels)
        self.conv2 = _conv3x3(channels, channels, stride, groups, dilation, bias=bias)
        self.bn2 = norm_layer(channels)
        self.conv3 = _conv1x1(channels, channels * self.expansion, bias=bias)
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

    @property
    def out_channels(self):
        return self.channels * self.expansion

    def forward(self, x):
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

        x += residual
        x = self.act3(x)

        if not self.norm_before:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.interp = nnf.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ResNetEndpointBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        activation={"name": "relu", "inplace": True},
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        bias = not norm_before
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.norm_before = norm_before

        if self.in_channels != self.out_channels:
            self.conv = _conv1x1(in_channels, out_channels, bias=bias)
            self.bn = norm_layer(out_channels)
            self.act = AF.create(activation)

        self.scale = scale
        if self.scale > 1:
            self.upsample = Interpolate(scale_factor=scale, mode="nearest")

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.conv(x)
            if self.norm_before:
                x = self.bn(x)

            x = self.act(x)
            if not self.norm_before:
                x = self.bn(x)

        if self.scale > 1:
            x = self.upsample(x)

        return x
