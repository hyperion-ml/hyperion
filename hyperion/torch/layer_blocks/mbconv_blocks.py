"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn

# from torch.nn import Conv2d, BatchNorm2d

from ..layers import ActivationFactory as AF
from ..layers import DropConnect2d
from .se_blocks import SEBlock2D, TSEBlock2D


def _conv1x1(in_channels, out_channels, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _dwconvkxk(channels, kernel_size=3, stride=1, bias=False):
    """kxk depth-wise convolution with padding"""
    return nn.Conv2d(
        channels,
        channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        groups=channels,
        bias=bias,
        padding_mode="zeros",
    )


def _make_downsample(in_channels, out_channels, stride, norm_layer):

    return nn.Sequential(
        _conv1x1(in_channels, out_channels, stride, bias=False),
        norm_layer(out_channels, momentum=0.01, eps=1e-3),
    )


class MBConvBlock(nn.Module):
    """MobileNet/EfficentNet Inverted bottleneck Block

    Attributes:
      in_channels:       input channels.
      out_channels:      output channels
      expansion:         expansion of channels for the inverted bottleneck.
      kernel_size:       kernel size of the convs.
      stride:            downsampling stride of the convs.
      activation:        Non-linear activation object, string of configuration dictionary.
      drop_connect_rate: Drop-connect rate for stochastic number of layers.
      norm_layer:        Normalization layer constructor, if None BatchNorm2d is used.
      se_r=None:         Squeeze-excitation compression ratio.
      time_se:           If true, squeeze is done only in time dimension.
      num_feats:         Number of features in dimension 2, needed if time_se=True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        stride=1,
        activation="swish",
        drop_connect_rate=0,
        norm_layer=None,
        se_r=None,
        time_se=False,
        num_feats=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.inner_channels = in_channels * expansion
        self.kernel_size = kernel_size
        self.act = AF.create(activation)
        self.stride = stride

        self.se_r = se_r
        self.has_se = se_r is not None and se_r > 1
        self.time_se = time_se
        self.num_feats = num_feats

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # expansion phase
        if self.expansion > 1:
            self.conv_exp = _conv1x1(in_channels, self.inner_channels)
            self.bn_exp = norm_layer(self.inner_channels, momentum=0.01, eps=1e-3)

        # depthwise conv phase
        self.conv_dw = _dwconvkxk(self.inner_channels, self.kernel_size, stride)
        self.bn_dw = norm_layer(self.inner_channels, momentum=0.01, eps=1e-3)

        # squeeze-excitation block
        if self.has_se:
            if time_se:
                self.se_layer = TSEBlock2D(
                    self.inner_channels,
                    (num_feats + stride - 1) // stride,
                    se_r,
                    activation,
                )
            else:
                self.se_layer = SEBlock2D(self.inner_channels, se_r, activation)

        # projection phase
        self.conv_proj = _conv1x1(self.inner_channels, out_channels)
        self.bn_proj = norm_layer(out_channels, momentum=0.01, eps=1e-3)
        self.drop_connect_rate = drop_connect_rate
        self.drop_connect = None
        if drop_connect_rate > 0:
            self.drop_connect = DropConnect2d(drop_connect_rate)

        # when input and output dimensions are different, we adapt the dimensions using conv1x1
        # this is different from official implementation where they remove the residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = _make_downsample(
                in_channels, out_channels, stride, norm_layer
            )

        self.context = stride * (kernel_size - 1) // 2
        self.downsample_factor = stride

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
        if self.expansion > 1:
            x = self.act(self.bn_exp(self.conv_exp(x)))

        x = self.act(self.bn_dw(self.conv_dw(x)))

        if self.has_se:
            x = self.se_layer(x)

        x = self.bn_proj(self.conv_proj(x))

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        return x


class MBConvInOutBlock(nn.Module):
    """Convolutional block used as input/output
        in MobileNet/EffcientNet

    Attributes:
      in_channels:       input channels.
      out_channels:      output channels
      kernel_size:       kernel size of the convs.
      stride:            downsampling stride of the convs.
      activation:        Non-linear activation object, string of configuration dictionary.
      norm_layer:        Normalization layer constructor, if None BatchNorm2d is used.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        activation="swish",
        norm_layer=None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode="zeros",
        )
        self.bn = norm_layer(out_channels, momentum=0.01, eps=1e-3)
        self.act = AF.create(activation)
        self.context = padding
        self.downsample_factor = stride

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        return self.act(self.bn(self.conv(x)))
