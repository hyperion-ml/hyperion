"""
 Copyright 2020 Magdalena Rybicka
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Dropout2d
import torch.nn.functional as nnf

from ..layers.subpixel_convs import SubPixelConv2d
from ..layers import ActivationFactory as AF

import logging


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.interp = nnf.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


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


def _subpixel_conv1x1(in_channels, out_channels, stride=1, bias=False):
    """point-wise subpixel convolution"""
    return SubPixelConv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
    )


def _make_downsample(in_channels, out_channels, stride, norm_layer, norm_before):

    if norm_before:
        return nn.Sequential(
            _conv3x3(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _conv3x3(in_channels, out_channels, stride, bias=True)


def _make_upsample(in_channels, out_channels, stride, norm_layer, norm_before):
    if norm_before:
        return nn.Sequential(
            _subpixel_conv1x1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _subpixel_conv1x1(in_channels, out_channels, stride, bias=True)


def _make_resample(
    channels, scale, norm_layer, norm_before, activation, upsampling_type="nearest"
):
    resample_block = nn.ModuleList([])
    if scale > 1:
        if upsampling_type == "subpixel":
            resample_block.append(
                _make_upsample(channels, channels, scale, norm_layer, norm_before)
            )
            resample_block.append(AF.create(activation))
        elif upsampling_type == "bilinear":
            resample_block.append(Interpolate(scale_factor=scale, mode="bilinear"))
        else:
            resample_block.append(Interpolate(scale_factor=scale, mode="nearest"))

    elif scale < 1:
        resample_block.append(
            _make_downsample(channels, channels, 2, norm_layer, norm_before)
        )
        resample_block.append(AF.create(activation))

        if scale < 0.5:
            new_kernel_size = 3 if scale >= 0.25 else 5
            resample_block.append(
                nn.MaxPool2d(
                    kernel_size=new_kernel_size,
                    stride=int(0.5 / scale),
                    padding=new_kernel_size // 2,
                )
            )
    return resample_block


class SpineConv(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        activation={"name": "relu", "inplace": True},
        norm_layer=None,
        norm_before=True,
    ):
        """
        Class that connects the ouputs of the SpineNet to the rest of the network
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.channels = channels
        self.norm_before = norm_before
        bias = not norm_before
        self.conv1 = _conv1x1(in_channels, channels, stride, bias=bias)
        self.bn1 = norm_layer(channels)
        self.act1 = AF.create(activation)

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)
        return x


class BlockSpec(object):
    """A container class that specifies the block configuration for SpineNet."""

    def __init__(self, level, block_fn, input_offsets, is_output):
        self.level = level
        self.block_fn = block_fn
        self.input_offsets = input_offsets
        self.is_output = is_output

    @staticmethod
    def build_block_specs(block_specs=None):
        """Builds the list of BlockSpec objects for SpineNet."""
        return [BlockSpec(*b) for b in block_specs]


class SpineEndpoints(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        level,
        target_level,
        upsampling_type="nearest",
        stride=1,
        activation={"name": "relu", "inplace": True},
        norm_layer=None,
        norm_before=True,
        do_endpoint_conv=True,
    ):
        """
        Class that connects the ouputs of the SpineNet to the rest of the network
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.channels = channels
        self.norm_before = norm_before
        self.scale = 2 ** (level - target_level)
        self.do_endpoint_conv = do_endpoint_conv
        self.upsampling_type = upsampling_type
        bias = not norm_before
        if self.do_endpoint_conv and in_channels != channels:
            # in some cases this convolution is not necessary
            self.conv1 = _conv1x1(in_channels, channels, stride, bias=bias)
            self.bn1 = norm_layer(channels)
            self.act1 = AF.create(activation)

        else:
            self.channels = in_channels

        self.resample = _make_resample(
            channels,
            self.scale,
            norm_layer,
            norm_before,
            activation,
            upsampling_type=upsampling_type,
        )

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        if self.do_endpoint_conv and self.in_channels != self.channels:
            x = self.conv1(x)
            if self.norm_before:
                x = self.bn1(x)
            x = self.act1(x)
        for mod in self.resample:
            x = mod(x)
        return x


class SpineResample(nn.Module):
    def __init__(
        self,
        spec,
        in_channels,
        out_channels,
        scale,
        alpha,
        upsampling_type="nearest",
        activation={"name": "relu", "inplace": True},
        norm_layer=None,
        norm_before=True,
    ):
        """
        Class that build a resampling connection between single SpineNet blocks.
        """
        super().__init__()
        self.spec = spec

        in_channels_alpha = int(in_channels * alpha)
        in_channels = in_channels * spec.block_fn.expansion
        self.scale = 2 ** scale
        bias = not norm_before
        self.norm_before = norm_before
        if norm_layer is None:
            norm_layer = BatchNorm2d

        self.conv1 = _conv1x1(in_channels, in_channels_alpha, bias=bias)
        self.bn1 = norm_layer(in_channels_alpha)
        self.act1 = AF.create(activation)

        self.resample = _make_resample(
            in_channels_alpha,
            self.scale,
            norm_layer,
            norm_before,
            activation,
            upsampling_type=upsampling_type,
        )

        self.conv2 = _conv1x1(in_channels_alpha, out_channels, bias=bias)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)

        for mod in self.resample:
            x = mod(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)
        return x
