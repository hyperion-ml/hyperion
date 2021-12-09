"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
import torch
import torch.nn as nn
from torch.nn import Conv1d, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d, DropConnect1d
from .se_blocks import SEBlock1d


def _convk(
    in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1, bias=False
):
    """kernel k convolution with padding"""
    padding = dilation * (kernel_size - 1) // 2
    return Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def _conv1(in_channels, out_channels, stride=1, bias=False):
    """point-wise convolution"""
    return Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _make_downsample(in_channels, out_channels, stride, norm_layer, norm_before):

    if norm_before:
        return nn.Sequential(
            _conv1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _conv1(in_channels, out_channels, stride, bias=True)


class Res2Net1dBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation={"name": "relu6", "inplace": True},
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        width_factor=1,
        scale=4,
        groups=1,
        dilation=1,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
        se_r=None,
    ):

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        width_in = in_channels // scale
        width_mid = int(width_factor * channels) // scale
        self.width_in = width_in
        self.has_proj1 = width_in != width_mid
        self.scale = scale
        channels_mid = width_mid * scale
        if scale == 1:
            self.num_k = 1
        else:
            self.num_k = scale - 1

        if scale > 1:
            single_width = in_channels % width_in + width_in
            self.conv1 = _conv1(single_width, width_mid, stride, bias=bias)

        conv1s = []
        proj1s = []
        bn1s = []
        for i in range(self.num_k):
            conv1s.append(
                _convk(
                    width_in,
                    width_mid,
                    kernel_size,
                    stride,
                    groups,
                    dilation,
                    bias=bias,
                )
            )
            if use_norm:
                bn1s.append(norm_layer(width_mid))
            if self.has_proj1 and i < self.num_k - 1:
                proj1s.append(_conv1(width_mid, width_in, bias=False))

        self.conv1s = nn.ModuleList(conv1s)
        self.bn1s = nn.ModuleList(bn1s)
        if self.has_proj1:
            self.proj1s = nn.ModuleList(proj1s)

        self.conv2 = _convk(
            channels_mid, channels, kernel_size, groups=groups, bias=bias
        )
        if use_norm:
            self.bn2 = norm_layer(channels)
        self.act1 = AF.create(activation)
        self.act2 = AF.create(activation)
        self.stride = stride

        self.downsample = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = _make_downsample(
                in_channels, channels * self.expansion, stride, norm_layer, norm_before
            )

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

        self.drop_connect_rate = drop_connect_rate
        self.drop_connect = None
        if drop_connect_rate > 0:
            self.drop_connect = DropConnect1d(drop_connect_rate)

        self.context = (dilation + stride) * (kernel_size - 1) // 2
        self.downsample_factor = stride

        if se_r is not None:
            self.se_layer = SEBlock1d(channels, se_r, activation)
        else:
            self.se_layer = None

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        residual = x
        split_size = [self.width_in for i in range(self.scale - 1)]
        split_size.append(self.in_channels % self.width_in + self.width_in)
        split_x = torch.split(x, split_size, 1)

        x = []
        for i in range(self.num_k):
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
                x_i = self.bn1(x_i)
            x.append(x_i)

        if self.scale > 1:
            x.append(self.conv1(split_x[-1]))

        x = torch.cat(x, dim=1)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        if self.se_layer:
            x = self.se_layer(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act2(x)

        if not self.norm_before:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class Res2Net1dBNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation={"name": "relu6", "inplace": True},
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        width_factor=1,
        scale=4,
        groups=1,
        dilation=1,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
        se_r=None,
        num_feats=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        bias = not norm_before

        width = int(width_factor * channels) // scale
        self.width = width
        self.scale = scale
        channels_bn = width * scale
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1(in_channels, channels_bn, bias=bias)
        if use_norm:
            self.bn1 = norm_layer(channels_bn)

        if scale == 1:
            self.num_k = 1
        else:
            self.num_k = scale - 1

        if stride > 1 and scale > 1:
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)

        conv2s = []
        bn2s = []
        for i in range(self.num_k):
            conv2s.append(
                _convk(width, width, kernel_size, stride, groups, dilation, bias=bias)
            )
            if use_norm:
                bn2s.append(norm_layer(width))

        self.conv2s = nn.ModuleList(conv2s)
        if use_norm:
            self.bn2s = nn.ModuleList(bn2s)

        self.conv3 = _conv1(channels_bn, channels, bias=bias)
        if use_norm:
            self.bn3 = norm_layer(channels)
        self.act1 = AF.create(activation)
        self.act2 = AF.create(activation)
        self.act3 = AF.create(activation)
        self.stride = stride

        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = _make_downsample(
                in_channels, channels, stride, norm_layer, norm_before
            )

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

        self.drop_connect_rate = drop_connect_rate
        self.drop_connect = None
        if drop_connect_rate > 0:
            self.drop_connect = DropConnect1d(drop_connect_rate)

        self.context = dilation * (kernel_size - 1) // 2
        self.downsample_factor = stride

        if se_r is not None:
            self.se_layer = SEBlock1d(channels, se_r, activation)
        else:
            self.se_layer = None

    @property
    def out_channels(self):
        return self.channels

    @property
    def expansion(self):
        return self.channels / self.width / self.scale

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)
        if not self.norm_before:
            x = self.bn1(x)

        split_x = torch.split(x, self.width, 1)
        x = []
        for i in range(self.num_k):
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

        if self.se_layer:
            x = self.se_layer(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        if not self.norm_before:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
