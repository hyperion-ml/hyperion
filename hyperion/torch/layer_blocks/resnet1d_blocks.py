"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch.nn as nn
from torch.nn import Conv1d, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d, DropConnect1d, Interpolate
from ..layers.subpixel_convs import SubPixelConv1d
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


def _subpixel_conv1(in_channels, out_channels, stride=1, bias=False):
    """point-wise subpixel convolution"""
    return SubPixelConv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
    )


def _subpixel_convk(
    in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1, bias=False
):
    """kernel k subpixel convolution with padding"""
    padding = dilation * (kernel_size - 1) // 2
    return SubPixelConv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def _make_downsample(in_channels, out_channels, stride, norm_layer, norm_before):

    if stride % 2 == 0:
        first_stride = 2
        second_stride = stride // 2
    else:
        first_stride = 1
        second_stride = stride

    if norm_before:
        layers = [
            _conv1(in_channels, out_channels, first_stride, bias=False),
            norm_layer(out_channels),
        ]
    else:
        layers = [_conv1(in_channels, out_channels, first_stride, bias=True)]

    if second_stride > 1:
        kernel_size = 2 * (second_stride // 2) + 1
        layers.append(
            nn.MaxPool1d(
                kernel_size=kernel_size,
                stride=second_stride,
                padding=(kernel_size - 1) // 2,
            )
        )

    return nn.Sequential(*layers)


def _make_upsample(
    in_channels, out_channels, stride, norm_layer, norm_before, mode="nearest"
):

    if mode == "subpixel":
        if norm_before:
            return nn.Sequential(
                _subpixel_conv1(in_channels, out_channels, stride, bias=False),
                norm_layer(out_channels),
            )

        return _subpixel_conv1(in_channels, out_channels, stride, bias=True)

    if norm_before:
        layers = [
            _conv1(in_channels, out_channels, stride=1, bias=False),
            norm_layer(out_channels),
        ]
    else:
        layers = [_conv1(in_channels, out_channels, stride=1, bias=True)]

    layers.append(Interpolate(scale_factor=stride, mode=mode))
    return nn.Sequential(*layers)


class ResNet1dBasicBlock(nn.Module):
    expansion = 1

    # __constants__ = ['downsample']

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            self.bn1 = norm_layer(channels)
            self.bn2 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _convk(
            in_channels, channels, kernel_size, stride, groups, dilation, bias=bias
        )
        self.act1 = AF.create(activation)
        self.conv2 = _convk(channels, channels, kernel_size, groups=groups, bias=bias)

        self.act2 = AF.create(activation)
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

        self.context = (stride + dilation) * (kernel_size - 1) // 2
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

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet1dBasicDecBlock(nn.Module):
    expansion = 1

    # __constants__ = ['downsample']

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            self.bn1 = norm_layer(channels)
            self.bn2 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _subpixel_convk(
            in_channels, channels, kernel_size, stride, groups, dilation, bias=bias
        )

        self.act1 = AF.create(activation)
        self.conv2 = _convk(channels, channels, kernel_size, groups=groups, bias=bias)

        self.act2 = AF.create(activation)
        self.stride = stride

        self.upsample = None
        if stride != 1 or in_channels != channels:
            self.upsample = _make_upsample(
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

        self.context = (stride + dilation) * (kernel_size - 1) // 2
        self.upsample_factor = stride

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.upsample is not None:
            residual = self.upsample(residual)

        x += residual
        x = self.act2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet1dBNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        self.expansion = expansion
        bn_channels = channels // expansion
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            self.bn1 = norm_layer(bn_channels)
            self.bn2 = norm_layer(bn_channels)
            self.bn3 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _conv1(in_channels, bn_channels, stride=1, bias=bias)
        self.conv2 = _convk(
            bn_channels,
            bn_channels,
            kernel_size,
            stride,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.conv3 = _conv1(bn_channels, channels, stride=1, bias=bias)

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

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)
        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        x = self.act2(x)
        if self.norm_after:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        if self.norm_after:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet1dBNDecBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        self.expansion = expansion
        bn_channels = channels // expansion
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm1d
            self.bn1 = norm_layer(bn_channels)
            self.bn2 = norm_layer(bn_channels)
            self.bn3 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _conv1(in_channels, bn_channels, stride=1, bias=bias)
        self.conv2 = _subpixel_convk(
            bn_channels, bn_channels, kernel_size, stride, groups, dilation, bias=bias
        )
        self.conv3 = _conv1(bn_channels, channels, stride=1, bias=bias)

        self.act1 = AF.create(activation)
        self.act2 = AF.create(activation)
        self.act3 = AF.create(activation)
        self.stride = stride

        self.upsample = None
        if stride != 1 or in_channels != channels:
            self.upsample = _make_upsample(
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
        self.upsample_factor = stride

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)
        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        x = self.act2(x)
        if self.norm_after:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.upsample is not None:
            residual = self.upsample(residual)

        x += residual
        x = self.act3(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet1dBasicBlock(ResNet1dBasicBlock):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            groups=groups,
            dilation=dilation,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock1d(channels, se_r, activation)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        x = self.se_layer(x)
        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet1dBasicDecBlock(ResNet1dBasicDecBlock):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            groups=groups,
            dilation=dilation,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock1d(channels, se_r, activation)

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        x = self.se_layer(x)
        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.upsample is not None:
            residual = self.upsample(residual)

        x += residual
        x = self.act2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet1dBNBlock(ResNet1dBNBlock):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            groups=groups,
            dilation=dilation,
            expansion=expansion,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock1d(channels, se_r, activation)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)
        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        x = self.act2(x)
        if self.norm_after:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        x = self.se_layer(x)
        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        if self.norm_after:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet1dBNDecBlock(ResNet1dBNDecBlock):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu6",
        stride=1,
        dropout_rate=0,
        drop_connect_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            groups=groups,
            dilation=dilation,
            expansion=expansion,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock1d(channels, se_r, activation)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)
        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)

        x = self.act2(x)
        if self.norm_after:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)

        x = self.se_layer(x)
        if self.drop_connect_rate > 0:
            x = self.drop_connect(x)

        if self.upsample is not None:
            residual = self.upsample(residual)

        x += residual
        x = self.act3(x)

        if self.norm_after:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet1dEndpoint(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        in_scale,
        scale,
        upsampling_mode="nearest",
        activation={"name": "relu6", "inplace": True},
        norm_layer=None,
        norm_before=True,
    ):
        """
        Class that connects the ouputs of the ResNet1d to the rest of the network
        when using multilevel feature aggregation

        It converts the features of all the levels that we are going to aggregate
        to the same temporal scale
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.in_channels = in_channels
        self.channels = channels
        self.norm_before = norm_before
        self.rel_scale = in_scale / scale
        if scale >= in_scale:
            stride = int(scale / in_scale)
            self.resample = _make_downsample(
                in_channels, channels, stride, norm_layer, norm_before
            )
        else:
            stride = int(in_scale / scale)
            self.resample = _make_upsample(
                in_channels,
                channels,
                stride,
                norm_layer,
                norm_before,
                mode=upsampling_mode,
            )

        self.act = AF.create(activation)
        if not self.norm_before:
            self.bn = norm_layer(channels)

    def forward(self, x):
        x = self.resample(x)
        x = self.act(x)
        if not self.norm_before:
            x = self.bn(x)
        return x
