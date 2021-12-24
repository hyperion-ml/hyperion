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

        x = self.se_layer(x)
        x += residual
        x = self.act2(x)

        if not self.norm_before:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNetBNBlock(ResNetBNBlock):
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

        x = self.se_layer(x)
        x += residual
        x = self.act3(x)

        if not self.norm_before:
            x = self.bn3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
