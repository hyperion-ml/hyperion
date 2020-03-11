"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d, Dropout2d

from ..layers import ActivationFactory as AF
from .resnet_blocks import ResNetBasicBlock, ResNetBNBlock


class SEBlock2D(nn.Module):
    """ From https://arxiv.org/abs/1709.01507
    """
    def __init__(self, num_channels, r=16, activation={'name':'relu', 'inplace': True}):
        super(SEBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, int(num_channels/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(int(num_channels/r), num_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        z = torch.mean(x, dim=(2,3), keepdim=True)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        y = scale * x
        return y


class TSEBlock2D(nn.Module):
    """ From https://arxiv.org/abs/1709.01507
        Modified to do pooling only in time dimension
    """
    def __init__(self, num_channels, num_feats, r=16, activation={'name':'relu', 'inplace': True}):
        super(TSEBlock2D, self).__init__()
        self.num_channels_1d = num_channels*num_feats
        self.conv1 = nn.Conv2d(self.num_channels_1d, int(self.num_channels_1d/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(int(self.num_channels_1d/r), self.num_channels_1d, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        num_feats = x.shape[2]
        num_channels = x.shape[1]
        z = torch.mean(x, dim=-1, keepdim=True)
        z = z.view(-1, self.num_channels_1d, 1, 1)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        scale = scale.view(-1, num_channels, num_feats, 1)
        y = scale * x
        return y


class SEResNetBasicBlock(ResNetBasicBlock):

    def __init__(self, in_channels, channels, 
                 activation={'name':'relu', 'inplace': True},
                 stride=1, dropout_rate=0, groups=1, dilation=1,
                 norm_layer=None, norm_before=True, 
                 r=16, time_se=False, num_feats=None):

        super(SEResNetBasicBlock, self).__init__(
            in_channels, channels, activation=activation,
            stride=stride, dropout_rate=dropout_rate, 
            groups=groups, dilation=dilation, 
            norm_layer=norm_layer, norm_before=norm_before)

        if time_se:
            self.se_layer = TSEBlock2D(channels, num_feats, r, activation)
        else:
            self.se_layer = SEBlock2D(channels, r, activation)


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

    def __init__(self, in_channels, channels, 
                 activation={'name':'relu', 'inplace': True},
                 stride=1, dropout_rate=0, groups=1,
                 dilation=1, norm_layer=None, norm_before=True, 
                 r=16, time_se=False, num_feats=None):

        super(SEResNetBNBlock, self).__init__(
            in_channels, channels, activation=activation,
            stride=stride, dropout_rate=dropout_rate,groups=groups,
            dilation=dilation, norm_layer=norm_layer, norm_before=norm_before)

        if time_se:
            self.se_layer = TSEBlock2D(channels, num_feats, r, activation)
        else:
            self.se_layer = SEBlock2D(channels, r, activation)


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


