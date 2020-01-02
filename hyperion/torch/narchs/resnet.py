"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layer_blocks import ResNetInputBlock, ResNetBasicBlock, ResNetBNBlock
from .net_arch import NetArch

class ResNet(NetArch):


    def __init__(self, block, num_layers, in_channels, conv_channels=64, base_channels=64, out_units=0,
                 hid_act={'name':'relu6', 'inplace': True}, out_act=None,
                 in_kernel_size=7, in_stride=2,
                 zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None, dropout_rate=0,
                 norm_layer=None, norm_before=True, do_maxpool=True, in_norm=True):

        super(ResNet, self).__init__()

        self.block = block
        if isinstance(block, str):
            if block == 'basic':
                self._block = ResNetBasicBlock
            elif block == 'bn':
                self._block = ResNetBNBlock
        else:
            self._block = block

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.base_channels = base_channels
        self.out_units = out_units
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        self.hid_act = hid_act
        self.groups = groups
        self.norm_before = norm_before
        self.do_maxpool = do_maxpool
        self.in_norm = in_norm
        self.dropout_rate = dropout_rate
        #self.width_per_group = width_per_group
        
        self.norm_layer = norm_layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.groups = groups
        #self.width_per_group = width_per_group

        if in_norm:
            self.in_bn = norm_layer(in_channels)


        self.in_block = ResNetInputBlock(
            in_channels, conv_channels, kernel_size=in_kernel_size, stride=in_stride,
            activation=hid_act, norm_layer=norm_layer, norm_before=norm_before, do_maxpool=do_maxpool)

        self._context = self.in_block.context
        self._downsample_factor = self.in_block.downsample_factor

        self.cur_in_channels = conv_channels
        self.layer1 = self._make_layer(self._block, base_channels, num_layers[0])
        self.layer2 = self._make_layer(self._block, 2*base_channels, num_layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self._block, 4*base_channels, num_layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self._block, 8*base_channels, num_layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output = nn.Linear(self.cur_in_channels, out_units)
            self.out_act = AF.create(out_act)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = 'relu'
                if isinstance(hid_act, dict):
                    act_name = hid_act['name']
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_name)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.zero_init_residual = zero_init_residual
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBNBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, channels, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(block(
            self.cur_in_channels, channels, activation=self.hid_act,
            stride=stride, dropout_rate=self.dropout_rate, groups=self.groups, dilation=previous_dilation, 
            norm_layer=self._norm_layer, norm_before=self.norm_before))

        self._context = layers[0].context * self._downsample_factor
        self._downsample_factor *= layers[0].downsample_factor

        self.cur_in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                self.cur_in_channels, channels, activation=self.hid_act,
                dropout_rate=self.dropout_rate,
                groups=self.groups, dilation=self.dilation, 
                norm_layer=self._norm_layer, norm_before=self.norm_before))
            self._context = layers[-1].context * self._downsample_factor

        return nn.Sequential(*layers)


    def _compute_out_size(self, in_size):
        out_size = int((in_size - 1)//self.in_stride+1)
        if self.do_maxpool:
            out_size = int((out_size - 1)//2+1)

        for i in range(3):
            if not self.replace_stride_with_dilation[i]:
                out_size = int((out_size - 1)//2+1)

        return out_size


    def in_context(self):
        return (self._context, self._context)


    def in_shape(self):
        return (None, self.in_channels, None, None)

            

    def out_shape(self, in_shape=None):
        if self.with_output:
            return (None, self.out_units)

        if in_shape is None:
            return (None, self.layer4[-1].out_channels, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])
            
        return (in_shape[0], self.layer4[-1].out_channels, H, W)



    def forward(self, x):

        if self.in_norm:
            x = self.in_bn(x)

        x = self.in_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_output:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x



    def get_config(self):
        
        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {'block': self.block,
                  'num_layers': self.num_layers,
                  'in_channels': self.in_channels,
                  'conv_channels': self.conv_channels,
                  'base_channels': self.base_channels,
                  'out_units': self.out_units,
                  'in_kernel_size': self.in_kernel_size,
                  'in_stride': self.in_stride,
                  'zero_init_residual': self.zero_init_residual,
                  'groups': self.groups,
                  'replace_stride_with_dilation': self.replace_stride_with_dilation,
                  'dropout_rate': self.dropout_rate,
                  'norm_layer': self.norm_layer,
                  'norm_before': self.norm_before,
                  'in_norm' : self.in_norm,
                  'do_maxpool' : self.do_maxpool,
                  'out_act': out_act,
                  'hid_act': hid_act,
              }
        
        base_config = super(ResNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        super(ResNet18, self).__init__(
            'basic', [2, 2, 2, 2], in_channels, **kwargs)


class ResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        super(ResNet34, self).__init__(
            'basic', [3, 4, 6, 3], in_channels, **kwargs)


class ResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        super(ResNet50, self).__init__(
            'bn', [3, 4, 6, 3], in_channels, **kwargs)

class ResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        super(ResNet101, self).__init__(
            'bn', [3, 4, 23, 3], in_channels, **kwargs)

class ResNet152(ResNet):
    def __init__(self, in_channels, **kwargs):
        super(ResNet152, self).__init__(
            'bn', [3, 8, 36, 3], in_channels, **kwargs)

class ResNext50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['groups'] = 32
        kwargs['base_channels'] = 128
        super(ResNext50_32x4d, self).__init__(
            'bn', [3, 4, 6, 3], in_channels, **kwargs)


class ResNext101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['groups'] = 32
        kwargs['base_channels'] = 256
        super(ResNext101_32x8d, self).__init__(
            'bn', [3, 4, 23, 3], in_channels, **kwargs)


class WideResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['base_channels'] = 128
        super(WideResNet50, self).__init__(
            'bn', [3, 4, 6, 3], in_channels, **kwargs)

class WideResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['base_channels'] = 128
        super(WideResNet101, self).__init__(
            'bn', [3, 4, 23, 3], in_channels, **kwargs)


class LResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LResNet18, self).__init__(
            'basic', [2, 2, 2, 2], in_channels, **kwargs)


class LResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LResNet34, self).__init__(
            'basic', [3, 4, 6, 3], in_channels, **kwargs)

class LResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['conv_channels'] = 16
        kwargs['base_channels'] = 16
        super(LResNet50, self).__init__(
            'bn', [3, 4, 6, 3], in_channels, **kwargs)

class LResNext50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs['groups'] = 4
        kwargs['base_channels'] = 16
        super(LResNext50_4x4d, self).__init__(
            'bn', [3, 4, 6, 3], in_channels, **kwargs)

