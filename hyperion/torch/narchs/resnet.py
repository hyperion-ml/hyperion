"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..layer_blocks import (
    ResNetInputBlock,
    ResNetBasicBlock,
    ResNetBNBlock,
    SEResNetBasicBlock,
    SEResNetBNBlock,
    Res2NetBasicBlock,
    Res2NetBNBlock,
)
from ..layer_blocks import ResNetEndpointBlock
from .net_arch import NetArch


class ResNet(NetArch):
    """ResNet2D base class

    Attributes:
      block: resnet basic block type in
             ['basic', 'bn', 'sebasic', 'sebn', 'res2basic'
             'res2bn', 'seres2basic', 'seres2bn'], meaning
             basic resnet block, bottleneck resnet block, basic block with squeeze-excitation,
             bottleneck block with squeeze-excitation, Res2Net basic and bottlenec, and
             squeeze-excitation Res2Net basic and bottleneck.

      num_layers: list with the number of layers in each of the 4 layer blocks that we find in
                  resnets, after each layer block feature maps are downsmapled times 2 in each dimension
                  and channels are upsampled times 2.
      in_channels: number of input channels
      conv_channels: number of output channels in first conv layer (stem)
      base_channels: number of channels in the first layer block
      out_units: number of logits in the output layer, if 0 there is no output layer and resnet is used just
                 as feature extractor, for example for x-vector encoder.
      in_kernel_size: kernels size of first conv layer
      hid_act: str or dictionary describing hidden activations.
      out_act: output activation
      zero_init_residual: initializes batchnorm weights to zero so each residual block behaves as identitiy at
                          the beggining. We observed worse results when using this option in x-vectors
      multilevel: if True, the output is the combination of the feature maps at different resolution levels.
      endpoint_channels: number of output channels when multilevel is True.
      groups: number of groups in convolutions
      replace_stride_with_dilation: use dialted conv nets instead of downsammpling, we never tested this.
      dropout_rate: dropout rate
      norm_layer: norm_layer object or str indicating type layer-norm object, if None it uses BatchNorm2d
      do_maxpool: if False, removes the maxpooling layer at the stem of the network.
      in_norm: if True, adds another batch norm layer in the input
      se_r: squeeze-excitation dimension compression
      time_se: if True squeeze-excitation embedding is obtaining by averagin only in the time dimension,
               instead of time-freq dimension or HxW dimensions
      in_feats: input feature size (number of components in dimension of 2 of input tensor), this is only
                required when time_se=True to calculcate the size of the squeeze excitation matrices.
      res2net_scale: Res2Net scale parameter
      res2net_width_factor: Res2Net multiplier for the width of the bottlneck layers.
    """

    def __init__(
        self,
        block,
        num_layers,
        in_channels,
        conv_channels=64,
        base_channels=64,
        out_units=0,
        hid_act={"name": "relu6", "inplace": True},
        out_act=None,
        in_kernel_size=7,
        in_stride=2,
        zero_init_residual=False,
        multilevel=False,
        endpoint_channels=64,
        groups=1,
        replace_stride_with_dilation=None,
        dropout_rate=0,
        norm_layer=None,
        norm_before=True,
        do_maxpool=True,
        in_norm=True,
        se_r=16,
        time_se=False,
        in_feats=None,
        res2net_scale=4,
        res2net_width_factor=1,
    ):

        super().__init__()
        logging.info("{}".format(locals()))
        self.block = block
        self.has_se = False
        self.is_res2net = False
        if isinstance(block, str):
            if block == "basic":
                self._block = ResNetBasicBlock
            elif block == "bn":
                self._block = ResNetBNBlock
            elif block == "sebasic":
                self._block = SEResNetBasicBlock
                self.has_se = True
            elif block == "sebn":
                self._block = SEResNetBNBlock
                self.has_se = True
            elif block == "res2basic":
                self._block = Res2NetBasicBlock
                self.is_res2net = True
            elif block == "res2bn":
                self._block = Res2NetBNBlock
                self.is_res2net = True
            elif block == "seres2bn" or block == "tseres2bn":
                self._block = Res2NetBNBlock
                self.has_se = True
                self.is_res2net = True
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
        # self.width_per_group = width_per_group
        self.se_r = se_r
        self.time_se = time_se
        self.in_feats = in_feats
        self.res2net_scale = res2net_scale
        self.res2net_width_factor = res2net_width_factor

        self.multilevel = multilevel
        self.endpoint_channels = endpoint_channels

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(base_channels // 2, 32)
            norm_groups = max(norm_groups, groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.groups = groups
        # self.width_per_group = width_per_group

        if in_norm:
            self.in_bn = norm_layer(in_channels)

        self.in_block = ResNetInputBlock(
            in_channels,
            conv_channels,
            kernel_size=in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
        )

        self._context = self.in_block.context
        self._downsample_factor = self.in_block.downsample_factor

        self.cur_in_channels = conv_channels
        self.layer1 = self._make_layer(self._block, base_channels, num_layers[0])
        self.layer2 = self._make_layer(
            self._block,
            2 * base_channels,
            num_layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            self._block,
            4 * base_channels,
            num_layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            self._block,
            8 * base_channels,
            num_layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        if self.multilevel:
            self.endpoint2 = ResNetEndpointBlock(
                2 * base_channels * self._block.expansion,
                self.endpoint_channels,
                1,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )
            self.endpoint3 = ResNetEndpointBlock(
                4 * base_channels * self._block.expansion,
                self.endpoint_channels,
                2,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )
            self.endpoint4 = ResNetEndpointBlock(
                8 * base_channels * self._block.expansion,
                self.endpoint_channels,
                4,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )

        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output = nn.Linear(self.cur_in_channels, out_units)
            self.out_act = AF.create(out_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = "relu"
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                try:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=act_name
                    )
                except:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
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

        kwargs = {}
        if self.has_se:
            if self.time_se:
                num_feats = int(self.in_feats / (self._downsample_factor * stride))
                kwargs = {"se_r": self.se_r, "time_se": True, "num_feats": num_feats}
            else:
                kwargs = {"se_r": self.se_r}

        if self.is_res2net:
            kwargs["scale"] = self.res2net_scale
            kwargs["width_factor"] = self.res2net_width_factor

        layers = []
        layers.append(
            block(
                self.cur_in_channels,
                channels,
                activation=self.hid_act,
                stride=stride,
                dropout_rate=self.dropout_rate,
                groups=self.groups,
                dilation=previous_dilation,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
                **kwargs
            )
        )

        self._context += layers[0].context * self._downsample_factor
        self._downsample_factor *= layers[0].downsample_factor

        self.cur_in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.cur_in_channels,
                    channels,
                    activation=self.hid_act,
                    dropout_rate=self.dropout_rate,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                    norm_before=self.norm_before,
                    **kwargs
                )
            )

            self._context += layers[-1].context * self._downsample_factor

        return nn.Sequential(*layers)

    def _compute_out_size(self, in_size):
        """Computes output size given input size.
           Output size is not the same as input size because of
           downsampling steps.

        Args:
           in_size: input size of the H or W dimensions

        Returns:
           output_size
        """
        out_size = int((in_size - 1) // self.in_stride + 1)
        if self.do_maxpool:
            out_size = int((out_size - 1) // 2 + 1)

        for i in range(3):
            if not self.replace_stride_with_dilation[i]:
                out_size = int((out_size - 1) // 2 + 1)

        return out_size

    def in_context(self):
        """
        Returns:
          Tuple (past, future) context required to predict one frame.
        """
        return (self._context, self._context)

    def in_shape(self):
        """
        Returns:
          Tuple describing input shape for the network
        """
        return (None, self.in_channels, None, None)

    def out_shape(self, in_shape=None):
        """Computes the output shape given the input shape

        Args:
          in_shape: input shape
        Returns:
          Tuple describing output shape for the network
        """

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

        if self.multilevel:
            return (in_shape[0], self.endpoint_channels, int(in_shape[2] // 2), None)

        return (in_shape[0], self.layer4[-1].out_channels, H, W)

    def forward(self, x, x_lengths=None):
        """forward function

        Args:
           x: input tensor of size=(batch, Cin, Hin, Win) for image or
              size=(batch, C, freq, time) for audio
           x_lengths: when x are sequences with time in Win dimension, it
                      contains the lengths of the sequences.
        Returns:
           Tensor with output logits of size=(batch, out_units) if out_units>0,
           otherwise, it returns tensor of represeantions of size=(batch, Cout, Hout, Wout)

        """

        if self.in_norm:
            x = self.in_bn(x)
        feats = []
        x = self.in_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.multilevel:
            feats.append(x)
        x = self.layer3(x)
        if self.multilevel:
            feats.append(x)
        x = self.layer4(x)
        if self.multilevel:
            feats.append(x)

        if self.multilevel:
            out2 = self.endpoint2(feats[0])
            out3 = self.endpoint3(feats[1])
            out4 = self.endpoint4(feats[2])
            x = torch.mean(torch.stack([out2, out3, out4]), 0)

        if self.with_output:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def forward_hid_feats(self, x, layers=None, return_output=False):
        """forward function which also returns intermediate hidden representations

        Args:
           x: input tensor of size=(batch, Cin, Hin, Win) for image or
              size=(batch, C, freq, time) for audio
           layers: list of hidden layers to return hidden representations
           return_output: if True if returns the output representations in a separate
                          tensor.
        Returns:
            List of hidden representation tensors
            Tensor with output representations if return_output is True

        """
        assert layers is not None or return_output
        if layers is None:
            layers = []

        if return_output:
            last_layer = 4
        else:
            last_layer = max(layers)

        h = []
        feats = []
        if self.in_norm:
            x = self.in_bn(x)

        x = self.in_block(x)
        if 0 in layers:
            h.append(x)
        if last_layer == 0:
            return h

        x = self.layer1(x)
        if 1 in layers:
            h.append(x)
        if last_layer == 1:
            return h

        x = self.layer2(x)
        if 2 in layers:
            h.append(x)
        if last_layer == 2:
            return h
        if return_output and self.multilevel:
            feats.append(x)

        x = self.layer3(x)
        if 3 in layers:
            h.append(x)
        if last_layer == 3:
            return h
        if return_output and self.multilevel:
            feats.append(x)

        x = self.layer4(x)
        if 4 in layers:
            h.append(x)
        if return_output and self.multilevel:
            feats.append(x)

        if return_output:
            if self.multilevel:
                out2 = self.endpoint2(feats[0])
                out3 = self.endpoint3(feats[1])
                out4 = self.endpoint4(feats[2])
                x = torch.mean(torch.stack([out2, out3, out4]), 0)

            return h, x

        return h

    def get_config(self):
        """Gets network config
        Returns:
           dictionary with config params
        """

        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {
            "block": self.block,
            "num_layers": self.num_layers,
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "base_channels": self.base_channels,
            "out_units": self.out_units,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "zero_init_residual": self.zero_init_residual,
            "groups": self.groups,
            "replace_stride_with_dilation": self.replace_stride_with_dilation,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "do_maxpool": self.do_maxpool,
            "out_act": out_act,
            "hid_act": hid_act,
            "se_r": self.se_r,
            "in_feats": self.in_feats,
            "res2net_scale": self.res2net_scale,
            "res2net_width_factor": self.res2net_width_factor,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Standard ResNets
class ResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("basic", [2, 2, 2, 2], in_channels, **kwargs)


class ResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)


class ResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class ResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class ResNet152(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("bn", [3, 8, 36, 3], in_channels, **kwargs)


class ResNext50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class ResNext101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class WideResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class WideResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class LResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("basic", [2, 2, 2, 2], in_channels, **kwargs)


class LResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)


class LResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class LResNext50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


# Squezee-Excitation ResNets


class SEResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class SEResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SEResNet152(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class SEResNext50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNext101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SEWideResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEWideResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SELResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class SELResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class SELResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SELResNext50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


# Time dimension Squezee-Excitation ResNets


class TSEResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class TSEResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEResNet152(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class TSEResNext50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNext101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEWideResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEWideResNet101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSELResNet18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class TSELResNet34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class TSELResNet50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSELResNext50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


#################### Res2Net variants ########################

# Standard Res2Nets
class Res2Net18(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("res2basic", [2, 2, 2, 2], in_channels, **kwargs)


class Res2Net34(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("res2basic", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class Res2Net152(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("res2bn", [3, 8, 36, 3], in_channels, **kwargs)


class Res2Next50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Next101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class WideRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class WideRes2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class LRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class LRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


# Squezee-Excitation Res2Nets
class SERes2Net18(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("seres2basic", [2, 2, 2, 2], in_channels, **kwargs)


class SERes2Net34(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("seres2basic", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SERes2Net152(ResNet):
    def __init__(self, in_channels, **kwargs):
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class SERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SEWideRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SEWideRes2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SELRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


# Time dimension Squezee-Excitation Res2Nets
class TSERes2Net18(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("se2basic", [2, 2, 2, 2], in_channels, **kwargs)


class TSERes2Net34(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("se2basic", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSERes2Net152(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class TSERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEWideRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEWideRes2Net101(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSELRes2Net50(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


# multi-level feature ResNet
class LResNet34_345(ResNet):
    def __init__(self, in_channels, **kwargs):
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["multilevel"] = True
        kwargs["endpoint_channels"] = 64
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)
