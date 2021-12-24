"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from .se_blocks import SEBlock1d


def _conv1(in_channels, out_channels, bias=False):
    """1x1 convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


def _dwconvk(channels, kernel_size, stride=1, bias=False):
    """kxk depth-wise convolution with padding"""
    return nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        groups=channels,
        bias=bias,
        padding_mode="zeros",
    )


def _make_downsample(in_channels, out_channels, stride):
    return _conv1(in_channels, out_channels, stride, bias=True)


class ConformerConvBlock(nn.Module):
    """Convolutional block for conformer introduced at
        https://arxiv.org/pdf/2005.08100.pdf

        This includes some optional extra features
        not included in the original paper:
           - Squeeze-Excitation after depthwise-conv
           - Allows downsampling in time dimension
           - Allows choosing activation and layer normalization type

    Attributes:
       num_channels : number of input/output channels
       kernel_size: kernel_size for depth-wise conv
       stride: stride for depth-wise conv
       activation: activation function str or object
       norm_layer: norm layer constructor,
                   if None it uses BatchNorm
       dropout_rate: dropout rate
       se_r:         Squeeze-Excitation compression ratio,
                     if None it doesn't use Squeeze-Excitation
    """

    def __init__(
        self,
        num_channels,
        kernel_size,
        stride=1,
        activation="swish",
        norm_layer=None,
        dropout_rate=0,
        se_r=None,
    ):
        super().__init__()
        self.num_channels = (num_channels,)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.act = AF.create(activation)
        self.se_r = se_r
        self.has_se = se_r is not None and se_r > 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.layer_norm = nn.LayerNorm(num_channels)
        # expansion phase
        self.conv_exp = _conv1(num_channels, 2 * num_channels, bias=True)

        # depthwise conv phase
        self.conv_dw = _dwconvk(num_channels, kernel_size, stride=stride, bias=False)
        self.norm_dw = norm_layer(num_channels, momentum=0.01, eps=1e-3)
        if self.has_se:
            self.se_layer = SEBlock1d(num_channels, se_r, activation)

        # final projection
        self.conv_proj = _conv1(num_channels, num_channels, bias=True)
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # when input and output dimensions are different, we adapt the dimensions using conv1x1
        self.downsample = None
        if stride != 1:
            self.downsample = _make_downsample(num_channels, num_channels, stride)

        self.context = stride * (kernel_size - 1) // 2

    def forward(self, x):
        """Forward function

        Args:
          x: input size = (batch, num_channels, time)

        Returns
          torch.Tensor size = (batch, num_channels, (time-1)//stride+1)
        """
        residual = x

        # layer norm
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)

        # expansion + glu
        x = self.conv_exp(x)
        x = nn.functional.glu(x, dim=1)

        # depthwide conv phase
        x = self.act(self.norm_dw(self.conv_dw(x)))
        if self.has_se:
            x = self.se_layer(x)

        # final projection
        x = self.conv_proj(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        return x
