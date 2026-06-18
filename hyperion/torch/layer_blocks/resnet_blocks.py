"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import BatchNorm2d, Conv2d, Dropout2d
from typing import Any, Callable, Dict, Optional, Union

from ..layers import ActivationFactory as AF

ActivationType = Union[nn.Module, Dict[str, Any]]


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
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


def _conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _make_downsample(
    in_channels: int,
    out_channels: int,
    stride: int,
    norm_layer: Callable[[int], nn.Module],
    norm_before: bool,
) -> nn.Module:
    if norm_before:
        return nn.Sequential(
            _conv1x1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _conv1x1(in_channels, out_channels, stride, bias=True)


def _require_num_feats(num_feats: Optional[int], what: str) -> int:
    if num_feats is None:
        raise ValueError(f"num_feats must be provided when {what}")
    return num_feats


class FreqPosEnc(nn.Module):
    """Frequency-wise positional encoding.

    Attributes:
      pos_enc: learnable positional offsets with shape=(num_feats, 1).
    """

    def __init__(self, num_feats: int) -> None:
        """Create a frequency positional encoder.

        Args:
          num_feats: number of feature bins.
        """
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros((num_feats, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding to the input.

        Args:
          x: input tensor.

        Returns:
          Tensor with the positional encoding added.
        """
        return x + self.pos_enc


class ResNetInputBlock(nn.Module):
    """Input block for ResNet architecture.

    Attributes:
      conv: input convolution.
      bn: normalization layer.
      act: activation function.
      norm_before: whether normalization is applied before activation.
      do_maxpool: whether to apply max pooling.
      context: receptive field radius.
      downsample_factor: spatial downsampling factor.
      maxpool: max pooling layer, created when ``do_maxpool`` is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        activation: ActivationType = {"name": "relu", "inplace": True},
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        do_maxpool: bool = True,
    ) -> None:
        """Create a ResNet input block.

        Args:
          in_channels: input channels.
          out_channels: output channels.
          kernel_size: convolution kernel size.
          stride: convolution stride.
          activation: activation specification.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
          do_maxpool: if True, apply max pooling at the output.
        """
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

    @property
    def out_channels(self):
        """Return the output channel count.

        Returns:
          Number of output channels.
        """
        return self.conv.out_channels

    @property
    def in_channels(self):
        """Return the input channel count.

        Returns:
          Number of input channels.
        """
        return self.conv.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the input block.

        Args:
          x: input tensor with shape=(batch, channels, height, width).

        Returns:
          Tensor after convolution, normalization, activation, and optional pooling.
        """
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
    """ResNet basic block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.

      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      freq_pos_enc: use frequency wise positional encoder
      num_feats:         Number of features in dimension 2, needed if freq_pos_enc=True.

    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: ActivationType = {"name": "relu", "inplace": True},
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        freq_pos_enc: bool = False,
        num_feats: Optional[int] = None,
    ) -> None:
        """Create a ResNet basic block.

        Args:
          in_channels: input channels.
          channels: output channels.
          activation: activation specification.
          stride: downsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
          freq_pos_enc: if True, add frequency positional encoding.
          num_feats: number of feature bins required by positional encoding.
        """
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
        self.pos_enc = None
        if freq_pos_enc:
            num_feats = _require_num_feats(
                num_feats, "freq_pos_enc=True in ResNetBasicBlock"
            )
            self.pos_enc = FreqPosEnc(num_feats * stride)

    @property
    def out_channels(self):
        """Return the output channel count.

        Returns:
          Number of output channels.
        """
        return self.channels

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the residual block.

        Args:
          x: input tensor with shape=(batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape=(batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if not self.norm_before:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)
            x += residual
            x = self.act2(x)
        else:
            x = self.act2(x)
            x = self.bn2(x)
            x += residual

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNetBNBlock(nn.Module):
    """ResNet bottleneck block.

    Attributes:
      in_channels:       input channels.
      channels:          channels in bottleneck layer when width_factor=1.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      freq_pos_enc: use frequency wise positional encoder
      num_feats:         Number of features in dimension 2, needed if freq_pos_enc=True.
    """

    expansion = 4
    # __constants__ = ['downsample']

    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: ActivationType = {"name": "relu", "inplace": True},
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        freq_pos_enc: bool = False,
        num_feats: Optional[int] = None,
    ) -> None:
        """Create a ResNet bottleneck block.

        Args:
          in_channels: input channels.
          channels: bottleneck channels.
          activation: activation specification.
          stride: downsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
          freq_pos_enc: if True, add frequency positional encoding.
          num_feats: number of feature bins required by positional encoding.
        """
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
        self.pos_enc = None
        if freq_pos_enc:
            num_feats = _require_num_feats(
                num_feats, "freq_pos_enc=True in ResNetBNBlock"
            )
            self.pos_enc = FreqPosEnc(num_feats)

    @property
    def out_channels(self):
        """Return the output channel count.

        Returns:
          Number of output channels.
        """
        return self.channels * self.expansion

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the bottleneck residual block.

        Args:
          x: input tensor with shape=(batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape=(batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

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
            x += residual
            x = self.act3(x)
        else:
            x = self.act3(x)
            x = self.bn3(x)
            x += residual

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class Interpolate(nn.Module):
    """Interpolation wrapper.

    Attributes:
      interp: interpolation function.
      scale_factor: interpolation scale factor.
      mode: interpolation mode.
    """

    def __init__(self, scale_factor: float, mode: str = "nearest") -> None:
        """Create an interpolation wrapper.

        Args:
          scale_factor: interpolation scale factor.
          mode: interpolation mode.
        """
        super().__init__()
        self.interp = nnf.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input tensor.

        Args:
          x: input tensor.

        Returns:
          Upsampled tensor.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ResNetEndpointBlock(nn.Module):
    """ResNet endpoint basic block. This is used as output block when
    the output combines feature maps from different resolution levels.

    Attributes:
      in_channels:       input channels.
      out_channels:      output channels.
      scale:             interpolation factor.
      activation:        Non-linear activation object, string of configuration dictionary.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        activation: ActivationType = {"name": "relu", "inplace": True},
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Create a ResNet endpoint block.

        Args:
          in_channels: input channels.
          out_channels: output channels.
          scale: interpolation factor.
          activation: activation specification.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
        """
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

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the endpoint block.

        Args:
          x: input tensor with shape=(batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape=(batch, out_channels, height, width).
        """

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

