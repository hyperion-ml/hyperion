"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import DropConnect2d
from .se_blocks import SEBlock2D, TSEBlock2D

# from torch.nn import Conv2d, BatchNorm2d


def _conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """Create a 1x1 convolution.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      stride: Convolution stride.
      bias: If True, includes a bias term.

    Returns:
      A 2D convolution layer with kernel size 1.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _dwconvkxk(
    channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """Create a kxk depth-wise convolution with padding.

    Args:
      channels: Number of input and output channels.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      bias: If True, includes a bias term.

    Returns:
      A depth-wise 2D convolution layer.
    """
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


def _make_downsample(
    in_channels: int,
    out_channels: int,
    stride: int,
    norm_layer: Any,
) -> nn.Sequential:
    """Create the residual downsampling path.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      stride: Convolution stride used to downsample the residual branch.
      norm_layer: Normalization layer constructor.

    Returns:
      A sequential module containing 1x1 projection and normalization.
    """
    return nn.Sequential(
        _conv1x1(in_channels, out_channels, stride, bias=False),
        norm_layer(out_channels, momentum=0.01, eps=1e-3),
    )


class MBConvBlock(nn.Module):
    """MobileNet/EfficientNet inverted bottleneck block.

    Attributes:
      in_channels: Input channels.
      out_channels: Output channels.
      expansion: Expansion ratio for the inverted bottleneck.
      inner_channels: Number of channels in the expanded hidden representation.
      kernel_size: Kernel size used by the depth-wise convolution.
      stride: Stride used by the depth-wise convolution and residual projection.
      activation: Activation specification used to build the non-linearity.
      se_r: Squeeze-excitation reduction ratio, or None to disable SE.
      has_se: True when squeeze-excitation is enabled.
      time_se: True when squeeze-excitation pools only along time.
      num_feats: Number of features in dimension 2, needed if time_se is True.
      drop_connect_rate: Drop-connect rate used on the projection output.
      drop_connect: Drop-connect module, created only when the rate is positive.
      downsample: Residual projection used when input and output shapes differ.
      context: Effective temporal/spatial context contributed by the block.
      downsample_factor: Total stride contribution of the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Union[str, Dict[str, Any], Callable[..., nn.Module]] = "swish",
        drop_connect_rate: float = 0,
        norm_layer: Optional[Any] = None,
        se_r: Optional[int] = None,
        time_se: bool = False,
        num_feats: Optional[int] = None,
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

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: Input tensor with shape = (batch, in_channels, in_height, in_width).
          x_mask: Optional binary mask for valid spatial positions. This block
            does not use the mask directly.

        Returns:
          Tensor with shape = (batch, out_channels, out_height, out_width).
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
    """Convolutional block used as input or output in MobileNet/EfficientNet.

    Attributes:
      in_channels: Input channels.
      out_channels: Output channels.
      kernel_size: Kernel size of the convolution.
      stride: Stride of the convolution.
      activation: Activation specification used to build the non-linearity.
      norm_layer: Normalization layer constructor, if None BatchNorm2d is used.
      context: Padding applied by the convolution, which determines context.
      downsample_factor: Stride contribution of the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        activation: Union[str, Dict[str, Any], Callable[..., nn.Module]] = "swish",
        norm_layer: Optional[Any] = None,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
          x: Input tensor with shape = (batch, in_channels, in_height, in_width).

        Returns:
          Tensor with shape = (batch, out_channels, out_height, out_width).
        """
        return self.act(self.bn(self.conv(x)))
