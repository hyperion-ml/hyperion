"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Dropout2d

from ..layers import ActivationFactory as AF
from ..layers.activation_factory import ActivationSpec
from ..layers.subpixel_convs import SubPixelConv2d
from .se_blocks import SEBlock2d


def _convkxk(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """kernel k convolution with padding"""
    padding = dilation * (kernel_size - 1) // 2
    return Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def _conv1x1(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> nn.Conv2d:
    """point-wise convolution"""
    return Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _subpixel_conv1x1(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> nn.Module:
    """point-wise subpixel convolution"""
    return SubPixelConv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
    )


def _subpixel_convkxk(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Module:
    """kernel k subpixel convolution with padding"""
    padding = dilation * (kernel_size - 1) // 2
    return SubPixelConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


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


def _make_upsample(
    in_channels: int,
    out_channels: int,
    stride: int,
    norm_layer: Callable[[int], nn.Module],
    norm_before: bool,
) -> nn.Module:

    if norm_before:
        return nn.Sequential(
            _subpixel_conv1x1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _subpixel_conv1x1(in_channels, out_channels, stride, bias=True)


class ResNet2dBasicBlock(nn.Module):
    """ResNet 2d basic Block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.

    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int = 3,
        activation: ActivationSpec = "relu",
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Initialize a 2D residual basic block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: convolution stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d
            self.bn1 = norm_layer(channels)
            self.bn2 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _convkxk(
            in_channels, channels, kernel_size, stride, groups, dilation, bias=bias
        )
        self.act1 = AF.create(activation)
        self.conv2 = _convkxk(channels, channels, kernel_size, groups=groups, bias=bias)

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
            self.dropout = Dropout2d(dropout_rate)

        self.context = (stride + dilation) * (kernel_size - 1) // 2
        self.downsample_factor = stride

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self.channels

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.norm_after:
            x = self.act2(x)
            x = self.bn2(x)
            x += residual
        else:
            if self.norm_before:
                x = self.bn2(x)

            x += residual
            x = self.act2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet2dBasicDecBlock(nn.Module):
    """ResNet 2d basic Block for decoders.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            upsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.

    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int = 3,
        activation: ActivationSpec = "relu",
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Initialize a 2D residual basic decoder block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: upsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d
            self.bn1 = norm_layer(channels)
            self.bn2 = norm_layer(channels)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.in_channels = in_channels
        self.channels = channels

        bias = not norm_before
        self.conv1 = _subpixel_convkxk(
            in_channels, channels, kernel_size, stride, groups, dilation, bias=bias
        )

        self.act1 = AF.create(activation)
        self.conv2 = _convkxk(channels, channels, kernel_size, groups=groups, bias=bias)

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
            self.dropout = Dropout2d(dropout_rate)

        self.context = (stride + dilation) * (kernel_size - 1) // 2
        self.upsample_factor = stride

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self.channels

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_after:
            x = self.act2(x)
            x = self.bn2(x)
            x += residual
        else:
            if self.norm_before:
                x = self.bn2(x)

            x += residual
            x = self.act2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet2dBNBlock(nn.Module):
    """ResNet 2d bottleneck Block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size in bottleneck.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      expansion:         expansion factor from bottleneck channels to output channels.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int = 3,
        activation: ActivationSpec = "relu",
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        expansion: int = 4,
        use_norm: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Initialize a 2D residual bottleneck block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: convolution stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          expansion: bottleneck expansion factor.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        self.expansion = expansion
        bn_channels = channels // expansion
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d
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
        self.conv1 = _conv1x1(in_channels, bn_channels, stride=1, bias=bias)
        self.conv2 = _convkxk(
            bn_channels,
            bn_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.conv3 = _conv1x1(bn_channels, channels, stride=1, bias=bias)

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
            self.dropout = Dropout2d(dropout_rate)

        self.context = dilation * (kernel_size - 1) // 2
        self.downsample_factor = stride

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self.channels

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

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
        if self.norm_after:
            x = self.act3(x)
            x = self.bn3(x)
            x += residual
        else:
            if self.norm_before:
                x = self.bn3(x)

            x += residual
            x = self.act3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class ResNet2dBNDecBlock(nn.Module):
    """ResNet 2d bottleneck Block decoder.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size in bottleneck.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            upsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      expansion:         expansion factor from bottleneck channels to output channels.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int = 3,
        activation: ActivationSpec = "relu",
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        expansion: int = 4,
        use_norm: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Initialize a 2D residual bottleneck decoder block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: upsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          expansion: bottleneck expansion factor.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__()

        self.norm_before = False
        self.norm_after = False
        self.expansion = expansion
        bn_channels = channels // expansion
        if use_norm:
            if norm_layer is None:
                norm_layer = BatchNorm2d
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
        self.conv1 = _conv1x1(in_channels, bn_channels, stride=1, bias=bias)
        self.conv2 = _subpixel_convkxk(
            bn_channels, bn_channels, kernel_size, stride, groups, dilation, bias=bias
        )
        self.conv3 = _conv1x1(bn_channels, channels, stride=1, bias=bias)

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
            self.dropout = Dropout2d(dropout_rate)

        self.context = dilation * (kernel_size - 1) // 2
        self.upsample_factor = stride

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self.channels

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

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
        if self.norm_after:
            x = self.act3(x)
            x = self.bn3(x)
            x += residual
        else:
            if self.norm_before:
                x = self.bn3(x)

            x += residual
            x = self.act3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet2dBasicBlock(ResNet2dBasicBlock):
    """Squeeze-excitation ResNet 2d basic Block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      drop_connect_rate: drop-connect rate for stochastic number of layers.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      se_r:              squeeze-excitation compression ratio.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu",
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):
        """Initialize a squeeze-excitation 2D residual basic block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: convolution stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          se_r: squeeze-excitation reduction ratio.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock2d(channels, se_r, activation)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_after:
            x = self.act2(x)
            x = self.bn2(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
        else:
            if self.norm_before:
                x = self.bn2(x)

            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet2dBasicDecBlock(ResNet2dBasicDecBlock):
    """Squeeze-excitation ResNet 2d basic Block for decoders.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      drop_connect_rate: drop-connect rate for stochastic number of layers.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      se_r:              squeeze-excitation compression ratio.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu",
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):
        """Initialize a squeeze-excitation 2D residual basic decoder block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: upsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          se_r: squeeze-excitation reduction ratio.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock2d(channels, se_r, activation)

    @property
    def out_channels(self):
        """Returns the number of output channels."""
        return self.channels

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if self.norm_after:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_after:
            x = self.act2(x)
            x = self.bn2(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
        else:
            if self.norm_before:
                x = self.bn2(x)

            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act2(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet2dBNBlock(ResNet2dBNBlock):
    """Squeeze-excitation ResNet 2d bottleneck Block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      expansion:         expansion factor from bottleneck channels to output channels.
      se_r:              squeeze-excitation compression ratio.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu",
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):
        """Initialize a squeeze-excitation 2D residual bottleneck block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: convolution stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          expansion: bottleneck expansion factor.
          se_r: squeeze-excitation reduction ratio.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            expansion=expansion,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock2d(channels, se_r, activation)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

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
        if self.norm_after:
            x = self.act3(x)
            x = self.bn3(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
        else:
            if self.norm_before:
                x = self.bn3(x)

            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNet2dBNDecBlock(ResNet2dBNDecBlock):
    """Squeeze-excitation ResNet 2d bottleneck Block for decoders.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      kernel_size:       kernel size.
      activation:        Activation specification accepted by ActivationFactory.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      expansion:         expansion factor from bottleneck channels to output channels.
      se_r:              squeeze-excitation compression ratio.
      use_norm:          if True, it uses normalization layers, otherwise it does not.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
    """

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        activation="relu",
        stride=1,
        dropout_rate=0,
        groups=1,
        dilation=1,
        expansion=4,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):
        """Initialize a squeeze-excitation 2D residual bottleneck decoder block.

        Args:
          in_channels: input channels.
          channels: output channels.
          kernel_size: convolution kernel size.
          activation: activation specification.
          stride: upsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          expansion: bottleneck expansion factor.
          se_r: squeeze-excitation reduction ratio.
          use_norm: if True, use normalization layers.
          norm_layer: normalization layer constructor, if any.
          norm_before: if True, normalize before activation.
        """

        super().__init__(
            in_channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            expansion=expansion,
            use_norm=use_norm,
            norm_layer=norm_layer,
            norm_before=norm_before,
        )

        self.se_layer = SEBlock2d(channels, se_r, activation)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, out_channels, height, width).
        """
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

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
        if self.norm_after:
            x = self.act3(x)
            x = self.bn3(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
        else:
            if self.norm_before:
                x = self.bn3(x)

            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act3(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
