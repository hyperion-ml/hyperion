"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Type, Union

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF, ActivationSpec
from ..layers import DropPath1d, DropPath2d, GRN1d, GRN2d, Interpolate


class ConvNext2dBlock(nn.Module):
    """ConvNeXtV2 block with 2D convolutions.

    Attributes:
      dwconv: Depthwise convolution over the spatial dimensions.
      norm: Normalization layer applied in channels-last format.
      pwconv1: First pointwise projection implemented with a linear layer.
      act: Activation module created by the activation factory.
      grn: Global response normalization layer.
      pwconv2: Second pointwise projection back to the input width.
      drop_path: Stochastic depth module.
      context: Effective padding/context contributed by the block.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 7,
        activation: ActivationSpec = "gelu",
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize the 2D ConvNeXt block.

        Args:
          num_channels: Number of input and output channels.
          kernel_size: Depthwise convolution kernel size.
          activation: Activation specification accepted by ``ActivationFactory``.
          norm_layer: Normalization layer constructor, if any.
          drop_path_rate: Stochastic depth rate.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_channels,
        )  # depthwise conv
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.norm = norm_layer(num_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(
            num_channels, 4 * num_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = AF.create(activation)
        self.grn = GRN2d(4 * num_channels, channels_last=True)
        self.pwconv2 = nn.Linear(4 * num_channels, num_channels)
        self.drop_path = (
            DropPath2d(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.context = padding

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the block.

        Args:
          x: Input tensor with shape ``(batch, channels, height, width)``.
          x_mask: Optional binary mask in the same spatial layout as ``x``.

        Returns:
          Output tensor with the same shape as ``x``.
        """
        input = x
        # x.contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # x = x.contiguous()
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, x_mask)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # x = x.contiguous()
        x = input + self.drop_path(x)
        return x


class ConvNext1dBlock(nn.Module):
    """ConvNeXtV2 block with 1D convolutions.

    Attributes:
      dwconv: Depthwise convolution over the temporal dimension.
      norm: Normalization layer applied in channels-last format.
      pwconv1: First pointwise projection implemented with a linear layer.
      act: Activation module created by the activation factory.
      grn: Global response normalization layer.
      pwconv2: Second pointwise projection back to the input width.
      drop_path: Stochastic depth module.
      context: Effective temporal context contributed by the block.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        activation: ActivationSpec = "gelu",
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize the 1D ConvNeXt block.

        Args:
          num_channels: Number of input and output channels.
          kernel_size: Depthwise convolution kernel size.
          dilation: Depthwise convolution dilation.
          activation: Activation specification accepted by ``ActivationFactory``.
          norm_layer: Normalization layer constructor, if any.
          drop_path_rate: Stochastic depth rate.
        """
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dwconv = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=num_channels,
        )  # depthwise conv
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.norm = norm_layer(num_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(
            num_channels, 4 * num_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = AF.create(activation)
        self.grn = GRN1d(4 * num_channels, channels_last=True)
        self.pwconv2 = nn.Linear(4 * num_channels, num_channels)
        self.drop_path = (
            DropPath1d(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.context = padding

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the block.

        Args:
          x: Input tensor with shape ``(batch, channels, time)``.
          x_mask: Optional binary mask in the same temporal layout as ``x``.

        Returns:
          Output tensor with the same shape as ``x``.
        """
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, x_mask)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        x = input + self.drop_path(x)
        return x


class ConvNext2dStemBlock(nn.Module):
    """ConvNeXt-v2 2D input stem block.

    Attributes:
      conv: Strided convolution used to downsample the input.
      norm: Normalization layer applied in channels-last format.
      context: Effective spatial context contributed by the stem.
      stride: Stride used by the stem convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 4,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 2D stem block.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          kernel_size: Convolution kernel size.
          stride: Convolution stride.
          norm_layer: Normalization layer constructor, if any.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_layer(out_channels, eps=1e-6)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem block.

        Args:
          x: Input tensor with shape ``(batch, channels, height, width)``.

        Returns:
          Downsampled tensor with shape ``(batch, out_channels, out_height, out_width)``.
        """
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1))  # .contiguous())
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNext1dStemBlock(nn.Module):
    """ConvNeXt-v2 1D input stem block.

    Attributes:
      conv: Strided convolution used to downsample the input.
      norm: Normalization layer applied in channels-last format.
      context: Effective temporal context contributed by the stem.
      stride: Stride used by the stem convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 4,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 1D stem block.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          kernel_size: Convolution kernel size.
          stride: Convolution stride.
          norm_layer: Normalization layer constructor, if any.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_layer(out_channels, eps=1e-6)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem block.

        Args:
          x: Input tensor with shape ``(batch, channels, time)``.

        Returns:
          Downsampled tensor with shape ``(batch, out_channels, out_time)``.
        """
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 1))
        return x.permute(0, 2, 1).contiguous()


class ConvNext2dDownsampleBlock(nn.Module):
    """ConvNeXt-v2 2D downsampling block.

    Attributes:
      norm: Normalization layer applied before the stride convolution.
      conv: Strided convolution used to downsample the input.
      context: Effective spatial context contributed by the block.
      stride: Stride used by the downsampling convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 2D downsampling block.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          kernel_size: Convolution kernel size.
          stride: Convolution stride.
          norm_layer: Normalization layer constructor, if any.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.norm = norm_layer(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the downsampling block.

        Args:
          x: Input tensor with shape ``(batch, channels, height, width)``.

        Returns:
          Downsampled tensor with shape ``(batch, out_channels, out_height, out_width)``.
        """
        x = self.norm(x.permute(0, 2, 3, 1))  # .contiguous())
        return self.conv(x.permute(0, 3, 1, 2).contiguous())


class ConvNext1dDownsampleBlock(nn.Module):
    """ConvNeXt-v2 1D downsampling block.

    Attributes:
      norm: Normalization layer applied before the stride convolution.
      conv: Strided convolution used to downsample the input.
      context: Effective temporal context contributed by the block.
      stride: Stride used by the downsampling convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 1D downsampling block.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          kernel_size: Convolution kernel size.
          stride: Convolution stride.
          norm_layer: Normalization layer constructor, if any.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.norm = norm_layer(in_channels, eps=1e-6)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the downsampling block.

        Args:
          x: Input tensor with shape ``(batch, channels, time)``.

        Returns:
          Downsampled tensor with shape ``(batch, out_channels, out_time)``.
        """
        x = self.norm(x.permute(0, 2, 1))
        return self.conv(x.permute(0, 2, 1).contiguous())


class ConvNext2dEndpoint(nn.Module):
    """Endpoint that maps 2D ConvNeXt features to a common scale.

    Attributes:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      rel_scale: Ratio between input and output scales.
      norm: Normalization layer applied before resampling.
      resample: Sequential module that performs upsampling or downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_scale: int,
        out_scale: int,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 2D endpoint.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          in_scale: Resolution scale of the input feature maps.
          out_scale: Resolution scale of the output feature maps.
          norm_layer: Normalization layer constructor, if any.
        """

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_scale = in_scale / out_scale
        self.norm = norm_layer(in_channels, eps=1e-6)
        if out_scale >= in_scale:
            stride = int(out_scale / in_scale)
            self.resample = self._make_downsample(in_channels, out_channels, stride)
        else:
            stride = int(in_scale / out_scale)
            self.resample = self._make_upsample(
                in_channels,
                out_channels,
                stride,
            )

    @staticmethod
    def _make_downsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        """Build a downsampling resampling path.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          stride: Total downsampling factor.

        Returns:
          Sequential module that downsamples and projects the tensor.
        """

        if stride % 2 == 0:
            first_stride = 2
            second_stride = stride // 2
        else:
            first_stride = 1
            second_stride = stride

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=first_stride,
                stride=first_stride,
                bias=True,
            )
        ]

        if second_stride > 1:
            kernel_size = 2 * (second_stride // 2) + 1
            layers.append(
                nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=second_stride,
                    padding=(kernel_size - 1) // 2,
                )
            )

        return nn.Sequential(*layers)

    @staticmethod
    def _make_upsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        """Build an upsampling resampling path.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          stride: Upsampling factor.

        Returns:
          Sequential module that projects and upsamples the tensor.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        ]
        layers.append(Interpolate(scale_factor=stride, mode="nearest"))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the endpoint.

        Args:
          x: Input tensor with shape ``(batch, in_channels, height, width)``.
          x_mask: Unused.

        Returns:
          Tensor with shape ``(batch, out_channels, out_height, out_width)``.
        """
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x = self.resample(x)
        return x


class ConvNext1dEndpoint(nn.Module):
    """Endpoint that maps 1D ConvNeXt features to a common scale.

    Attributes:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      rel_scale: Ratio between input and output scales.
      norm: Normalization layer applied before resampling.
      resample: Sequential module that performs upsampling or downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_scale: int,
        out_scale: int,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the 1D endpoint.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          in_scale: Resolution scale of the input feature maps.
          out_scale: Resolution scale of the output feature maps.
          norm_layer: Normalization layer constructor, if any.
        """

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_scale = in_scale / out_scale
        self.norm = norm_layer(in_channels, eps=1e-6)
        if out_scale >= in_scale:
            stride = int(out_scale / in_scale)
            self.resample = self._make_downsample(in_channels, out_channels, stride)
        else:
            stride = int(in_scale / out_scale)
            self.resample = self._make_upsample(
                in_channels,
                out_channels,
                stride,
            )

    @staticmethod
    def _make_downsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        """Build a downsampling resampling path.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          stride: Total downsampling factor.

        Returns:
          Sequential module that downsamples and projects the tensor.
        """

        if stride % 2 == 0:
            first_stride = 2
            second_stride = stride // 2
        else:
            first_stride = 1
            second_stride = stride

        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=first_stride,
                stride=first_stride,
                bias=True,
            )
        ]

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

    @staticmethod
    def _make_upsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        """Build an upsampling resampling path.

        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          stride: Upsampling factor.

        Returns:
          Sequential module that projects and upsamples the tensor.
        """
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        ]
        layers.append(Interpolate(scale_factor=stride, mode="nearest"))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the endpoint.

        Args:
          x: Input tensor with shape ``(batch, in_channels, time)``.
          x_mask: Unused.

        Returns:
          Tensor with shape ``(batch, out_channels, out_time)``.
        """
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.resample(x)
        return x
