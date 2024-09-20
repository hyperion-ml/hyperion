"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Type, Union

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import DropPath1d, DropPath2d, GRN1d, GRN2d, Interpolate


class ConvNext2dBlock(nn.Module):
    """ConvNeXtV2 Block with 2d convolutions.

    Args:
        num_channels (int): Number of input channels.
        kernel_size: kernel size
        activation: activation function name or object
        norm_layer: normalization layer constructor, if None, LayerNorm is used.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 7,
        activation: Union[str, nn.Module] = "gelu",
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
    ):
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

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
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
    """ConvNeXtV2 Block with 1d convolutions.

    Args:
        num_channels (int): Number of input channels.
        kernel_size: kernel size
        dilation: dilation factor of convolution
        activation: activation function name or object
        norm_layer: normalization layer constructor, if None, LayerNorm is used.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        activation: Union[str, nn.Module] = "gelu",
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
    ):
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

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
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
    """ConvNext-v2 2d input block

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 4,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
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

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1))  # .contiguous())
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNext1dStemBlock(nn.Module):
    """ConvNext-v2 1d input block

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 4,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
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

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 1))
        return x.permute(0, 2, 1).contiguous()


class ConvNext2dDownsampleBlock(nn.Module):
    """ConvNext-v2 2d downsample block

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
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

    def forward(self, x: torch.Tensor):
        x = self.norm(x.permute(0, 2, 3, 1))  # .contiguous())
        return self.conv(x.permute(0, 3, 1, 2).contiguous())


class ConvNext1dDownsampleBlock(nn.Module):
    """ConvNext-v2 1d downsample block

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
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

    def forward(self, x: torch.Tensor):
        x = self.norm(x.permute(0, 2, 1))
        return self.conv(x.permute(0, 2, 1).contiguous())


class ConvNext2dEndpoint(nn.Module):
    """Class that connects the ouputs of the ConvNext2d to the rest of the network
        when using multilevel feature aggregation.

        It converts the features of all the levels that we are going to aggregate
        to the same temporal scale.

    Attributes:
      in_channels:       input channels.
      out_channels:      output channels.
      in_scale:          resolution scale of the input feature maps.
      out_scale:         resolution scale of the output feature maps.
      norm_layer:        normalization layer constructor, if None BatchNorm1d is used.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_scale,
        out_scale,
        norm_layer=None,
    ):

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
    def _make_downsample(in_channels, out_channels, stride):

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
    def _make_upsample(in_channels, out_channels, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        ]
        layers.append(Interpolate(scale_factor=stride, mode="nearest"))
        return nn.Sequential(*layers)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_heigh, in_width).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_heigh, out_width).
        """
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x = self.resample(x)
        return x


class ConvNext1dEndpoint(nn.Module):
    """Class that connects the ouputs of the ConvNext2d to the rest of the network
        when using multilevel feature aggregation.

        It converts the features of all the levels that we are going to aggregate
        to the same temporal scale.

    Attributes:
      in_channels:       input channels.
      out_channels:      output channels.
      in_scale:          resolution scale of the input feature maps.
      out_scale:         resolution scale of the output feature maps.
      norm_layer:        normalization layer constructor, if None BatchNorm1d is used.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_scale,
        out_scale,
        norm_layer=None,
    ):

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
    def _make_downsample(in_channels, out_channels, stride):

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
    def _make_upsample(in_channels, out_channels, stride):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        ]
        layers.append(Interpolate(scale_factor=stride, mode="nearest"))
        return nn.Sequential(*layers)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_time).
          x_mask: unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_time).
        """
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.resample(x)
        return x
