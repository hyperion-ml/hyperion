"""
Copyright 2020 Magdalena Rybicka
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import BatchNorm2d, Conv2d, Dropout2d

from ..layers import ActivationFactory as AF
from ..layers.subpixel_convs import SubPixelConv2d


class Interpolate(nn.Module):
    """Interpolation layer used by SpineNet resampling paths.

    Attributes:
      interp: Interpolation function used at runtime.
      scale_factor: Multiplicative scale applied to the spatial axes.
      mode: Interpolation mode passed to the interpolation function.
    """

    def __init__(self, scale_factor: float, mode: str = "nearest") -> None:
        """Create an interpolation module.

        Args:
          scale_factor: Multiplicative scale applied to the spatial axes.
          mode: Interpolation mode passed to `torch.nn.functional.interpolate`.
        """
        super().__init__()
        self.interp = nnf.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interpolation to a tensor.

        Args:
          x: Input tensor with shape `(batch, channels, height, width)`.

        Returns:
          The interpolated tensor.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


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
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def _subpixel_conv1x1(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> SubPixelConv2d:
    """point-wise subpixel convolution"""
    return SubPixelConv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
    )


def _make_downsample(
    in_channels: int,
    out_channels: int,
    stride: int,
    norm_layer: Callable[[int], nn.Module],
    norm_before: bool,
) -> nn.Module:
    """Create a downsampling path for a SpineNet resampling branch.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      stride: Stride used by the convolution.
      norm_layer: Normalization layer constructor.
      norm_before: Whether normalization is applied before activation.

    Returns:
      A module that downsamples the input tensor.
    """

    if norm_before:
        return nn.Sequential(
            _conv3x3(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _conv3x3(in_channels, out_channels, stride, bias=True)


def _make_upsample(
    in_channels: int,
    out_channels: int,
    stride: int,
    norm_layer: Callable[[int], nn.Module],
    norm_before: bool,
) -> nn.Module:
    """Create an upsampling path for a SpineNet resampling branch.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      stride: Stride used by the subpixel convolution.
      norm_layer: Normalization layer constructor.
      norm_before: Whether normalization is applied before activation.

    Returns:
      A module that upsamples the input tensor.
    """
    if norm_before:
        return nn.Sequential(
            _subpixel_conv1x1(in_channels, out_channels, stride, bias=False),
            norm_layer(out_channels),
        )

    return _subpixel_conv1x1(in_channels, out_channels, stride, bias=True)


def _make_resample(
    channels: int,
    scale: float,
    norm_layer: Callable[[int], nn.Module],
    norm_before: bool,
    activation: Any,
    upsampling_type: str = "nearest",
) -> nn.ModuleList:
    """Create the resampling modules used inside SpineNet blocks.

    Args:
      channels: Number of channels in the resampled tensor.
      scale: Spatial scale factor relative to the target tensor.
      norm_layer: Normalization layer constructor.
      norm_before: Whether normalization is applied before activation.
      activation: Activation configuration passed to the activation factory.
      upsampling_type: Upsampling strategy for scale factors greater than one.

    Returns:
      A `ModuleList` containing the resampling modules.
    """
    resample_block = nn.ModuleList([])
    if scale > 1:
        if upsampling_type == "subpixel":
            resample_block.append(
                _make_upsample(channels, channels, scale, norm_layer, norm_before)
            )
            resample_block.append(AF.create(activation))
        elif upsampling_type == "bilinear":
            resample_block.append(Interpolate(scale_factor=scale, mode="bilinear"))
        else:
            resample_block.append(Interpolate(scale_factor=scale, mode="nearest"))

    elif scale < 1:
        resample_block.append(
            _make_downsample(channels, channels, 2, norm_layer, norm_before)
        )
        resample_block.append(AF.create(activation))

        if scale < 0.5:
            new_kernel_size = 3 if scale >= 0.25 else 5
            resample_block.append(
                nn.MaxPool2d(
                    kernel_size=new_kernel_size,
                    stride=int(0.5 / scale),
                    padding=new_kernel_size // 2,
                )
            )
    return resample_block


class SpineConv(nn.Module):
    """Project endpoint features to the final SpineNet output channels.

    Attributes:
      channels: Number of output channels.
      norm_before: Whether normalization is applied before activation.
      conv1: Projection convolution.
      bn1: Normalization layer applied after the projection.
      act1: Activation applied after the projection.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        activation: Any = {"name": "relu", "inplace": True},
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Project SpineNet endpoint features to the requested channel count.

        Args:
          in_channels: Number of channels in the input tensor.
          channels: Number of output channels.
          stride: Stride used by the projection convolution.
          dropout_rate: Kept for API compatibility.
          groups: Number of convolution groups.
          dilation: Dilation factor for the projection convolution.
          activation: Activation configuration passed to the activation factory.
          norm_layer: Normalization layer constructor.
          norm_before: Whether normalization is applied before activation.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.channels = channels
        self.norm_before = norm_before
        bias = not norm_before
        self.conv1 = _conv1x1(in_channels, channels, stride, bias=bias)
        self.bn1 = norm_layer(channels)
        self.act1 = AF.create(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection convolution and activation.

        Args:
          x: Input tensor with shape `(batch, in_channels, height, width)`.

        Returns:
          Output tensor with shape `(batch, channels, height, width)`.
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)
        return x


class BlockSpec:
    """A container class that specifies the block configuration for SpineNet.

    Attributes:
      level: Feature level for the block.
      block_fn: Block class used to construct the node.
      input_offsets: Relative offsets to the block inputs.
      is_output: Whether the block contributes to the final outputs.
    """

    def __init__(
        self, level: int, block_fn: Any, input_offsets: Sequence[int], is_output: bool
    ) -> None:
        """Store one SpineNet block specification.

        Args:
          level: Feature level for the block.
          block_fn: Block class used to construct the node.
          input_offsets: Relative offsets to the block inputs.
          is_output: Whether the block contributes to the final outputs.
        """
        self.level = level
        self.block_fn = block_fn
        self.input_offsets = input_offsets
        self.is_output = is_output

    @staticmethod
    def build_block_specs(
        block_specs: Optional[Sequence[Sequence[Any]]] = None,
    ) -> List["BlockSpec"]:
        """Build the list of `BlockSpec` objects for SpineNet.

        Args:
          block_specs: Sequence of raw block specification tuples.

        Returns:
          The parsed `BlockSpec` objects.
        """
        return [BlockSpec(*b) for b in block_specs] if block_specs is not None else []


class SpineEndpoints(nn.Module):
    """Map SpineNet block outputs to the final endpoint resolution.

    Attributes:
      in_channels: Number of channels in the input tensor.
      channels: Number of output channels.
      norm_before: Whether normalization is applied before activation.
      scale: Spatial scale factor relative to the target level.
      do_endpoint_conv: Whether to apply the endpoint projection convolution.
      upsampling_type: Upsampling strategy used during resampling.
      resample: Sequence of resampling modules applied after projection.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        level: int,
        target_level: int,
        upsampling_type: str = "nearest",
        stride: int = 1,
        activation: Any = {"name": "relu", "inplace": True},
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        do_endpoint_conv: bool = True,
    ) -> None:
        """Map a SpineNet block output to the endpoint resolution.

        Args:
          in_channels: Number of channels in the input tensor.
          channels: Number of channels in the output tensor.
          level: Level of the source feature map.
          target_level: Desired output feature-map level.
          upsampling_type: Resampling mode used when scale is greater than one.
          stride: Stride used by the endpoint projection convolution.
          activation: Activation configuration passed to the activation factory.
          norm_layer: Normalization layer constructor.
          norm_before: Whether normalization is applied before activation.
          do_endpoint_conv: Whether to apply the endpoint projection convolution.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.channels = channels
        self.norm_before = norm_before
        self.scale = 2 ** (level - target_level)
        self.do_endpoint_conv = do_endpoint_conv
        self.upsampling_type = upsampling_type
        bias = not norm_before
        if self.do_endpoint_conv and in_channels != channels:
            # in some cases this convolution is not necessary
            self.conv1 = _conv1x1(in_channels, channels, stride, bias=bias)
            self.bn1 = norm_layer(channels)
            self.act1 = AF.create(activation)

        else:
            self.channels = in_channels

        self.resample = _make_resample(
            channels,
            self.scale,
            norm_layer,
            norm_before,
            activation,
            upsampling_type=upsampling_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply endpoint projection and resampling.

        Args:
          x: Input tensor with shape `(batch, in_channels, height, width)`.

        Returns:
          Output tensor with shape `(batch, channels, height, width)`.
        """
        if self.do_endpoint_conv and self.in_channels != self.channels:
            x = self.conv1(x)
            if self.norm_before:
                x = self.bn1(x)
            x = self.act1(x)
        for mod in self.resample:
            x = mod(x)
        return x


class SpineResample(nn.Module):
    """Build a resampling connection between two SpineNet blocks.

    Attributes:
      spec: Block specification associated with this connection.
      scale: Spatial scale factor derived from the block levels.
      norm_before: Whether normalization is applied before activation.
      resample: Sequence of resampling modules applied between projections.
    """

    def __init__(
        self,
        spec: BlockSpec,
        in_channels: int,
        out_channels: int,
        scale: int,
        alpha: float,
        upsampling_type: str = "nearest",
        activation: Any = {"name": "relu", "inplace": True},
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
    ) -> None:
        """Build a resampling connection between two SpineNet blocks.

        Args:
          spec: Block specification associated with the connection.
          in_channels: Number of channels in the input tensor.
          out_channels: Number of channels in the output tensor.
          scale: Relative scale exponent between the input and output blocks.
          alpha: Channel reduction factor used before resampling.
          upsampling_type: Upsampling strategy used when scale is greater than one.
          activation: Activation configuration passed to the activation factory.
          norm_layer: Normalization layer constructor.
          norm_before: Whether normalization is applied before activation.
        """
        super().__init__()
        self.spec = spec

        in_channels_alpha = int(in_channels * alpha)
        in_channels = in_channels * spec.block_fn.expansion
        self.scale = 2**scale
        bias = not norm_before
        self.norm_before = norm_before
        if norm_layer is None:
            norm_layer = BatchNorm2d

        self.conv1 = _conv1x1(in_channels, in_channels_alpha, bias=bias)
        self.bn1 = norm_layer(in_channels_alpha)
        self.act1 = AF.create(activation)

        self.resample = _make_resample(
            in_channels_alpha,
            self.scale,
            norm_layer,
            norm_before,
            activation,
            upsampling_type=upsampling_type,
        )

        self.conv2 = _conv1x1(in_channels_alpha, out_channels, bias=bias)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the resampling branch between two blocks.

        Args:
          x: Input tensor with shape `(batch, in_channels, height, width)`.

        Returns:
          Output tensor with shape `(batch, out_channels, height, width)`.
        """
        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)

        for mod in self.resample:
            x = mod(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)
        return x
