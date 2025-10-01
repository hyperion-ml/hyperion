"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from ..layers import Snake1d, StreamingCausalConv1d, StreamingCausalConvTranspose1d


class DACResBlock(nn.Module):
    """
    Descript Audio Codec residual block.

    Structure:
        Snake1d → WN(Conv1d(ch, ch, kernel_size, dilation, padding)) →
        Snake1d → WN(Conv1d(ch, ch, kernel_size=1)) → residual add

    Notes:
        - Expects **channels-first** tensors of shape ``(B, C, T)``.
        - Padding keeps time length (or off-by-one which is corrected by center-cropping).

    Args:
        channels: Number of input/output channels (C).
        kernel_size: Convolution kernel size of the dilated conv (odd recommended).
        dilation: Dilation factor of the first conv.

    Shapes:
        Input:  (B, C, T)
        Output: (B, C, T)
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.layers = nn.Sequential(
            Snake1d(channels),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=pad,
                )
            ),
            Snake1d(channels),
            weight_norm(Conv1d(channels, channels, kernel_size=1)),
        )

    def in_context(self) -> int:
        """Return half-context (in samples) contributed by the dilated conv."""
        return (self.layers[1].kernel_size[0] - 1) * self.layers[1].dilation[0] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block.

        Args:
            x: Tensor of shape ``(B, C, T)``.

        Returns:
            Tensor of shape ``(B, C, T)``.
        """
        y = self.layers(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal Conv1d layers."""
        for module in self.layers.modules():
            if isinstance(module, Conv1d):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


class DACEncoderBlock(nn.Module):
    """
    Descript Audio Codec encoder block.

    Structure:
        [Residual stack @ in_channels] → Snake1d →
        WN(Conv1d(in_channels, out_channels, kernel_size=2*stride, stride=stride, padding=ceil(stride/2)))

    Purpose:
        Downsample the time axis by ``stride`` while (typically) doubling channels.

    Args:
        in_channels: Input channels (C_in).
        out_channels: Output channels (C_out).
        kernel_size: Kernel size used inside residual blocks.
        stride: Downsampling factor of the final strided conv.
        dilations: Dilation factors for residual blocks (default: [1, 3, 9]).

    Shapes:
        Input:  (B, C_in, T)
        Output: (B, C_out, T_out) with
            ``T_out = floor((T + 2*ceil(stride/2) - 2*stride) / stride) + 1``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Optional[List[int]] = None,
    ):
        if dilations is None:
            dilations = [1, 3, 9]
        super().__init__()
        self.stride = stride
        blocks = [DACResBlock(in_channels, kernel_size, d) for d in dilations]
        blocks += [
            Snake1d(in_channels),
            weight_norm(
                Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=int(math.ceil(stride / 2)),
                )
            ),
        ]
        self.blocks = nn.Sequential(*blocks)
        context = 0
        for block in blocks[:-1]:
            if isinstance(block, DACResBlock):
                context += block.in_context()

        p = int(math.ceil(stride / 2))
        self.right_context = context + p
        self.left_context = context + (2 * stride - 1 - p)

    def in_context(self) -> Tuple[int, int]:
        """Return (left_context, right_context) in input samples for one stage."""
        return (self.left_context, self.right_context)

    def max_out_length(self, in_length: int) -> int:
        """Max output length for a single example given input length ``in_length``."""
        stride = self.stride
        pad = int(math.ceil(stride / 2))
        kernel_size = 2 * stride
        return (in_length + 2 * pad - kernel_size) // stride + 1

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized version of :meth:`max_out_length` for a batch of input lengths."""
        stride = self.stride
        pad = int(math.ceil(stride / 2))
        kernel_size = 2 * stride
        return (
            torch.div(in_lengths + 2 * pad - kernel_size, stride, rounding_mode="floor")
            + 1
        )

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the encoder block.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            x_mask: Optional mask broadcastable to ``x`` (e.g., ``(B, 1, T)``).

        Returns:
            Tensor of shape ``(B, C_out, T_out)``.
        """
        if x_mask is not None:
            x = x * x_mask
        x = self.blocks(x)
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal Conv1d layers."""
        for module in self.blocks.modules():
            if isinstance(module, Conv1d):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


class DACDecoderBlock(nn.Module):
    """
    Descript Audio Codec decoder block.

    Structure:
        Snake1d → WN(ConvTranspose1d(in_ch, out_ch, kernel_size=2*stride, stride=stride, padding=ceil(stride/2)))
        → [Residual stack @ out_channels]

    Purpose:
        Upsample the time axis by ``stride`` while (typically) halving channels.

    Args:
        in_channels: Input channels (C_in).
        out_channels: Output channels (C_out).
        kernel_size: Kernel size used inside residual blocks.
        stride: Upsampling factor of the transposed convolution.
        dilations: Dilation factors for residual blocks (default: [1, 3, 9]).

    Shapes:
        Input:  (B, C_in, T)
        Output: (B, C_out, T_out) with
            ``T_out = (T - 1)*stride - 2*ceil(stride/2) + (2*stride - 1) + 1``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Optional[List[int]] = None,
    ):
        if dilations is None:
            dilations = [1, 3, 9]
        super().__init__()
        blocks = [
            Snake1d(in_channels),
            weight_norm(
                ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=int(math.ceil(stride / 2)),
                )
            ),
        ]
        blocks += [DACResBlock(out_channels, kernel_size, d) for d in dilations]
        self.blocks = nn.Sequential(*blocks)

        context = 0
        for block in blocks[1:]:
            if isinstance(block, DACResBlock):
                context += block.in_context()

        self.stride = stride
        self._context = context / stride

    def in_context(self) -> Tuple[float, float]:
        """
        Return (left_context, right_context) in **input** samples for this stage.

        Notes:
            For transposed-conv upsampling, the effective input context is fractional
            when mapped back through the stride; we expose it as floats.
        """
        return (self._context + 1, self._context)

    def max_out_length(self, in_length: int) -> int:
        """Max output length for a single example given input length ``in_length``."""
        stride = self.stride
        pad = int(math.ceil(stride / 2))
        kernel_size = 2 * stride
        return (in_length - 1) * stride - 2 * pad + (kernel_size - 1) + 1

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized version of :meth:`max_out_length` for a batch of input lengths."""
        stride = self.stride
        pad = int(math.ceil(stride / 2))
        kernel_size = 2 * stride
        return (in_lengths - 1) * stride - 2 * pad + (kernel_size - 1) + 1

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the decoder block.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            x_mask: Optional mask broadcastable to ``x`` (e.g., ``(B, 1, T)``).

        Returns:
            Tensor of shape ``(B, C_out, T_out)``.
        """
        if x_mask is not None:
            x = x * x_mask
        x = self.blocks(x)
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal conv / transposed-conv layers."""
        for module in self.blocks.modules():
            if isinstance(module, (Conv1d, ConvTranspose1d)):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


class StreamingDACResBlock(nn.Module):
    """
    Streaming Descript Audio Codec residual block.

    Structure:
        Snake1d → WN(Conv1d(ch, ch, kernel_size, dilation, padding)) →
        Snake1d → WN(Conv1d(ch, ch, kernel_size=1)) → residual add

    Notes:
        - Expects **channels-first** tensors of shape ``(B, C, T)``.
        - Padding keeps time length (or off-by-one which is corrected by center-cropping).

    Args:
        channels: Number of input/output channels (C).
        kernel_size: Convolution kernel size of the dilated conv (odd recommended).
        dilation: Dilation factor of the first conv.

    Shapes:
        Input:  (B, C, T)
        Output: (B, C, T)
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            Snake1d(channels),
            weight_norm(
                StreamingCausalConv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            ),
            Snake1d(channels),
            weight_norm(Conv1d(channels, channels, kernel_size=1)),
        )

    def in_context(self) -> int:
        """Return  contributed by the dilated conv."""
        return ((self.layers[1].kernel_size[0] - 1) * self.layers[1].dilation[0], 0)

    def max_out_length(self, in_length: int) -> int:
        """Max output length for a single example given input length ``in_length``."""
        kernel_size = self.layers[1].kernel_size[0]
        dilation = self.layers[1].dilation[0]
        return in_length - (kernel_size - 1) * dilation

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized version of :meth:`max_out_length` for a batch of input lengths."""
        kernel_size = self.layers[1].kernel_size[0]
        dilation = self.layers[1].dilation[0]
        return in_lengths - (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block.

        Args:
            x: Tensor of shape ``(B, C, T)``.

        Returns:
            Tensor of shape ``(B, C, T)``.
        """
        y = self.layers(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            x = x[..., pad:]
        return x + y

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal Conv1d layers."""
        for module in self.layers.modules():
            if isinstance(module, Conv1d):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


class StreamingDACEncoderBlock(nn.Module):
    """
    Streaming Descript Audio Codec encoder block.

    Structure:
        [Residual stack @ in_channels] → Snake1d →
        WN(Conv1d(in_channels, out_channels, kernel_size=2*stride, stride=stride, padding=ceil(stride/2)))

    Purpose:
        Downsample the time axis by ``stride`` while (typically) doubling channels.

    Args:
        in_channels: Input channels (C_in).
        out_channels: Output channels (C_out).
        kernel_size: Kernel size used inside residual blocks.
        stride: Downsampling factor of the final strided conv.
        dilations: Dilation factors for residual blocks (default: [1, 3, 9]).

    Shapes:
        Input:  (B, C_in, T)
        Output: (B, C_out, T_out) with
            ``T_out = floor((T + 2*ceil(stride/2) - 2*stride) / stride) + 1``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Optional[List[int]] = None,
    ):
        if dilations is None:
            dilations = [1, 3, 9]
        super().__init__()
        self.stride = stride
        blocks = [StreamingDACResBlock(in_channels, kernel_size, d) for d in dilations]
        blocks += [
            Snake1d(in_channels),
            weight_norm(
                StreamingCausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                )
            ),
        ]
        self.blocks = nn.Sequential(*blocks)
        left_context = 0
        right_context = 0
        for block in blocks[:-1]:
            if isinstance(block, DACResBlock):
                lc, rc = block.in_context()
                left_context += lc
                right_context += rc

        self.right_context = right_context
        self.left_context = left_context + (2 * stride - 1)

    def in_context(self) -> Tuple[int, int]:
        """Return (left_context, right_context) in input samples for one stage."""
        return (self.left_context, self.right_context)

    def max_out_length(self, in_length: int) -> int:
        """Max output length for a single example given input length ``in_length``."""
        for block in self.blocks[:-2]:
            in_length = block.max_out_length(in_length)

        stride = self.stride
        kernel_size = 2 * stride
        return (in_length - kernel_size) // stride + 1

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized version of :meth:`max_out_length` for a batch of input lengths."""
        for block in self.blocks[:-2]:
            in_lengths = block.out_lengths(in_lengths)
        stride = self.stride
        kernel_size = 2 * stride
        return torch.div(in_lengths - kernel_size, stride, rounding_mode="floor") + 1

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the encoder block.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            x_mask: Optional mask broadcastable to ``x`` (e.g., ``(B, 1, T)``).

        Returns:
            Tensor of shape ``(B, C_out, T_out)``.
        """
        if x_mask is not None:
            x = x * x_mask
        x = self.blocks(x)
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal Conv1d layers."""
        for module in self.blocks.modules():
            if isinstance(module, Conv1d):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


class StreamingDACDecoderBlock(nn.Module):
    """
    Streaming Descript Audio Codec decoder block.

    Structure:
        Snake1d → WN(ConvTranspose1d(in_ch, out_ch, kernel_size=2*stride, stride=stride, padding=ceil(stride/2)))
        → [Residual stack @ out_channels]

    Purpose:
        Upsample the time axis by ``stride`` while (typically) halving channels.

    Args:
        in_channels: Input channels (C_in).
        out_channels: Output channels (C_out).
        kernel_size: Kernel size used inside residual blocks.
        stride: Upsampling factor of the transposed convolution.
        dilations: Dilation factors for residual blocks (default: [1, 3, 9]).

    Shapes:
        Input:  (B, C_in, T)
        Output: (B, C_out, T_out) with
            ``T_out = (T - 1)*stride - 2*ceil(stride/2) + (2*stride - 1) + 1``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilations: Optional[List[int]] = None,
    ):
        if dilations is None:
            dilations = [1, 3, 9]
        super().__init__()
        blocks = [
            Snake1d(in_channels),
            weight_norm(
                StreamingCausalConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                )
            ),
        ]
        blocks += [
            StreamingDACResBlock(out_channels, kernel_size, d) for d in dilations
        ]
        self.blocks = nn.Sequential(*blocks)

        left_context = 0
        right_context = 0
        for block in blocks[:-1]:
            if isinstance(block, DACResBlock):
                lc, rc = block.in_context()
                left_context += lc
                right_context += rc

        self.left_context = left_context + 1
        self.right_context = right_context
        self.stride = stride

    def in_context(self) -> Tuple[float, float]:
        """
        Return (left_context, right_context) in **input** samples for this stage.

        Notes:
            For transposed-conv upsampling, the effective input context is fractional
            when mapped back through the stride; we expose it as floats.
        """
        return (self.left_context, self.right_context)

    def max_out_length(self, in_length: int) -> int:
        """Max output length for a single example given input length ``in_length``."""
        stride = self.stride
        kernel_size = 2 * stride
        out_length = (in_length - 1) * stride + (kernel_size - 1) + 1
        for block in self.blocks[2:]:
            out_length = block.max_out_length(out_length)
        return out_length

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized version of :meth:`max_out_length` for a batch of input lengths."""
        stride = self.stride
        kernel_size = 2 * stride
        out_lengths = (in_lengths - 1) * stride + (kernel_size - 1) + 1
        for block in self.blocks[2:]:
            out_lengths = block.out_lengths(out_lengths)
        return out_lengths

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the decoder block.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            x_mask: Optional mask broadcastable to ``x`` (e.g., ``(B, 1, T)``).

        Returns:
            Tensor of shape ``(B, C_out, T_out)``.
        """
        if x_mask is not None:
            x = x * x_mask
        x = self.blocks(x)
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal conv / transposed-conv layers."""
        for module in self.blocks.modules():
            if isinstance(module, (Conv1d, ConvTranspose1d)):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass
