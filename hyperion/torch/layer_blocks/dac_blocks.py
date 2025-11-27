"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

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

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Initialize internal states for streaming inference.

        Args:
            batch_size: Batch size.
            device: Device where the tensors are allocated.
        """
        for module in self.layers.modules():
            if isinstance(module, StreamingCausalConv1d):
                return module.init_state(batch_size, device=device, dtype=dtype)

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

    @torch.no_grad()
    def stream(
        self,
        x: torch.Tensor,
        state: Dict[str, Any],
        flush: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply the residual block in streaming mode.

        Args:
            x: Tensor of shape ``(B, C, T)``.
            states: Dictionary containing internal states.
            flush: If True, flush the inner streaming conv (end of stream).

        Returns:
            Tuple of:
            - Tensor of shape ``(B, C, T)``.
            - Updated states dictionary.
        """
        y = x.clone()
        for i, layer in enumerate(self.layers):
            if i == 1:
                y, new_state = layer.stream(y, state, flush=flush)
            else:
                y = layer(y)

        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            x = x[..., pad:]
        return x + y, new_state

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

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, Any]]:
        """Initialize internal states for streaming inference.

        Args:
            batch_size: Batch size.
            device: Device where the tensors are allocated.
        """
        state = []
        for layer in self.blocks:
            if isinstance(layer, (StreamingCausalConv1d, StreamingDACResBlock)):
                state_module = layer.init_state(batch_size, device=device, dtype=dtype)
                state.append(state_module)

        return state

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

    @torch.no_grad()
    def stream(
        self,
        x: torch.Tensor,
        state: List[Dict[str, Any]],
        flush: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """a
        Apply the encoder block in streaming mode.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            state: List of per-layer states.
            flush: If True, flush the final streaming conv (end of stream).
        """
        act_idx = len(self.blocks) - 2
        state_cur_idx = 0
        for i, block in enumerate(self.blocks):
            if i == act_idx:
                x = block(x)
            else:
                x, new_state = block.stream(x, state[state_cur_idx], flush=flush)
                state[state_cur_idx] = new_state
                state_cur_idx += 1

        return x, state

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

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, Any]]:
        """Initialize internal states for streaming inference.

        Args:
            batch_size: Batch size.
            device: Device where the tensors are allocated.
        """
        state = []
        for layer in self.blocks:
            if isinstance(
                layer, (StreamingCausalConvTranspose1d, StreamingDACResBlock)
            ):
                state_module = layer.init_state(batch_size, device=device, dtype=dtype)
                state.append(state_module)

        return state

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

    @torch.no_grad()
    def stream(
        self,
        x: torch.Tensor,
        state: List[Dict[str, Any]],
        flush: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply the decoder block in streaming mode.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.
            states: List containing internal states.
            flush: If True, flush the first streaming deconv (end of stream).

        Returns:
            Tuple of:
            - Tensor of shape ``(B, C_out, T_out)``.
            - Updated states list.
        """
        act_idx = 0
        state_cur_idx = 0
        for i, block in enumerate(self.blocks):
            if i == act_idx:
                x = block(x)
            else:
                x, new_state = block.stream(x, state[state_cur_idx], flush=flush)
                state[state_cur_idx] = new_state
                state_cur_idx += 1

        return x, state

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from internal conv / transposed-conv layers."""
        for module in self.blocks.modules():
            if isinstance(module, (Conv1d, ConvTranspose1d)):
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    pass


def stream_dac_resblock_demo(B=1, C=8, T=480, chunk=160, kernel_size=7, dilation=1):
    """Compare forward vs streaming for DACResBlock."""
    torch.manual_seed(0)
    block = StreamingDACResBlock(C, kernel_size=kernel_size, dilation=dilation).eval()
    x_full = torch.randn(B, C, T)
    y_ref = block(x_full)

    state = block.init_state(B, device=x_full.device, dtype=x_full.dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = (t + x_chunk.size(-1)) >= T
        y_emit, state = block.stream(x_chunk, state, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)
    y_stream = torch.cat(outs, dim=-1)
    assert y_ref.shape == y_stream.shape
    assert torch.allclose(y_ref, y_stream, atol=1e-5, rtol=1e-4)
    return y_stream, y_ref


def stream_dac_encoder_demo(
    B=1, Cin=8, Cout=16, T=480, chunk=160, kernel_size=7, stride=2, dilations=None
):
    """Compare forward vs streaming for StreamingDACEncoderBlock."""
    torch.manual_seed(0)
    enc = StreamingDACEncoderBlock(
        Cin, Cout, kernel_size=kernel_size, stride=stride, dilations=dilations
    ).eval()
    x_full = torch.randn(B, Cin, T)
    y_ref = enc(x_full)

    states = enc.init_state(B, x_full.device, x_full.dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = (t + x_chunk.size(-1)) >= T
        y_emit, states = enc.stream(x_chunk, states, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)
    y_stream = torch.cat(outs, dim=-1)
    assert y_ref.shape == y_stream.shape
    assert torch.allclose(y_ref, y_stream, atol=1e-5, rtol=1e-4)
    return y_stream, y_ref


def stream_dac_decoder_demo(
    B=1, Cin=16, Cout=8, T=120, chunk=40, kernel_size=7, stride=2, dilations=None
):
    """Compare forward vs streaming for StreamingDACDecoderBlock."""
    torch.manual_seed(0)
    dec = StreamingDACDecoderBlock(
        Cin, Cout, kernel_size=kernel_size, stride=stride, dilations=dilations
    ).eval()
    x_full = torch.randn(B, Cin, T)
    y_ref = dec(x_full)

    states = dec.init_state(B, x_full.device, x_full.dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = (t + x_chunk.size(-1)) >= T
        y_emit, states = dec.stream(x_chunk, states, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)
    y_stream = torch.cat(outs, dim=-1)
    assert y_ref.shape == y_stream.shape
    assert torch.allclose(y_ref, y_stream, atol=1e-5, rtol=1e-4)
    return y_stream, y_ref


def stream_dac_pair_demo(
    B=1,
    Cin=1,
    Cmid=8,
    Cout=1,
    T=640,
    chunk=160,
    k=7,
    s=2,
    dilations=None,
):
    """
    Compare full forward vs streaming of encoder+decoder cascade.
    """
    torch.manual_seed(0)
    enc = StreamingDACEncoderBlock(
        Cin, Cmid, kernel_size=k, stride=s, dilations=dilations
    ).eval()
    dec = StreamingDACDecoderBlock(
        Cmid, Cout, kernel_size=k, stride=s, dilations=dilations
    ).eval()

    x_full = torch.randn(B, Cin, T)
    y_ref = dec(enc(x_full))

    st_enc = enc.init_state(B, x_full.device, x_full.dtype)
    st_dec = dec.init_state(B, x_full.device, x_full.dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[..., t : t + chunk]
        flush = (t + x_chunk.size(-1)) >= T
        z_emit, st_enc = enc.stream(x_chunk, st_enc, flush=flush)
        y_emit, st_dec = dec.stream(z_emit, st_dec, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(-1)

    y_stream = torch.cat(outs, dim=-1)
    assert y_ref.shape == y_stream.shape
    assert torch.allclose(y_ref, y_stream, atol=1e-5, rtol=1e-4)
    return y_stream, y_ref
