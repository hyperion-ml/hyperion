"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ...utils import HyperDataClass
from ...utils.misc import filter_func_args
from ..layer_blocks.dac_blocks import StreamingDACEncoderBlock
from ..layers import Snake1d, StreamingCausalConv1d
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


@dataclass
class StreamingDACEncoderState(HyperDataClass):
    """Aggregated cache state for `StreamingDACEncoder`.

    Attributes:
        in_conv_state: Cache state of the input convolution.
        out_conv_state: Cache state of the output convolution.
        block_states: List of per-block cache states mirroring the encoder layout.
    """

    in_conv_state: Dict[str, Any] = field(default_factory=dict)
    block_states: List[Dict[str, Any]] = field(default_factory=list)
    out_conv_state: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of cached state components."""
        return len(self.block_states) + 2  # in_conv + out_conv


class StreamingDACEncoder(NetArch):
    """
    Streaming Descript Audio Codec (DAC) Encoder.

    Architecture
    ------------
    - Stem `Streaming Conv1d`
    - A stack of `StreamingDACEncoderBlock`s; each stage downsamples by its `stride`
      and doubles the channel count.
    - Final projection `Streaming Conv1d`.

    Shapes
    ------
    Input:
        x: (B, in_feats, T) or (B, T) when in_feats == 1

    Output:
        y: (B, T_out, out_feats),
           with approximately T_out ≈ T / prod(strides)
           (exact length depends on padding)

    Attributes:
        in_feats:  Number of input feature channels.
        out_feats: Number of output feature channels.
        init_inner_channels: Channels after the stem convolution.
        kernel_size: Kernel size for residual blocks.
        strides: Per-stage downsampling strides. If None, defaults to [2, 4, 8, 8].
        dilations: Dilations used inside each residual block. If None, defaults to [1, 3, 9].
        look_aheads: place holder for future use, if needed.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        init_inner_channels: int = 64,
        kernel_size: int = 7,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        look_aheads: Optional[List[int]] = None,
    ) -> None:
        """
        Create a streaming DAC encoder.

        Args:
            in_feats: Number of input feature channels.
            out_feats: Number of output feature channels.
            init_inner_channels: Channels after the stem convolution.
            kernel_size: Kernel size for residual blocks.
            strides: Per-stage downsampling strides.
            dilations: Dilations used inside each residual block.
            look_aheads: Placeholder for future use, if needed.
        """
        if strides is None:
            strides = [2, 4, 8, 8]
        if dilations is None:
            dilations = [1, 3, 9]
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.init_inner_channels = init_inner_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.stride = math.prod(self.strides)
        self.look_aheads = look_aheads

        # Create first convolution
        self.in_conv = weight_norm(
            StreamingCausalConv1d(
                in_feats,
                init_inner_channels,
                kernel_size=kernel_size,
            )
        )

        # Create EncoderBlocks that double channels as they downsample by `stride`
        inner_channels = init_inner_channels
        blocks = []
        for stride in strides:
            blocks += [
                StreamingDACEncoderBlock(
                    inner_channels,
                    inner_channels * 2,
                    stride=stride,
                    kernel_size=kernel_size,
                    dilations=dilations,
                )
            ]
            inner_channels = inner_channels * 2

        self.blocks = nn.ModuleList(blocks)

        # Create last convolution
        self.out_act = Snake1d(inner_channels)
        self.out_conv = weight_norm(
            StreamingCausalConv1d(inner_channels, out_feats, kernel_size=3)
        )
        self.init_weights()

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """
        Return constructor configuration merged with `NetArch` base config.

        Args:
            no_class_name: If True, omit the class name entry from the config.

        Returns:
            dict: A JSON-serializable configuration dictionary.
        """
        config = {
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "init_inner_channels": self.init_inner_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilations": self.dilations,
            "look_aheads": self.look_aheads,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        config.update(base_config)
        return config

    def in_context(self) -> Tuple[int, int]:
        """
        Compute the receptive-field context of the encoder.

        Returns:
            (left_context, right_context) in samples.
        """
        left_context = self.kernel_size - 1
        right_context = 0
        stride = 1
        for block, s in zip(self.blocks, self.strides):
            lc, rc = block.in_context()
            left_context += lc * stride
            right_context += rc * stride
            stride *= s

        left_context += 2 * stride
        return (left_context, right_context)

    @property
    def frame_length(self) -> int:
        """Total receptive field (left + right + center sample) in samples."""
        left_context, right_context = self.in_context()
        return left_context + right_context + 1

    @property
    def frame_shift(self) -> int:
        """Total downsampling factor (frame shift) in samples."""
        return self.stride

    @property
    def hop_size(self) -> int:
        """Total downsampling factor (hop size) in samples."""
        return self.stride

    def init_weights(self) -> None:
        """
        Initialize convolutional weights with N(0, 0.01) and zero biases.

        If weight norm is applied, initialize the underlying parametrized weight.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # If parametrized (e.g., weight_norm), init the original weight
                if (
                    is_parametrized(m)
                    and hasattr(m, "parametrizations")
                    and "weight" in m.parametrizations
                ):
                    g = m.parametrizations.weight.original0
                    v = m.parametrizations.weight.original1
                    nn.init.normal_(v, 0.0, 0.01)
                    with torch.no_grad():
                        g.copy_(v.flatten(1).norm(dim=1, keepdim=True).view_as(g))
                else:
                    w = m.weight
                    nn.init.normal_(w, 0.0, 0.01)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for the encoder.

        - Ensures channels-first layout (B, in_feats, T).
        - Pads sequence length to a multiple of total stride.

        Args:
            x: Input tensor of shape (B, in_feats, T) or (B, T).

        Returns:
            Padded tensor of shape (B, in_feats, T_pad).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        hop_size = self.stride
        length = x.shape[-1]
        right_pad = math.ceil(length / hop_size) * hop_size - length
        x = nn.functional.pad(x, (0, right_pad))

        return x

    def max_out_length(self, max_in_length: int) -> int:
        """
        Compute maximum output length for a given input length.

        Args:
            max_in_length: Maximum input length in samples.

        Returns:
            Maximum output length in samples after all downsampling.
        """
        hop_length = self.stride
        max_in_length = int(math.ceil(max_in_length / hop_length) * hop_length)
        max_out_length = max_in_length
        for block in self.blocks:
            max_out_length = block.max_out_length(max_out_length)

        return max_out_length

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths for a batch of input lengths.

        Args:
            in_lengths: Tensor of shape (B,) with input lengths in samples.

        Returns:
            Tensor of shape (B,) with output lengths in samples.
        """
        out_lengths = in_lengths
        out_lengths = out_lengths - (self.kernel_size - 1)
        for block in self.blocks:
            out_lengths = block.out_lengths(out_lengths)

        out_lengths = out_lengths - 2  # last conv reduces 2 samples
        return out_lengths.clamp(min=0)

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output tensor shape given an input shape.

        Args:
            in_shape: Tuple (B, in_feats, T_in).

        Returns:
            Tuple (B, T_out, out_feats).
        """

        B = in_shape[0]
        T = in_shape[-1]
        if T is None:
            return (B, None, self.out_feats)
        else:
            out_length = self.max_out_length(T)
            return (B, out_length, self.out_feats)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (B, in_feats, T) or (B, T).
            x_lengths: Optional tensor of shape (B,) with valid lengths.

        Returns:
            Tensor of shape (B, T_out, out_feats) and valid output lengths.
        """

        x = self.preprocess(x)
        if x_lengths is not None:
            x_mask = seq_lengths_to_mask(x_lengths, x.size(2), time_dim=2).to(x.dtype)
            x = x * x_mask

        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)

        x = self.out_act(x)
        z = self.out_conv(x)
        z_lengths = self.out_lengths(x_lengths) if x_lengths is not None else None
        return z.transpose(1, 2).contiguous(), z_lengths

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StreamingDACEncoderState:
        """
        Initialize the states of all encoder blocks for streaming inference.

        Args:
            batch_size: Batch size for which to initialize states.
            device: Device on which to create the states.
            dtype: Data type of the states.

        Returns:
            StreamingDACEncoderState: Initial streaming state.
        """
        in_conv_state = self.in_conv.init_state(batch_size, device=device, dtype=dtype)
        out_conv_state = self.out_conv.init_state(
            batch_size, device=device, dtype=dtype
        )
        block_states = []
        for block in self.blocks:
            block_states.append(
                block.init_state(batch_size, device=device, dtype=dtype)
            )

        return StreamingDACEncoderState(
            in_conv_state=in_conv_state,
            block_states=block_states,
            out_conv_state=out_conv_state,
        )

    def stream_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for the encoder.

        - Ensures channels-first layout (B, in_feats, T).

        Args:
            x: Input tensor of shape (B, in_feats, T) or (B, T).

        Returns:
            Tensor of shape (B, in_feats, T).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        return x

    @torch.no_grad()
    def stream(
        self,
        x: torch.Tensor,
        state: StreamingDACEncoderState,
        flush: bool = False,
    ) -> Tuple[torch.Tensor, StreamingDACEncoderState]:
        """
        Streaming forward pass of the encoder.

        Args:
            x: Input tensor of shape (B, in_feats, T) or (B, T).
            state: Aggregated encoder state.
            flush: If True, flush buffered look-ahead in the final chunk.

        Returns:
            y: Output tensor of shape (B, T_out, out_feats).
            new_state: Updated encoder state.
        """
        x = self.stream_preprocess(x)
        x, new_in_conv_state = self.in_conv.stream(x, state.in_conv_state, flush=flush)

        new_block_states = []
        for block, block_state in zip(self.blocks, state.block_states):
            x, new_block_state = block.stream(x, block_state, flush=flush)
            new_block_states.append(new_block_state)

        x = self.out_act(x)
        x, new_out_conv_state = self.out_conv.stream(
            x, state.out_conv_state, flush=flush
        )

        new_state = StreamingDACEncoderState(
            in_conv_state=new_in_conv_state,
            block_states=new_block_states,
            out_conv_state=new_out_conv_state,
        )

        return x.transpose(1, 2).contiguous(), new_state

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization from all layers (useful for inference export).
        """
        logging.info("Removing weight norm...")
        for m in [self.in_conv, self.out_conv]:
            try:
                remove_parametrizations(m, "weight")
            except ValueError:
                pass  # already removed or not parametrized

        for block in self.blocks:
            block.remove_weight_norm()

    @staticmethod
    def filter_args(**kwargs) -> Dict[str, Any]:
        """
        Filter keyword arguments relevant to `StreamingDACEncoder.__init__`.

        Args:
            kwargs: Keyword arguments to filter.

        Returns:
            dict: Filtered kwargs usable to instantiate `StreamingDACEncoder`.
        """
        return filter_func_args(StreamingDACEncoder.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ) -> None:
        """
        Register encoder hyperparameters on a CLI parser.

        Args:
            parser: The (outer) `ArgumentParser` to extend.
            prefix: If provided, arguments are grouped under this nested parser flag.
            skip: Set of parameter names to omit (e.g., {"in_feats"}).
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats", type=int, default=1, help="Input feature channels."
            )
        if "out_feats" not in skip:
            parser.add_argument(
                "--out-feats",
                type=int,
                default=1,
                help="Output feature channels (e.g., 1 for mono audio).",
            )
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=7,
            help="Residual block kernel sizes.",
        )
        parser.add_argument(
            "--init-inner-channels",
            type=int,
            default=64,
            help="Initial number of internal channels.",
        )
        parser.add_argument(
            "--dilations",
            type=int,
            nargs="+",
            default=[1, 3, 9],
            help="Dilation rates for residual blocks.",
        )
        parser.add_argument(
            "--strides",
            type=int,
            nargs="+",
            default=[2, 4, 8, 8],
            help="Upsample stride sizes.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


def stream_dac_encoder_demo(
    B: int = 1,
    Cout: int = 1,
    init_inner_channels: int = 4,
    kernel_size: int = 5,
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    T: int = 2408,
    chunk: int = 512,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compare full forward vs streaming for StreamingDACEncoder.

    Args:
        B: Batch size.
        Cout: Output feature channels.
        init_inner_channels: Initial internal channel count.
        kernel_size: Residual block kernel size.
        strides: Optional per-stage downsampling strides.
        dilations: Optional residual-block dilation rates.
        T: Total input length in samples.
        chunk: Streaming chunk length in samples.
        device: Device on which to run the demo.
        dtype: Tensor dtype to use.

    Returns:
        Tuple `(y_stream, y_ref)` with streaming and reference outputs.
    """
    torch.manual_seed(0)
    enc = StreamingDACEncoder(
        in_feats=1,
        out_feats=Cout,
        init_inner_channels=init_inner_channels,
        kernel_size=kernel_size,
        strides=strides,
        dilations=dilations,
    ).to(device=device, dtype=dtype)

    x_full = torch.randn(B, T, device=device, dtype=dtype)
    y_ref, _ = enc(x_full)

    state = enc.init_state(B, device=device, dtype=dtype)
    outs = []
    t = 0
    while t < T:
        x_chunk = x_full[:, t : t + chunk]
        flush = (t + x_chunk.size(1)) >= T
        y_emit, state = enc.stream(x_chunk, state, flush=flush)
        outs.append(y_emit)
        t += x_chunk.size(1)

    y_stream = torch.cat(outs, dim=1)
    print(f"y_ref={y_ref}")
    print(f"y_stream={y_stream}")
    print(f"y_ref.shape={y_ref.shape}, y_stream.shape={y_stream.shape}")
    assert y_ref.shape == y_stream.shape, f"{y_ref.shape=} {y_stream.shape=}"
    atol = 1e-5 if dtype == torch.float32 else 5e-4
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    max_abs = (y_ref - y_stream).abs().max().item()
    assert torch.allclose(y_ref, y_stream, atol=atol, rtol=rtol), f"max_abs={max_abs}"
    return y_stream, y_ref
