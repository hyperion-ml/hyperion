"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ...utils.misc import filter_func_args
from ..layer_blocks.dac_blocks import DACEncoderBlock
from ..layers import Snake1d
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


class DACEncoder(NetArch):
    """
    Descript Audio Codec (DAC) Encoder.

    Architecture
    ------------
    - Stem `Conv1d` over channels-first audio.
    - A stack of `DACEncoderBlock`s; each stage downsamples by its `stride`
      and doubles the channel count.
    - Final projection `Conv1d`.

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
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        init_inner_channels: int = 64,
        kernel_size: int = 7,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
    ) -> None:
        """
        Create a DAC encoder.

        Args:
            in_feats: Number of input feature channels.
            out_feats: Number of output feature channels.
            init_inner_channels: Number of channels after the stem convolution.
            kernel_size: Kernel size used by the stem and residual blocks.
            strides: Per-stage downsampling strides. Defaults to [2, 4, 8, 8].
            dilations: Dilation rates used inside residual blocks.
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

        # Create first convolution
        self.in_conv = weight_norm(
            nn.Conv1d(
                in_feats,
                init_inner_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        )

        # Create EncoderBlocks that double channels as they downsample by `stride`
        inner_channels = init_inner_channels
        blocks = []
        for stride in strides:
            blocks += [
                DACEncoderBlock(
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
            nn.Conv1d(inner_channels, out_feats, kernel_size=3, padding=1)
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
        p = (self.kernel_size - 1) // 2
        left_context = p * 1
        right_context = p * 1
        stride = 1
        for block, s in zip(self.blocks, self.strides):
            lc, rc = block.in_context()
            left_context += lc * stride
            right_context += rc * stride
            stride *= s

        left_context += stride
        right_context += stride
        return (left_context, right_context)

    @property
    def frame_length(self) -> int:
        """Total receptive field (left + right + center sample) in samples."""
        left_context, right_context = self.in_context()
        return left_context + right_context + 1

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
        - Pads sequence length to a multiple of total stride so `forward()`
          can produce a tensor with predictable shape.

        Args:
            x: Input tensor of shape (B, in_feats, T) or (B, T).

        Returns:
            Padded tensor of shape (B, in_feats, T_pad).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        hop_length = self.stride
        length = x.shape[-1]
        right_pad = math.ceil(length / hop_length) * hop_length - length
        x = nn.functional.pad(x, (0, right_pad))

        return x

    def max_out_length(self, max_in_length: int) -> int:
        """
        Compute the tensor time length produced by `forward()` for an input length.

        This accounts for the right-padding applied by :meth:`preprocess`, so it
        predicts the allocated output shape, not the number of fully valid output
        frames.

        Args:
            max_in_length: Maximum input length in samples.

        Returns:
            Output tensor length in frames after all downsampling.
        """
        hop_length = self.stride
        max_in_length = math.ceil(max_in_length / hop_length) * hop_length
        max_out_length = max_in_length
        for block in self.blocks:
            max_out_length = block.max_out_length(max_out_length)
        return max_out_length

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute valid output lengths for a batch of input lengths.

        Unlike :meth:`max_out_length`, this excludes output frames whose receptive
        field depends on the right-padding introduced by :meth:`preprocess`.

        Args:
            in_lengths: Tensor of shape (B,) with input lengths in samples.

        Returns:
            Tensor of shape (B,) with valid output lengths in frames.
        """
        out_lengths = in_lengths
        for block in self.blocks:
            out_lengths = block.out_lengths(out_lengths)
        return out_lengths

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, Optional[int], int]:
        """
        Compute the output tensor shape given an input shape.

        Args:
            in_shape: Tuple (B, in_feats, T_in) or (B, T_in), where T_in may be `None`.

        Returns:
            Tuple (B, T_out, out_feats), where T_out may be `None`.
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
            Encoded tensor of shape (B, T_out, out_feats) and valid output lengths
            in frames.
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
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments relevant to `DACEncoder.__init__`.

        Args:
            **kwargs: Candidate keyword arguments for `DACEncoder.__init__`.

        Returns:
            dict: Filtered kwargs usable to instantiate `DACEncoder`.
        """
        return filter_func_args(DACEncoder.__init__, kwargs)

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
