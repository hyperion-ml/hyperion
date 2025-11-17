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
from ..layer_blocks.dac_blocks import DACDecoderBlock
from ..layers import Snake1d
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


class DACDecoder(NetArch):
    """
    Descript Audio Codec (DAC) Decoder.

    Architecture:
        A stem Conv1d, followed by a sequence of upsampling `DACDecoderBlock`s
        that typically **halve** channel width at each stage, and a final
        projection Conv1d with optional tanh output squash.

    Shapes:
        Input:  (B, T_in, in_feats)
        Output: (B, T_out, out_feats) where T_out ≈ T_in * prod(strides)
                (exact length depends on padding/transpose-conv details)

    Args:
        in_feats: Number of input feature channels.
        out_feats: Number of output feature channels.
        init_inner_channels: Channels after the stem convolution.
        kernel_size: Kernel size for stem/final convs (assumed odd if using same padding).
        strides: Per-stage upsampling strides. Defaults to [8, 8, 4, 2] if None.
        dilations: Dilations used inside each residual block. Defaults to [1, 3, 9] if None.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        init_inner_channels: int = 1536,
        kernel_size: int = 7,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
    ):
        if strides is None:
            strides = [8, 8, 4, 2]
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

        # Create DecoderBlocks that halves channels as they upsample by `stride`
        inner_channels = init_inner_channels
        blocks = []
        for stride in strides:
            blocks += [
                DACDecoderBlock(
                    inner_channels,
                    inner_channels // 2,
                    stride=stride,
                    kernel_size=kernel_size,
                    dilations=dilations,
                )
            ]
            inner_channels = inner_channels // 2

        self.blocks = nn.ModuleList(blocks)

        # Create last convolution
        self.out_act = Snake1d(inner_channels)
        self.out_conv = weight_norm(
            nn.Conv1d(
                inner_channels,
                out_feats,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        )
        self.init_weights()

    def get_config(self, no_class_name: bool = False):
        """
        Return constructor configuration merged with `NetArch` base config.

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
        Returns the input context (left, right) in samples.

        Notes:
            This is the number of input samples that affect each output sample.
            For example, a context of (2, 3) means that to compute each output
            sample y[t], the model needs access to x[t-2:t+3] (5 input samples).
        """
        p = (self.kernel_size - 1) // 2
        left_context = p * 1
        right_context = p * 1
        stride = 1
        for block, s in zip(self.blocks, self.strides):
            lc, rc = block.in_context()
            left_context = left_context + lc / stride
            right_context = right_context + rc / stride
            stride *= s

        left_context = int(math.ceil(left_context + p / stride))
        right_context = int(math.ceil(right_context + p / stride))
        return (left_context, right_context)

    def max_out_length(self, max_in_length: int) -> int:
        """
        Returns the maximum output length given an input length.

        Args:
            max_in_length (int): Maximum input length in samples.

        Returns:
            int: Maximum output length in samples.
        """
        max_out_length = max_in_length
        for block in self.blocks:
            max_out_length = block.max_out_length(max_out_length)
        return max_out_length

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns the output lengths given input lengths.
        Args:
            in_lengths (torch.Tensor): Input lengths in samples.

        Returns:
            torch.Tensor: Output lengths in frames.
        """
        out_lengths = in_lengths
        for block in self.blocks:
            out_lengths = block.out_lengths(out_lengths)
        return out_lengths

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        B = in_shape[0]
        T = in_shape[1]
        if T is None:
            return (B, None, self.out_feats)
        else:
            out_length = self.max_out_length(T)
            return (B, out_length, self.out_feats)

    def init_weights(self):
        """
        Initialize convolutional weights with N(0, 0.01) and zero biases.

        Notes:
            If a layer is parametrized (e.g., `weight_norm`), initialize the
            underlying `weight` parameter (`parametrizations.weight.original`).
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

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a time-channel sequence.

        Args:
            x: Input tensor of shape (B, T, in_feats).
            x_lengths: Optional valid lengths per batch element (B,).

        Returns:
            Tensor of shape (B, T', out_feats), where T' depends on strides/padding.
        """
        x = x.transpose(1, 2).contiguous()  # (B, T, in_feats) -> (B, in_feats, T)

        if x_lengths is not None:
            x_mask = seq_lengths_to_mask(x_lengths, x.size(2), time_dim=2).to(x.dtype)
            x = x * x_mask

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_act(x)
        x = self.out_conv(x)
        x = torch.tanh(x)
        return x

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
        Filter keyword arguments relevant to `DACDecoder.__init__`.

        Returns:
            dict: Filtered kwargs usable to instantiate `DACDecoder`.
        """
        return filter_func_args(DACDecoder.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set = set()
    ) -> None:
        """
        Register Decoder hyperparameters on a CLI parser.

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
            "--dilations",
            type=int,
            nargs="+",
            default=[1, 3, 9],
            help="Dilation rates for residual blocks.",
        )
        parser.add_argument(
            "--init-inner-channels",
            type=int,
            default=1536,
            help="Initial number of internal channels.",
        )
        parser.add_argument(
            "--strides",
            type=int,
            nargs="+",
            default=[8, 8, 4, 2],
            help="Upsample stride sizes.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
