"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ...utils.misc import filter_func_args
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


@torch.jit.script
def add_tanh_sigmoid_multiply(
    x: torch.Tensor, y: torch.Tensor, num_channels: int
) -> torch.Tensor:
    """
    Fused element-wise addition, gated tanh-sigmoid activation, and multiplication.

    This function adds two input tensors, applies tanh activation to the first half
    of the channels, sigmoid to the second half, and multiplies the two results.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T).
        y (torch.Tensor): Input tensor of shape (B, C, T), typically the conditioning signal.
        num_channels (int): Number of channels in the first half (for tanh and gating).

    Returns:
        torch.Tensor: Output tensor after element-wise gated activation.
    """
    z = x + y
    z_tanh = torch.tanh(z[:, :num_channels, :])
    z_sig = torch.sigmoid(z[:, num_channels:, :])
    output = z_tanh * z_sig
    return output


class WaveNet(NetArch):
    """
    Dilated convolutional WaveNet-style module with gated activation units and
    optional local conditioning, used in flow-based and autoregressive audio models.

    Attributes:
        num_layers (int): Number of dilated convolution layers.
        hidden_channels (int): Number of channels in the hidden layers.
        kernel_size (Tuple[int]): Size of the convolutional kernel stored as a 1-tuple (must be odd).
        dilation_rate (int): Dilation base (layer i uses dilation_rate ** i).
        cond_channels (int): Number of conditioning input channels. Default is 0 (no conditioning).
        dropout_rate (float): Dropout rate applied after gated activations. Default is 0.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        cond_channels: int = 0,
        dropout_rate: float = 0,
    ) -> None:
        """
        Initialize a WaveNet stack.

        Args:
            num_layers (int): Number of dilated convolution layers.
            hidden_channels (int): Number of channels in the residual stack.
            kernel_size (int): Size of each convolutional kernel. Must be odd.
            dilation_rate (int): Dilation base used for layer ``i`` as ``dilation_rate ** i``.
            cond_channels (int): Number of conditioning channels. Set to 0 to disable conditioning.
            dropout_rate (float): Dropout probability applied after gated activations.
        """
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels
        self.dropout_rate = dropout_rate

        self.dconv_layers = torch.nn.ModuleList()
        self.pwconv_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        if cond_channels != 0:
            cond_layer = nn.Conv1d(cond_channels, 2 * hidden_channels * num_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(num_layers):
            dilation = dilation_rate**i
            padding = dilation * (kernel_size - 1) // 2
            layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            layer = weight_norm(layer, name="weight")
            self.dconv_layers.append(layer)

            # last one is not necessary
            if i < num_layers - 1:
                pwconv_channels = 2 * hidden_channels
            else:
                pwconv_channels = hidden_channels

            pwconv_layer = nn.Conv1d(hidden_channels, pwconv_channels, 1)
            pwconv_layer = weight_norm(pwconv_layer, name="weight")
            self.pwconv_layers.append(pwconv_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the WaveNet module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B = batch, C = channels, T = time steps.
            x_lengths (torch.Tensor, optional): Lengths of each sequence for masking. Used to infer x_mask if not provided.
            x_mask (torch.Tensor, optional): Mask tensor of shape (B, 1, T), used to zero out padding areas.
            condition (torch.Tensor, optional): Optional local conditioning tensor of shape (B, Cc, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        if x_mask is None:
            if x_lengths is None:
                raise ValueError("x_lengths must be provided when x_mask is None")
            x_mask = seq_lengths_to_mask(x_lengths, x.shape[2], time_dim=2)
        x_mask = x_mask.to(device=x.device, dtype=x.dtype)

        output = torch.zeros_like(x)

        if condition is not None:
            if self.cond_channels == 0:
                raise ValueError(
                    "condition was provided but this WaveNet was initialized with cond_channels=0"
                )
            condition = self.cond_layer(condition)

        for i in range(self.num_layers):
            z = self.dconv_layers[i](x)
            if condition is not None:
                cond_start_idx = i * 2 * self.hidden_channels
                cond_end_idx = cond_start_idx + 2 * self.hidden_channels
                condition_i = condition[:, cond_start_idx:cond_end_idx, :]
            else:
                condition_i = torch.zeros_like(z)

            z = add_tanh_sigmoid_multiply(z, condition_i, self.hidden_channels)
            z = self.dropout(z)

            z = self.pwconv_layers[i](z)

            if i < self.num_layers - 1:
                residual = z[:, : self.hidden_channels, :]
                skip = z[:, self.hidden_channels :, :]
                x = (x + residual) * x_mask
                output = output + skip
            else:
                output = output + z

        output = output * x_mask
        # (batch_size, channels, time) -> (batch_size, time, channels)
        # output = output.transpose(1, 2).contiguous()
        return output

    def remove_weight_norm(self) -> None:
        """
        Removes weight normalization from all convolutional layers, including the
        conditioning layer if it exists. This is typically called after training
        to improve inference efficiency.
        """
        if self.cond_channels != 0 and is_parametrized(self.cond_layer):
            remove_parametrizations(self.cond_layer, "weight")

        for l in self.dconv_layers:
            if is_parametrized(l):
                remove_parametrizations(l, "weight")

        for l in self.pwconv_layers:
            if is_parametrized(l):
                remove_parametrizations(l, "weight")
