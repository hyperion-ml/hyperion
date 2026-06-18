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

from ...utils.misc import filter_func_args
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch
from .wavenet import WaveNet


class NVPFlow(NetArch):
    """
    Base class for Normalizing Volume Preserving (NVP) Flow architectures.
    These models use a series of invertible transformations (e.g., affine coupling layers)
    to learn a complex distribution over data.

    This class is intended to be subclassed by specific NVP implementations such as those
    using WaveNet-based coupling layers.

    Attributes:
        None: This base class does not define additional instance attributes.
    """

    def __init__(self) -> None:
        """Initialize the base NVP flow module."""
        super().__init__()


class WaveNetNVPFlow(NVPFlow):
    """
    Normalizing Volume Preserving Flow architecture using WaveNet-based affine coupling layers.

    Attributes:
        num_coupling_layers (int): Number of coupling layers (flow steps).
        num_wavenet_layers (int): Number of layers in each WaveNet used inside a coupling layer.
        channels (int): Number of input/output channels (must be divisible by 2).
        hidden_channels (int): Number of channels in the hidden layers of the WaveNet.
        kernel_size (int): Kernel size used in WaveNet convolutions.
        dilation_rate (int): Dilation rate used in WaveNet layers.
        cond_channels (int, optional): Number of conditioning channels (e.g., speaker or style embedding).
        dropout_rate (float): Dropout rate used inside each WaveNet block.
        volume_preserving (bool): If True, the flow is volume-preserving.
    """

    def __init__(
        self,
        num_coupling_layers: int,
        num_wavenet_layers: int,
        channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        cond_channels: int = 0,
        dropout_rate: float = 0.0,
        volume_preserving: bool = True,
    ) -> None:
        """Initialize the WaveNet-based NVP flow.

        Args:
            num_coupling_layers: Number of affine coupling layers in the flow.
            num_wavenet_layers: Number of convolution layers in each WaveNet block.
            channels: Number of input and output channels.
            hidden_channels: Number of hidden channels inside each WaveNet block.
            kernel_size: Kernel size used in the WaveNet convolutions.
            dilation_rate: Base dilation rate for the WaveNet layers.
            cond_channels: Number of conditioning channels.
            dropout_rate: Dropout rate used in the WaveNet blocks.
            volume_preserving: If True, use a volume-preserving coupling transform.
        """
        super().__init__()
        self.num_coupling_layers = num_coupling_layers
        self.num_wavenet_layers = num_wavenet_layers
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.cond_channels = cond_channels
        self.dropout_rate = dropout_rate
        self.volume_preserving = volume_preserving
        self.coupling_layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            self.coupling_layers.append(
                WaveNetAffineCouplingLayer(
                    num_wavenet_layers,
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    cond_channels=cond_channels,
                    dropout_rate=dropout_rate,
                    volume_preserving=volume_preserving,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse pass through the NVP flow.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).
            x_lengths (Tensor, optional): Valid sequence lengths of shape (B,).
                If omitted, the full sequence is treated as valid.
            condition (Tensor, optional): Conditioning features of shape (B, T, Cond. C)
                or (B, Cond. C). A 2D tensor is converted to a length-1 time axis.
            reverse (bool): Whether to perform the inverse transformation.

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and log-determinant of the
            Jacobian. The same return structure is used for both forward and reverse
            directions.
        """
        log_det = x.new_zeros(x.size(0))
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        if condition is not None:
            if condition.dim() == 2:
                condition = condition.unsqueeze(-1)
            else:
                assert condition.dim() == 3, "Conditioning tensor must be 2D or 3D"
                condition = condition.transpose(1, 2)

            condition = condition.contiguous()

        if x_lengths is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2), dtype=x.dtype, device=x.device)
        else:
            x_mask = seq_lengths_to_mask(x_lengths, x.size(2), time_dim=2).to(
                dtype=x.dtype, device=x.device
            )

        if not reverse:
            for layer in self.coupling_layers:
                x, log_det_l = layer(x, x_mask, condition=condition, reverse=False)
                x = torch.flip(x, [1])  # logdet Jacobian of flip is zero
                log_det += log_det_l
        else:
            for layer in reversed(self.coupling_layers):
                x = torch.flip(x, [1])
                x, log_det_l = layer(x, x_mask, condition=condition, reverse=True)
                log_det += log_det_l

        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        return x, log_det

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """
        Returns the configuration of the WaveNetNVPFlow as a dictionary.

        Args:
            no_class_name: If True, omit the class name from the base configuration.

        Returns:
            Dict[str, Any]: Configuration values used to initialize the model.
        """
        config = {
            "num_coupling_layers": self.num_coupling_layers,
            "num_wavenet_layers": self.num_wavenet_layers,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "cond_channels": self.cond_channels,
            "dropout_rate": self.dropout_rate,
            "volume_preserving": self.volume_preserving,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds WaveNetNVPFlow arguments to an ArgumentParser.

        Args:
            parser (ArgumentParser): The target parser object.
            prefix (str, optional): Optional prefix for the argument namespace.
            skip (Set[str], optional): Argument names to omit from the parser.
        """
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "num_coupling_layers" not in skip:
            parser.add_argument(
                "--num-coupling-layers",
                type=int,
                default=4,
                help="Number of affine coupling layers in the flow.",
            )
        if "num_wavenet_layers" not in skip:
            parser.add_argument(
                "--num-wavenet-layers",
                type=int,
                default=4,
                help="Number of convolution layers in each WaveNet block.",
            )
        if "channels" not in skip:
            parser.add_argument(
                "--channels",
                type=int,
                default=192,
                help="Number of input/output channels for the flow.",
            )
        if "hidden_channels" not in skip:
            parser.add_argument(
                "--hidden-channels",
                type=int,
                default=192,
                help="Hidden layer size used inside each WaveNet block.",
            )
        if "kernel_size" not in skip:
            parser.add_argument(
                "--kernel-size",
                type=int,
                default=5,
                help="Convolution kernel size used in WaveNet layers.",
            )
        if "dilation_rate" not in skip:
            parser.add_argument(
                "--dilation-rate",
                type=int,
                default=1,
                help="Base of exponential dilation across WaveNet layers.",
            )
        if "cond_channels" not in skip:
            parser.add_argument(
                "--cond-channels",
                type=int,
                default=0,
                help="Number of external conditioning channels.",
            )
        if "volume_preserving" not in skip:
            parser.add_argument(
                "--volume-preserving",
                action=ActionYesNo,
                default=True,
                help="If True, the flow is volume-preserving.",
            )
        if "dropout_rate" not in skip:
            parser.add_argument(
                "--dropout-rate",
                type=float,
                default=0.0,
                help="Dropout rate used inside the WaveNet blocks.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))


class WaveNetAffineCouplingLayer(NetArch):
    """
    WaveNet-based affine coupling layer for NVP flows.

    The input tensor is split into two halves: one half is used to compute
    scale and shift parameters via a WaveNet, and the other half is transformed.

    Attributes:
        num_wavenet_layers (int): Number of layers in the internal WaveNet.
        channels (int): Number of total input channels (must be even).
        hidden_channels (int): Hidden layer size in WaveNet.
        kernel_size (int): Kernel size in WaveNet convolutions.
        dilation_rate (int): Dilation rate per layer in WaveNet.
        cond_channels (int, optional): Number of conditioning channels.
        dropout_rate (float): Dropout rate used in WaveNet.
        volume_preserving (bool): If True, only the shift term is predicted.
    """

    def __init__(
        self,
        num_wavenet_layers: int,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        cond_channels: int = 0,
        dropout_rate: float = 0.0,
        volume_preserving: bool = False,
    ) -> None:
        """Initialize a WaveNet affine coupling layer.

        Args:
            num_wavenet_layers: Number of layers in the internal WaveNet.
            channels: Number of total input channels. Must be even.
            hidden_channels: Hidden size inside the WaveNet.
            kernel_size: WaveNet convolution kernel size.
            dilation_rate: Base dilation rate for the WaveNet layers.
            cond_channels: Number of conditioning channels.
            dropout_rate: Dropout rate used inside the WaveNet.
            volume_preserving: If True, predict only a shift term.
        """
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.num_wavenet_layers = num_wavenet_layers
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.cond_channels = cond_channels
        self.condition_channels = cond_channels
        self.half_channels = channels // 2
        self.dropout_rate = dropout_rate
        self.volume_preserving = volume_preserving

        self.in_conv = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.wavenet = WaveNet(
            num_layers=num_wavenet_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            cond_channels=cond_channels,
            dropout_rate=dropout_rate,
        )
        r = 1 if volume_preserving else 2
        self.out_conv = nn.Conv1d(hidden_channels, self.half_channels * r, 1)
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse pass through the coupling layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, T).
            x_mask (Tensor): Binary mask of shape (B, 1, T).
            condition (Tensor, optional): Conditioning tensor of shape (B, Cc, T)
                or (B, Cc). A 2D tensor is converted to a length-1 time axis.
            reverse (bool): Whether to run the inverse transformation.

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and log-determinant of the
            Jacobian.
        """
        x = x * x_mask
        x_a, x_b = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.in_conv(x_a)
        if condition is not None and condition.dim() == 2:
            condition = condition.unsqueeze(-1)

        h = self.wavenet(h, x_mask=x_mask, condition=condition)
        h = self.out_conv(h) * x_mask
        if self.volume_preserving:
            mean = h
            logs = torch.zeros_like(h)
        else:
            mean, logs = torch.split(h, [self.half_channels] * 2, dim=1)

        if not reverse:
            x_b = mean + x_b * torch.exp(logs)
            log_det = torch.sum(logs * x_mask, dim=[1, 2])
        else:
            x_b = (x_b - mean) * torch.exp(-logs)
            log_det = -torch.sum(logs * x_mask, dim=[1, 2])

        x = torch.cat([x_a, x_b], dim=1)
        return x, log_det
