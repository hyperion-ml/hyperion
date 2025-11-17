"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch
from .wavenet import WaveNet


class WaveNetPosteriorEncoder(NetArch):
    """
    Posterior Encoder based on a WaveNet architecture.

    This encoder is used in VITS and FreeVC models to infer the posterior latent
    distribution from input audio features. It predicts a Gaussian distribution
    (mean and log-std) from which latent variables are sampled.

    Attributes:
        num_layers (int): Number of convolutional layers in the WaveNet.
        in_feats (int): Number of input feature channels (e.g., mel bins).
        out_feats (int): Size of the output latent dimension (z).
        hidden_channels (int): Number of channels in the hidden layers.
        kernel_size (int): Kernel size used in each dilated convolution.
        dilation_rate (int): Base of exponential dilation (layer i uses dilation_rate ** i).
        cond_channels (int, optional): Number of external conditioning channels (e.g., speaker embeddings).
        dropout_rate (float, optional): Dropout applied within WaveNet blocks.
    """

    def __init__(
        self,
        num_layers: int = 16,
        in_feats: int = 80,
        out_feats: int = 192,
        hidden_channels: int = 512,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        cond_channels: int = 0,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the WaveNetPosteriorEncoder.

        Args:
            num_layers (int): Number of dilated convolution layers in the WaveNet.
            in_feats (int): Number of input feature channels (e.g., mel bins).
            out_feats (int): Size of the output latent dimension (z).
            hidden_channels (int): Number of channels used inside the WaveNet.
            kernel_size (int): Kernel size for dilated convolutions (must be odd).
            dilation_rate (int): Dilation rate base across layers.
            cond_channels (int, optional): Number of conditioning channels. Default is 0.
            dropout_rate (float, optional): Dropout probability used inside WaveNet. Default is 0.1.
        """
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels
        self.dropout_rate = dropout_rate

        self.in_conv = nn.Conv1d(
            in_channels=in_feats,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.out_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_feats * 2,
            kernel_size=1,
        )
        self.wavenet = WaveNet(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            cond_channels=cond_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the posterior encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, in_feats), where B = batch size, T = time steps.
            x_lengths (torch.Tensor, optional): Lengths of each input sequence (used to create a mask).
            condition (torch.Tensor, optional): Optional conditioning input (e.g., speaker embedding) of shape (B, Cc, T).
            deterministic (bool): If True, returns the mean of the posterior distribution instead of sampling.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z (Tensor): Sampled latent representation of shape (B, T, out_feats).
                - m (Tensor): Predicted mean of the posterior (B, out_feats, T, out_feats).
                - logs (Tensor): Predicted log standard deviation (B, T, out_feats).
        """
        x = x.transpose(1, 2)  # (B, T, in_feats) -> (B, in_feats, T)
        if condition is not None:
            if condition.dim() == 2:
                condition = condition.unsqueeze(2)
            else:
                assert condition.dim() == 3, "Conditioning tensor must be 2D or 3D"
                condition = condition.transpose(1, 2)

            condition = condition.contiguous()

        x_mask = seq_lengths_to_mask(x_lengths, x.size(2), time_dim=2).to(x.dtype)
        x = self.in_conv(x) * x_mask
        x = self.wavenet(x, x_mask=x_mask, condition=condition)
        stats = self.out_conv(x) * x_mask
        stats = stats.transpose(
            1, 2
        ).contiguous()  # (B, out_feats * 2, T) -> (B, T, out_feats * 2)
        mean_z, logs_z = torch.split(stats, self.out_feats, dim=2)
        if deterministic:
            z = mean_z
        else:
            z = mean_z + torch.randn_like(mean_z) * torch.exp(logs_z)

        return z, mean_z, logs_z

    def remove_weight_norm(self):
        self.wavenet.remove_weight_norm()

    def get_config(self, no_class_name: bool = False):
        """
        Returns the encoder configuration as a dictionary.

        Returns:
            Dict[str, Any]: Configuration values used to initialize the encoder.
        """
        config = {
            "num_layers": self.num_layers,
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "cond_channels": self.cond_channels,
            "dropout_rate": self.dropout_rate,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def add_class_args(parser, prefix: Optional[str] = None, skip: Set = set()):
        """
        Adds encoder arguments to the CLI parser.

        Args:
            parser (ArgumentParser): Argument parser object.
            prefix (str, optional): Optional prefix for argument names.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--num-layers",
            type=int,
            default=16,
            help="Number of layers in the WaveNet stack.",
        )
        parser.add_argument(
            "--in-feats",
            type=int,
            default=80,
            help="Input feature dimension (e.g., mel bins).",
        )
        parser.add_argument(
            "--out-feats",
            type=int,
            default=192,
            help="Output latent dimension size (z).",
        )
        parser.add_argument(
            "--hidden-channels",
            type=int,
            default=512,
            help="Number of hidden channels in WaveNet layers.",
        )
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=5,
            help="Convolution kernel size (must be odd).",
        )
        parser.add_argument(
            "--dilation-rate",
            type=int,
            default=1,
            help="Base of exponential dilation in WaveNet.",
        )
        if "cond_channels" not in skip:
            parser.add_argument(
                "--cond-channels",
                type=int,
                default=0,
                help="Number of conditioning channels.",
            )
        parser.add_argument(
            "--dropout-rate",
            type=float,
            default=0.1,
            help="Dropout rate used inside the WaveNet.",
        )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
