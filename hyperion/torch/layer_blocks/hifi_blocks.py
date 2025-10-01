"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


class HiFiBlock(nn.Module):
    """
    Residual block with dilated and non-dilated convolutions used in HiFi-GAN-like models.

    Each residual unit consists of:
      - a dilated convolution
      - a non-dilated convolution
      - residual connection

    Attributes:
        channels (int): Number of input/output channels.
        kernel_size (int): Size of convolutional kernels.
        dilation (Tuple[int, int, int]): Dilation rates for the three residual layers.
        activation (nn.Module): Activation module (e.g., nn.LeakyReLU).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.activation = activation or nn.LeakyReLU(negative_slope=0.1)

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        padding=(d * (kernel_size - 1)) // 2,
                        dilation=d,
                    )
                )
                for d in dilations
            ]
        )

        padding = (kernel_size - 1) // 2
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=1,
                    )
                )
                for _ in dilations
            ]
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of all convolutional layers.
        """
        for layer in self.convs1 + self.convs2:
            layer.weight.data.normal_(0.0, 0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (Tensor): Input tensor of shape (B, C, T).
            x_mask (Tensor, optional): Mask of shape (B, 1, T) to apply after each operation.

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            z = self.activation(x)
            if x_mask is not None:
                z = z * x_mask
            z = conv1(z)

            z = self.activation(z)
            if x_mask is not None:
                z = z * x_mask
            z = conv2(z)

            x = x + z

        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization from all convolutional layers.
        """
        for l in self.convs1 + self.convs2:
            remove_parametrizations(l, "weight")
