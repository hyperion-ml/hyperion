"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward layer for transformer.

    Attributes:
       num_feats: Input/output feature dimension.
       hid_feats: Number of hidden units.
       activation: Activation function for hidden layers.
       dropout_rate: Dropout rate.
       time_dim: Time dimension in the input tensor.
    """

    def __init__(
        self,
        num_feats: int,
        hid_feats: int,
        activation: Any = "relu6",
        dropout_rate: float = 0,
        time_dim: int = 1,
    ) -> None:
        """Initializes the positionwise feed-forward block.

        Args:
          num_feats: Input and output feature dimension.
          hid_feats: Hidden feature dimension in the intermediate projection.
          activation: Activation specification accepted by ``ActivationFactory``.
          dropout_rate: Dropout probability applied between the two projections.
          time_dim: Index of the time dimension in the input tensor.

        Returns:
          None.
        """
        super().__init__()
        if time_dim not in (1, 2, -1):
            raise ValueError(
                f"invalid time_dim={time_dim}, expected one of (1, 2, -1)"
            )
        self.w_1 = nn.Linear(num_feats, hid_feats)
        self.w_2 = nn.Linear(hid_feats, num_feats)
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        self.activation = AF.create(activation)
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies positionwise feed-forward projections.

        Args:
          x: Input tensor with shape ``(batch, time, num_feats)`` when
            ``time_dim=1``, or equivalent with time at ``time_dim``.

        Returns:
          Output tensor with the same shape as ``x``.
        """
        if self.time_dim != 1:
            x = x.transpose(1, self.time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = self.w_2(x)
        if self.time_dim != 1:
            x = x.transpose(1, self.time_dim)

        return x


class Conv1dx2(nn.Module):
    """Two-layer Conv1d block for transformer feed-forward networks.

    Introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Attributes:
      num_channels: Input/output channels.
      hid_channels: Hidden channels.
      kernel_size: Convolution kernel size.
      activation: Activation function for hidden layers.
      dropout_rate: Dropout rate.
      time_dim: Index of the time dimension in the input tensor.
    """

    def __init__(
        self,
        num_channels: int,
        hid_channels: int,
        kernel_size: int,
        activation: Any = "relu6",
        dropout_rate: float = 0,
        time_dim: int = -1,
    ) -> None:
        """Initializes the two-layer 1D convolutional feed-forward block.

        Args:
          num_channels: Input and output channel dimension.
          hid_channels: Hidden channel dimension.
          kernel_size: Convolution kernel size for both convolution layers.
          activation: Activation specification accepted by ``ActivationFactory``.
          dropout_rate: Dropout probability applied after the first convolution.
          time_dim: Index of the time dimension in the input tensor.

        Returns:
          None.
        """

        super().__init__()
        if time_dim not in (1, 2, -1):
            raise ValueError(
                f"invalid time_dim={time_dim}, expected one of (1, 2, -1)"
            )
        self.w_1 = nn.Conv1d(
            num_channels,
            hid_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            hid_channels,
            num_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        self.activation = AF.create(activation)
        if self.dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates forward propagation.

        Args:
            x: Input tensor with shape ``(batch, num_channels, time)``.

        Returns:
            Output tensor with the same shape as ``x``.
        """
        if self.time_dim != -1:
            x = x.transpose(-1, self.time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x = x.transpose(-1, self.time_dim)

        return x


class Conv1dLinear(nn.Module):
    """Conv1d + pointwise projection block for transformer feed-forward networks.

    Attributes:
      num_channels: Input/output channels.
      hid_channels: Hidden channels.
      kernel_size: Convolution kernel size.
      activation: Activation function for hidden layers.
      dropout_rate: Dropout rate.
      time_dim: Index of the time dimension in the input tensor.

    """

    def __init__(
        self,
        num_channels: int,
        hid_channels: int,
        kernel_size: int,
        activation: Any = "relu6",
        dropout_rate: float = 0,
        time_dim: int = -1,
    ) -> None:
        """Initializes the Conv1d + pointwise linear feed-forward block.

        Args:
          num_channels: Input and output channel dimension.
          hid_channels: Hidden channel dimension in the first convolution.
          kernel_size: Convolution kernel size for the first convolution layer.
          activation: Activation specification accepted by ``ActivationFactory``.
          dropout_rate: Dropout probability applied after the first convolution.
          time_dim: Index of the time dimension in the input tensor.

        Returns:
          None.
        """
        super().__init__()
        if time_dim not in (1, 2, -1):
            raise ValueError(
                f"invalid time_dim={time_dim}, expected one of (1, 2, -1)"
            )
        self.w_1 = nn.Conv1d(
            num_channels,
            hid_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = nn.Conv1d(hid_channels, num_channels, 1)

        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        self.activation = AF.create(activation)
        if self.dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates forward propagation.

        Args:
            x: Input tensor with shape ``(batch, num_channels, time)``.

        Returns:
            Output tensor with the same shape as ``x``.
        """
        if self.time_dim != -1:
            x = x.transpose(-1, self.time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x = x.transpose(-1, self.time_dim)

        return x
