"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer for transfomer.

    Attributes:
       num_feats: input/output dimenstion
       hid_feats: number of hidden units
       activation: activation function for hidden layers
       dropout_rate: dropout rate
       time_dim: time dimension in the input tensor
    """

    def __init__(
        self, num_feats, hid_feats, activation="relu6", dropout_rate=0, time_dim=1
    ):
        super().__init__()
        self.w_1 = nn.Linear(num_feats, hid_feats)
        self.w_2 = nn.Linear(hid_feats, num_feats)
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        self.activation = AF.create(activation)
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward function.

        Args:
          x: input size=(batch, time, num_feats)

        Returns:
          tensor size=(batch, time, num_feats)
        """
        if self.time_dim != 1:
            x = x.transpose(1, time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = self.w_2(x)
        if self.time_dim != 1:
            x = x.transpose(1, time_dim)

        return x


class Conv1dx2(nn.Module):
    """Two layer Conv1d for transformer feed-forward block

    Introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Attributes:
      num_channels: input/output channels.
      hid_channels: hidden channels
      kernel_size: conv kernel size
      activation: activation function for hidden layers
      dropout_rate: dropout rate
      time_dim: indicates what is the time dimension in the input tensor.
    """

    def __init__(
        self, num_channels, hid_channels, kernel_size, dropout_rate=0, time_dim=-1
    ):

        super().__init__()
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

    def forward(self, x):
        """Calculates forward propagation.
        Args:
            x: input tensors with size=(batch, time, num_channels) or
               size=(batch, num_channels, time).
        Returns:
            output tensor same size as input
        """
        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        return x


class Conv1dLinear(nn.Module):
    """Conv1D + Linear for Transformer block.

    Attributes:
      num_channels: input/output channels.
      hid_channels: hidden channels
      kernel_size: conv kernel size
      activation: activation function for hidden layers
      dropout_rate: dropout rate
      time_dim: indicates what is the time dimension in the input tensor.

    """

    def __init__(
        self, num_channels, hid_channels, kernel_size, dropout_rate=0, time_dim=-1
    ):
        super().__init__()
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

    def forward(self, x):
        """Calculates forward propagation.
        Args:
            x: input tensors with size=(batch, time, num_channels) or
               size=(batch, num_channels, time).
        Returns:
            output tensor same size as input
        """
        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        x = self.activation(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        return x
