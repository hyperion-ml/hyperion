"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropout import Dropout1d

class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.
    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, num_feats, hid_feats, dropout_rate=0, time_dim=1):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(num_feats, hid_feats)
        self.w_2 = torch.nn.Linear(hid_feats, num_feats)
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        if self.time_dim != 1:
            x = x.transpose(1, time_dim)

        x = F.relu6(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = self.w_2(x)
        if self.time_dim != 1:
            x = x.transpose(1, time_dim)

        return x



class Conv1dx2(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.
    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, num_channels, hid_channels, kernel_size, dropout_rate=0, time_dim=-1):
        """Initialize MultiLayeredConv1d module.
        Args:
            in_chans (int): Number of input channels.
            hid_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(Conv1dx2, self).__init__()
        self.w_1 = torch.nn.Conv1d(num_channels, hid_channels, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hid_channels, num_channels, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        if self.dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)


    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Batch of input tensors (B, ..., num_channels).
        Returns:
            Tensor: Batch of output tensors (B, ..., hid_channels).
        """
        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        x = F.relu6(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        return x


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.
    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.
    """
    def __init__(self, num_channels, hid_channels, kernel_size, dropout_rate=0, time_dim=-1):
        """Initialize MultiLayeredConv1d module.
        Args:
            in_chans (int): Number of input channels.
            hid_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(num_channels, hid_channels, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hid_channels, num_channels, 1 )
        self.dropout_rate = dropout_rate
        self.time_dim = time_dim
        if self.dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Batch of input tensors (B, ..., num_channels).
        Returns:
            Tensor: Batch of output tensors (B, ..., hid_channels).
        """
        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        x = F.relu6(self.w_1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.w_2(x)

        if self.time_dim != -1:
            x.transpose(-1, self.time_dim)

        return x


