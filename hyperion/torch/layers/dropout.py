"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout2d


class Dropout1d(Dropout2d):
    """Dropout for tensors with 1d spatial (time) dimension (3d tensors).

    Attributes:
      p: Drop probability.
    """

    def forward(self, inputs):
        """Applies dropout 1d.

        Args:
          inputs: Input tensor with shape = (batch, C, time).

        Returns:
          Tensor with shape = (batch, C, time).
        """
        x = torch.unsqueeze(inputs, dim=-2)
        x = F.dropout2d(x, self.p, self.training, self.inplace)
        return torch.squeeze(x, dim=-2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s


class DropConnect2d(nn.Module):
    """DropsConnect for tensor with 2d spatial dimanions (4d tensors).
        It drops the full feature map. It used to create residual networks
        with stochastic depth.

    Attributes:
      p: Probability of dropping the feature map.

    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s

    def forward(self, inputs):
        """Applies drop-connect.

        Args:
          inputs: Input tensor with shape = (batch, C, H, W).

        Returns:
          Tensor with shape = (batch, C, H, W).
        """
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - self.p
        random_tensor = (
            torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
            + keep_prob
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


class DropConnect1d(nn.Module):
    """DropsConnect for tensor with 1d spatial dimanions (3d tensors).
        It drops the full feature map. It used to create residual networks
        with stochastic depth.

    Attributes:
      p: Probability of dropping the feature map.

    """

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s

    def forward(self, inputs):
        """Applies drop-connect.

        Args:
          inputs: Input tensor with shape = (batch, C, time).

        Returns:
          Tensor with shape = (batch, C, time).
        """
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - self.p
        random_tensor = (
            torch.rand([batch_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
            + keep_prob
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output
