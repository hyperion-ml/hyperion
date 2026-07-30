"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout2d


class Dropout1d(Dropout2d):
    """Channel-wise dropout for 3-D tensors with shape ``(batch, channels, time)``.

    This module reuses :func:`torch.nn.functional.dropout2d` by temporarily
    inserting a singleton spatial dimension, so dropout is applied per channel
    and shared across the full time axis.

    Attributes:
      p: Drop probability in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        if p < 0 or p >= 1:
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")
        super().__init__(p=p, inplace=inplace)

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies channel-wise dropout to a 3-D tensor.

        Args:
          inputs: Input tensor with shape ``(batch, channels, time)``.

        Returns:
          Output tensor with the same shape as ``inputs``.
        """
        x = torch.unsqueeze(inputs, dim=-2)
        x = F.dropout2d(x, self.p, self.training, self.inplace)
        return torch.squeeze(x, dim=-2)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class DropConnect2d(nn.Module):
    """Drop-connect for 4-D tensors with shape ``(batch, channels, height, width)``.

    During training, one binary mask value is sampled per batch element and
    broadcast to all channels and spatial locations. This effectively drops or
    keeps the full feature map per sample, which is commonly used for stochastic
    depth-style regularization.

    Attributes:
      p: Probability of dropping the feature map in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")
        self.p = p

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies drop-connect to a 4-D tensor.

        Args:
          inputs: Input tensor with shape ``(batch, channels, height, width)``.

        Returns:
          Tensor with the same shape as ``inputs``. In evaluation mode the
          input is returned unchanged.
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
    """Drop-connect for 3-D tensors with shape ``(batch, channels, time)``.

    During training, one binary mask value is sampled per batch element and
    broadcast to all channels and time steps. This drops or keeps the full
    sample-level feature map.

    Attributes:
      p: Probability of dropping the feature map in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")
        self.p = p

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies drop-connect to a 3-D tensor.

        Args:
          inputs: Input tensor with shape ``(batch, channels, time)``.

        Returns:
          Tensor with the same shape as ``inputs``. In evaluation mode the
          input is returned unchanged.
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


DropPath2d = DropConnect2d
DropPath1d = DropConnect1d
