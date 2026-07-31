"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

#

from typing import Optional, Type

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Conv1d

from ..layers import ActivationSpec, Dropout1d
from .etdnn_blocks import ETDNNBlock


class ResETDNNBlock(ETDNNBlock):
    """Building block for Residual Extended-TDNN.

    Args:
      num_channels:  input and output channels.
      kernel_size:   kernel size for the convolution.
      dilation:      kernel dilation.
      activation:    non-linear activation function object, string or config dict.
      dropout_rate:  dropout rate.
      use_norm:      if True, applies normalization.
      norm_layer:    Normalization layer constructor; if None, uses BatchNorm1d.
      norm_before:   if True, layer normalization is before the non-linearity, else
                     after the non-linearity.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        dilation: int = 1,
        activation: ActivationSpec = {"name": "relu", "inplace": True},
        dropout_rate: float = 0,
        norm_layer: Optional[Type[nn.Module]] = None,
        use_norm: bool = True,
        norm_before: bool = False,
    ):
        """Initializes the Residual Extended-TDNN block.

        Args:
          num_channels: Input and output channels.
          kernel_size: Convolution kernel size.
          dilation: Convolution dilation factor.
          activation: Non-linear activation specification.
          dropout_rate: Dropout probability.
          norm_layer: Normalization layer constructor; if ``None``, uses ``BatchNorm1d``.
          use_norm: If ``True``, applies normalization.
          norm_before: If ``True``, normalization is applied before the activation.
        """

        super().__init__(
            num_channels,
            num_channels,
            kernel_size,
            dilation,
            activation,
            dropout_rate,
            norm_layer,
            use_norm,
            norm_before,
        )

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, in_channels, in_time).
          x_mask: Optional input mask, unused.

        Returns:
          Tensor with shape = (batch, out_channels, out_time).
        """

        residual = x
        x = self.conv1(x)

        if self.norm_before:
            x = self.bn1(x)

        x = self.activation1(x)

        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout1(x)

        x = self.conv2(x)

        if self.norm_before:
            x = self.bn2(x)

        x += residual
        x = self.activation2(x)

        if self.norm_after:
            x = self.bn2(x)

        if self.dropout_rate > 0:
            x = self.dropout2(x)

        return x
