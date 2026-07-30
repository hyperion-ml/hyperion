"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

#

# import numpy as np

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, Linear

from ..layers import ActivationFactory as AF


class FCBlock(nn.Module):
    """Fully connected block.

    Attributes:
      in_feats: input feature dimension
      out_feats: output feature dimension
      activation: activation specification used to build the non-linearity
      dropout_rate: dropout probability applied after the block
      dropout: dropout module, created only when `dropout_rate > 0`
      bn1: normalization layer when `use_norm` is True
      linear: fully connected linear projection
      norm_layer: normalization layer constructor, if None it uses batch-norm
      use_norm: if True, it applies the normalization layer, if False no normalization is applied
      norm_before: if True, normalization layer is applied before the activation function, if False after
      norm_after: if True, normalization layer is applied after the activation function, if False before
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        activation: Union[str, Dict[str, Any], Callable[..., nn.Module]] = {
            "name": "relu",
            "inplace": True,
        },
        dropout_rate: float = 0,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        use_norm: bool = True,
        norm_before: bool = False,
    ) -> None:
        """Initializes the fully connected block.

        Args:
          in_feats: input feature dimension.
          out_feats: output feature dimension.
          activation: activation specification used to build the non-linearity.
          dropout_rate: dropout probability applied after the block.
          norm_layer: normalization layer constructor, if any.
          use_norm: if True, apply normalization in the block.
          norm_before: if True, apply normalization before activation.
        """

        super().__init__()

        self.activation = AF.create(activation)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            if norm_layer is None:
                self.bn1 = BatchNorm1d(out_feats)
            else:
                self.bn1 = norm_layer(out_feats)
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.linear = Linear(in_feats, out_feats, bias=(not self.norm_before))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block.

        Args:
          x: input tensor.

        Returns:
          Output tensor after linear projection, optional normalization, activation, and dropout.
        """
        x = self.linear(x)
        if self.norm_before:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear part of the block.

        Args:
          x: input tensor.

        Returns:
          Output tensor after linear projection and optional normalization, without activation or dropout.
        """
        x = self.linear(x)

        if self.norm_before:
            x = self.bn1(x)

        if self.activation is None and self.norm_after:
            x = self.bn1(x)

        return x
