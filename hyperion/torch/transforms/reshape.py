"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Sequence, Union

import torch

ShapeLike = Union[torch.Size, Sequence[int]]


class Reshape:
    """Callable transform that reshapes an input tensor."""

    def __init__(self, shape: ShapeLike) -> None:
        """Initialize the reshape transform.

        Args:
            shape: Target output shape passed to ``torch.reshape``.
        """
        self.shape = shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor.

        Args:
            x: Input tensor to reshape.

        Returns:
            Reshaped tensor.
        """
        return torch.reshape(x, shape=self.shape)
