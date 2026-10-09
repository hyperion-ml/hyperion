"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalization over the last tensor dimension.

    Args:
        dim: Size of the last dimension to normalize.
        eps: Positive constant added for numerical stability.
        with_scale: Whether to learn a per-dimension scale (default: True).

    Attributes:
        dim: Size of the normalized dimension.
        eps: Numerical stability constant.
        with_scale: Whether a learned scale is enabled.
        weight: Learned scale, or None when with_scale=False.
        training: Training mode inherited from nn.Module.

    Shape:
        - Input: ``(..., dim)``
        - Output: ``(..., dim)``
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        """Initialize RMS normalization with optional learned scaling.

        Args:
            dim: Size of the last dimension to normalize.
            eps: Positive numerical stability constant.
            with_scale: Allocate a learned scale when True.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.dim = dim
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by RMS over the last dimension.

        Args:
            x: Input tensor.

        Returns:
            RMS-normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization and optional learned scaling.

        Args:
            x: Input tensor with last dimension equal to dim.

        Returns:
            Normalized tensor with optional per-dimension scaling.
        """
        if x.size(-1) != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.size(-1)}")
        output = self._norm(x.float()).type_as(x)
        return output if self.weight is None else output * self.weight
