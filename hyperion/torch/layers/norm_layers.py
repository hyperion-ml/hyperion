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

    Shape:
        - Input: ``(..., dim)``
        - Output: ``(..., dim)``
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes ``x`` by its RMS over the last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization and per-dimension affine scaling."""
        if x.size(-1) != self.weight.numel():
            raise ValueError(
                f"expected last dimension {self.weight.numel()}, got {x.size(-1)}"
            )
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
