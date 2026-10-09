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

    Dtype behavior follows native LayerNorm for CPU, CUDA, MPS, and XPU:
        FP16/BF16 inputs use FP32 computation, including the learned scale.
        Outputs retain the input dtype, even with FP32 weights, except under
        CUDA/MPS/XPU autocast, where non-FP64 inputs produce FP32 outputs.
        FP64 inputs retain FP64 computation and output.
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
            Normalized tensor with optional per-dimension scaling and the
            LayerNorm output dtype described in the class docstring.
        """
        if x.size(-1) != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.size(-1)}")
        if not torch.is_floating_point(x):
            raise TypeError("RMSNorm expects a floating-point input")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        output_dtype = x.dtype
        if (
            x.dtype != torch.float64
            and x.device.type in ("cuda", "mps", "xpu")
            and torch.is_autocast_enabled(x.device.type)
        ):
            output_dtype = torch.float32
        output = self._norm(x.to(dtype=compute_dtype))
        if self.weight is not None:
            output = output * self.weight.to(dtype=compute_dtype)
        return output.to(dtype=output_dtype)
