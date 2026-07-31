"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Applies the Snake activation function.

    The input is flattened across all dimensions after `(batch, channels)`,
    the Snake nonlinearity is applied, and then the original shape is restored.

    Reference:
        Ziyin et al., "Neural Networks Fail to Learn Periodic Functions and
        How to Fix It", NeurIPS 2020 (arXiv:2006.08195).

    Args:
        x: Input tensor with shape `(batch, channels, ...)`.
        alpha: Learnable Snake parameter broadcastable to `x`, typically
            shaped `(1, channels, 1)`.

    Returns:
        Tensor with the same shape as `x` after Snake activation.
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Channel-wise Snake activation for 1D-style `(batch, channels, time)` data.

    Reference:
        Ziyin et al., "Neural Networks Fail to Learn Periodic Functions and
        How to Fix It", NeurIPS 2020 (arXiv:2006.08195).

    Attributes:
        alpha: Learnable per-channel Snake parameter with shape
            `(1, channels, 1)`.

    Args:
        channels: Number of input channels.

    Returns:
        Tensor with the same shape as the input after Snake activation.
    """

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """Runs the Snake activation.

        Args:
            x: Input tensor with shape `(batch, channels, length)` or
                compatible `(batch, channels, ...)`.

        Returns:
            Activated tensor with the same shape as `x`.
        """
        return snake(x, self.alpha)
