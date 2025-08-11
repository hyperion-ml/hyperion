"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvergenceLoss(nn.Module):
    """
    Computes the convergence loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Choose from 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(
        self,
        x_pred: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the convergence loss between estimated and reference tensors

        Args:
            x_pred (torch.Tensor): Estimated audio signal of shape (B, *).
            x_ref (torch.Tensor): Reference audio signal of shape (B, *).

        Returns:
            torch.Tensor: Computed convergence loss.
        """
        eps = 1e-8
        dims = [i for i in range(1, x_ref.dim())]
        loss = torch.norm(x_pred - x_ref, p=2, dim=dims) / (
            torch.norm(x_ref, p=2, dim=dims) + eps
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
