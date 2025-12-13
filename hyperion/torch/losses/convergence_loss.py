"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class ConvergenceLoss(nn.Module):
    """
    Computes the convergence loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'mean', 'sum', or 'none'.
        ref_floor (float): Minimum value for the reference norm to avoid
            exploding ratios when the reference energy is close to zero.
    """

    def __init__(
        self,
        reduction: str = "mean",
        ref_floor: float = 1e-2,
    ):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Choose from 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction
        if ref_floor <= 0:
            raise ValueError("ref_floor must be positive.")
        self.ref_floor = ref_floor

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
        dims = [i for i in range(1, x_ref.dim())]
        ref_norm = torch.norm(x_ref, p=2, dim=dims).clamp_min(self.ref_floor).detach()
        loss = torch.norm(x_pred - x_ref, p=2, dim=dims) / ref_norm
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
