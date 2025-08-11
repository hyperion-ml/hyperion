"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Attributes:
        scale_invariant (bool): Whether to compute scale-invariant SDR.
        reduction (str): Specifies the reduction to apply to the output:
            'mean', 'sum', or 'none'.
        zero_mean (bool): If True, zero-mean the references and estimates
            before computing the loss.
        clip_min (float, optional): Minimum possible loss value. Helps
            network to not focus on making already good examples better.
    """

    def __init__(
        self,
        scale_invariant: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
    ):
        self.scale_invariant = scale_invariant
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        super().__init__()

    def forward(self, x_pred: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """
        Computes the SISDR loss between estimated and reference audio signals.
        Args:
            x_pred (torch.Tensor): Estimated audio signals or features (B, C, T).
            x_ref (torch.Tensor): Reference audio signals or features (B, C, T).
        Returns:
            torch.Tensor: Computed SISDR loss.
        """
        eps = 1e-8
        batch_size = x_ref.shape[0]
        x_ref = x_ref.reshape(batch_size, 1, -1).permute(0, 2, 1)
        x_pred = x_pred.reshape(batch_size, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            x_ref = x_ref - x_ref.mean(dim=1, keepdim=True)
            x_pred = x_pred - x_pred.mean(dim=1, keepdim=True)

        if self.scale_invariant:
            corr_x_ref = (x_ref**2).sum(dim=1, keepdim=True) + eps
            corr_x_ref_x_pred = (x_ref * x_pred).sum(dim=1, keepdim=True)
            scale = corr_x_ref_x_pred / corr_x_ref
            x_ref = scale * x_ref

        noise = x_pred - x_ref

        p_signal = (x_ref**2).sum(dim=1) + eps
        p_noise = (noise**2).sum(dim=1) + eps
        sdr = -10 * torch.log10(p_signal / p_noise)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr
