"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .convergence_loss import ConvergenceLoss


class MultiResolutionSTFTLoss(nn.Module):
    """Computes the multi-resultion STFT loss from [1].

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Attributes:
        window_lengths (List[int]): List of window lengths for multi-resolution STFT loss.
        hop_lengths (List[int]): List of hop lengths for multi-resolution STFT loss.
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'mean', 'sum', or 'none'.
        clamp_eps (float): Minimum value to clamp the log magnitude to avoid log(0).
        compute_conv_loss (bool): Whether to compute the convergence loss.
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        hop_lengths: Optional[List[int]] = None,
        reduction: str = "mean",
        clamp_eps: float = 1e-5,
        compute_conv_loss: bool = True,
    ):
        super().__init__()
        self.window_lengths = window_lengths
        if hop_lengths is None:
            hop_lengths = [w // 4 for w in window_lengths]
        elif len(hop_lengths) != len(window_lengths):
            raise ValueError(
                "hop_lengths must be either None or have the same length as window_lengths."
            )

        self.hop_lengths = hop_lengths
        self.n_ffts = [2 ** ((w - 1).bit_length()) for w in window_lengths]
        for i, w in enumerate(window_lengths):
            self.register_buffer(
                f"window_{i}",
                torch.hann_window(w),
            )
        self.l1_loss = nn.L1Loss(reduction=reduction)
        if compute_conv_loss:
            self.conv_loss = ConvergenceLoss(reduction=reduction)

        self.clamp_eps = clamp_eps
        self.compute_conv_loss = compute_conv_loss

    def get_mag_spectrogram(self, x, n_fft, window_length, hop_length, window):
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)

        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
        )
        return spec.abs()  # (B, F, T) where F is frequency bins and T is time frames

    def forward(
        self, x_pred: torch.Tensor, x_ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes multi-resolution STFT between an estimate and a reference
        signal.
        Args:
            x_pred (torch.Tensor): Estimated audio signal of shape (B, T).
            x_ref (torch.Tensor): Reference audio signal of shape (B, T).
        Returns:
            torch.Tensor: Computed multi-resolution STFT loss.
        """
        loss_sc = 0.0
        loss_mag = 0.0
        for i in range(len(self.window_lengths)):
            w = self.window_lengths[i]
            h = self.hop_lengths[i]
            window = getattr(self, f"window_{i}")
            n_fft = self.n_ffts[i]
            X_pred = self.get_mag_spectrogram(x_pred, n_fft, w, h, window)
            X_ref = self.get_mag_spectrogram(x_ref, n_fft, w, h, window)
            loss_mag += self.l1_loss(
                X_pred.clamp(self.clamp_eps).log(),
                X_ref.clamp(self.clamp_eps).log(),
            )
            if self.compute_conv_loss:
                loss_sc += self.conv_loss(X_pred, X_ref)

        return loss_sc, loss_mag

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--window-lengths",
            type=int,
            nargs="+",
            default=[2048, 512],
            help="List of window lengths for multi-resolution STFT loss.",
        )
        parser.add_argument(
            "--hop-lengths",
            type=int,
            nargs="+",
            default=None,
            help="List of hop lengths for multi-resolution STFT loss. If None, defaults to window_length // 4.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
