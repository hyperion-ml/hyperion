"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torchaudio.transforms import Loudness


class LoudnessNorm(nn.Module):
    """
    Loudness normalization to a target LUFS using torchaudio.transforms.Loudness.

    Expects input shaped (..., channels, time), e.g. (B, 1, T) or (B, C, T).
    If `rescale_to_max_value` is set, acts as a *limiter-style peak rescaler*:
    only rescales when peak exceeds that value, to avoid undoing loudness match.

    Args:
        sample_freq: Sample frequency of the input signal.
        target_lufs: Target loudness level in dB.
        rescale_to_max_value: If not None, rescales the output to have maximum absolute value equal to this.
    """

    def __init__(
        self,
        sample_freq: int,
        target_lufs: Optional[float] = None,
        rescale_to_max_value: Optional[float] = None,
    ):
        super().__init__()
        self.loudness_meter = Loudness(sample_rate=sample_freq)
        self.target_lufs = target_lufs
        self.rescale_to_max_value = rescale_to_max_value

    def forward(
        self,
        x: torch.Tensor,
        target_lufs: Optional[torch.Tensor] = None,
        return_input_lufs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input signal (..., channels, time) or (batch, time).
            target_lufs: If not None, use this target loudness instead of the one set at init.
            return_in_lufs: If True, returns the input loudness in LUFS as second output.
        Returns:
            A tuple containing the processed signal and the input loudness in dB.
        """
        target_lufs = target_lufs if target_lufs is not None else self.target_lufs
        assert (
            target_lufs is not None
        ), "Either provide target_lufs at init or in forward."

        with torch.no_grad():
            with amp.autocast(enabled=False):
                if x.dim() == 2:
                    x_in = x.unsqueeze(1).float()  # (B, 1, T)
                else:
                    x_in = x.float()  # (B, C, T)

                input_lufs = self.loudness_meter(x_in)

            input_lufs = input_lufs.to(x.dtype)  # (B, C) or (B, 1)
            gain_db = target_lufs - input_lufs
            gain_db = gain_db.view(*gain_db.shape, *([1] * (x.dim() - gain_db.dim())))
            gain = 10 ** (gain_db / 20)

        x = x * gain
        if self.rescale_to_max_value is not None:
            with torch.no_grad():
                max_val = (
                    x.abs()
                    .amax(dim=-1, keepdim=True)
                    .clamp(min=self.rescale_to_max_value)
                )

            x = x / (max_val + 1e-9) * self.rescale_to_max_value

        if return_input_lufs:
            return x, input_lufs.squeeze(-1)
        else:
            return x
