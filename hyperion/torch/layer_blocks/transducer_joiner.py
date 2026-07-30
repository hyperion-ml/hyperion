"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class TransducerJoiner(nn.Module):
    """RNN-T joiner network.

    Implementation based on
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer_stateless7/joiner.py

    Attributes:
      enc_feats: Encoder input feature dimension.
      pred_feats: Predictor input feature dimension.
      hid_feats: Hidden projection dimension used before output projection.
      vocab_size: Output vocabulary size.
    """

    def __init__(
        self, enc_feats: int, pred_feats: int, hid_feats: int, vocab_size: int
    ) -> None:
        """
        Initializes the RNN-T joiner network layers.

        Args:
          enc_feats: Encoder feature dimension.
          pred_feats: Predictor feature dimension.
          hid_feats: Hidden projection dimension used by the joiner.
          vocab_size: Output vocabulary size.
        """
        super().__init__()
        self.enc_feats = enc_feats
        self.pred_feats = pred_feats
        self.hid_feats = hid_feats
        self.vocab_size = vocab_size

        self.enc_proj = nn.Linear(enc_feats, hid_feats)
        self.pred_proj = nn.Linear(pred_feats, hid_feats)
        self.output = nn.Linear(hid_feats, vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable configuration for the joiner block.

        Returns:
          A dictionary with joiner configuration fields:
          `joiner_type` and `hid_feats`.
        """
        config = {
            "joiner_type": "basic",
            "hid_feats": self.hid_feats,
        }
        return config

    def forward(
        self, enc_out: torch.Tensor, pred_out: torch.Tensor, project_input: bool = True
    ) -> torch.Tensor:
        """
        Args:
          enc_out: Output from the encoder with shape `(N, T, C)` or
            `(N, T, s_range, C)`.
          pred_out: Output from the predictor with shape `(N, U, C)` or
            `(N, T, s_range, C)`.
          project_input: If `True`, projects encoder and predictor features
            inside this method. If `False`, expects both inputs to already be
            in the same projected feature space.
        Returns:
          Symbol logits of shape `(N, T, U, vocab_size)` for 3-D inputs, or
          `(N, T, s_range, vocab_size)` for 4-D inputs.
        """
        if enc_out.ndim != pred_out.ndim:
            raise ValueError(
                f"enc_out.ndim ({enc_out.ndim}) and pred_out.ndim ({pred_out.ndim}) must match"
            )
        if enc_out.ndim not in (3, 4):
            raise ValueError(
                f"enc_out.ndim and pred_out.ndim must be 3 or 4, got {enc_out.ndim}"
            )

        if enc_out.ndim == 3:
            enc_out = enc_out.unsqueeze(2)  # (N, T, 1, C)
            pred_out = pred_out.unsqueeze(1)  # (N, 1, U, C)

        if project_input:
            x = self.enc_proj(enc_out) + self.pred_proj(pred_out)
        else:
            x = enc_out + pred_out

        x = torch.tanh(x)
        logits = self.output(x)
        return logits
