"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from .film_blocks import FiLM


class TransducerFiLMJoiner(nn.Module):
    """ RNN-T Joiner network.
    Implementation based on 
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer_stateless7/joiner.py

    Attributes:
      in_feats: input feature dimension.
      vocab_size: vocabulary size
    """

    def __init__(self, enc_feats: int, pred_feats: int, hid_feats: int, vocab_size: int, condition_size: int):
        
        super().__init__()
        self.enc_feats = enc_feats
        self.pred_feats = pred_feats
        self.hid_feats = hid_feats
        self.vocab_size = vocab_size

        self.enc_proj = nn.Linear(enc_feats, hid_feats)
        self.pred_proj = nn.Linear(pred_feats, hid_feats)
        self.output = nn.Linear(hid_feats, vocab_size)

        self.FiLM_encoder = FiLM(hid_feats, condition_size)
        self.FiLM_joiner = FiLM(hid_feats, condition_size)
        
    def get_config(self):
        config = {
            "joiner_type": "basic",
            "hid_feats": self.hid_feats,
        }
        return config

    def forward(self,
            enc_out: torch.Tensor,
            pred_out: torch.Tensor,
            condition: torch.Tensor, 
            project_input: bool = True) -> torch.Tensor:
        
        """
        Args:
          enc_out: output from the encoder with shape = (N, T, C) or (N, T, s_range, C)
          pred_out: output from the predictor with shape = (N, U, C) or (N, T, s_range, C)
          project_input: if True projects the encoder and predictor features 
            in the forward founction, if False it expects them outside.
        Returns:
          Symbols' logits of shape (N, T, U, C).
        """
        assert enc_out.ndim == pred_out.ndim
        assert enc_out.ndim in (3, 4)

        if enc_out.ndim == 3:
            enc_out = enc_out.unsqueeze(2)  # (N, T, 1, C)
            pred_out = pred_out.unsqueeze(1)  # (N, 1, U, C)
        
        enc_out = self.FiLM_encoder(enc_out, condition)

        if project_input:
            x = self.enc_proj(enc_out) + self.pred_proj(pred_out)
        else:
            x = enc_out + pred_out

        x = self.FiLM_joiner(x, condition)
        
        x = torch.tanh(x)
        logits = self.output(x)
        return logits
