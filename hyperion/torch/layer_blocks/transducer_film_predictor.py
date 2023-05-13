"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layers import ActivationFactory as AF
from .film_blocks import FiLM, LSTMWithFiLM

class TransducerRNNFiLMPredictor(nn.Module):
    """ RNN-T prediction network with LSTM or GRU
    Attributes:
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the input embedding.
      num_layers: Number of LSTM layers.
      hid_feats: Hidden dimension of LSTM layers.
      out_feats: Output dimension of the predictor.
      embed_dropout_rate: Dropout rate for the embedding layer.
      rnn_dropout_rate: Dropout for LSTM layers.
      rnn_type: between lstm and gru
      blank_id: The ID of the blank symbol.           
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_layers: int,
                 hid_feats: int,
                 condition_size: int,
                 out_feats: Optional[int] = None,
                 embed_dropout_rate: float = 0.0,
                 rnn_dropout_rate: float = 0.0,
                 rnn_type: str = "lstm",
                 blank_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        if rnn_type == "lstm":
            self.rnn = LSTMWithFiLM(
                input_size=embed_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                dropout=rnn_dropout_rate,
                condition_size=condition_size,
                batch_first=True,
            )
        else:
            raise Exception(f"Unknown RNN type {rnn_type}")

        self.out_feats = out_feats
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hid_feats = hid_feats
        self.embed_dropout_rate = embed_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        if out_feats is None:
            out_feats = hid_feats

        self.out_feats = out_feats
        if out_feats != hid_feats:
            self.output_proj = nn.Linear(hid_feats, out_feats)
        else:
            self.output_proj = None

    def get_config(self):
        config = {
            "pred_type": "conv",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "hid_feats": self.hid_feats,
            "out_feats": self.out_feats,
            "embed_dropout_rate": self.embed_dropout_rate,
            "rnn_dropout_rate": self.rnn_dropout_rate,
            "rnn_type": self.rnn_type,
            "blank_id": self.blank_id,
        }
        return config

    def forward(
        self,
        y: torch.Tensor,
        condition: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args: 
          y: previous y_{<t} tensor of shape (N, U) with <sos> prepended.
          states: tuple of tensors containing RNN layers states
        Returns:
          - rnn_output, a tensor of shape (N, U, C)
          - (h, c), containing the states i for RNN layers with shape (num_layers, N, C).
        """
        embed = self.embedding(y)
        embed = self.embed_dropout(embed)
        out, (h, c) = self.rnn(embed, states, condition)
        if self.output_proj:
            out = self.output_proj(out)

        return out, (h, c)

    def change_config(
        self,
        override_dropouts=False,
        embed_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ):
        logging.info("changing decoder config")

        if override_dropouts:
            logging.info("overriding decoder dropouts")
            self.rnn_dropout_rate = rnn_dropout_rate
            self.rnn.p = self.rnn_dropout_rate
            self.embed_dropout_rate = embed_dropout_rate
            self.embed_dropout = nn.Dropout(self.embed_dropout_rate)
