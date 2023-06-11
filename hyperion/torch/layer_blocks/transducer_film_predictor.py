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
from .film_blocks import FiLM, RNNWithFiLM, RNNWithFiLMResidual

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
                 film_type: str = "linear",
                 blank_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        if rnn_type in ["lstm","gru"]:
            self.rnn = RNNWithFiLM(
                input_size=embed_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                dropout=rnn_dropout_rate,
                condition_size=condition_size,
                batch_first=True,
                rnn_type=rnn_type,
                film_type=film_type
            )
        elif rnn_type in ["lstm_residual","gru_residual"]:
            self.rnn = RNNWithFiLMResidual(
                input_size=embed_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                dropout=rnn_dropout_rate,
                condition_size=condition_size,
                batch_first=True,
                rnn_type=rnn_type,
                film_type=film_type
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
            "film_type": self.film_type,
            "blank_id": self.blank_id,
        }
        return config

    def forward(
        self,
        y: torch.Tensor,
        lang_condition: torch.Tensor,
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
        out, (h, c) = self.rnn(embed, states, lang_condition)
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

class TransducerConvPredictor(nn.Module):
    """ RNN-T prediction network based on Convolutions
    Implmentation  based on:
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7/decoder.py

    Attributes:
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the input embedding.
      blank_id: The ID of the blank symbol.
      out_feats: Output dimension of the predictor.
      embed_dropout_rate: Dropout rate for the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        condition_size: int,
        out_feats: Optional[int] = None,
        context_size: int = 2,
        embed_dropout_rate: float = 0.0,
        hid_act: str = "relu",
        blank_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        assert context_size >= 1, context_size
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=context_size,
                padding=0,
                groups=out_feats // 4,
                bias=False,
            )

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_dropout_rate = embed_dropout_rate
        self.context_size = context_size
        self.hid_act = AF.create(hid_act)

        if out_feats is None:
            out_feats = embed_dim

        self.out_feats = out_feats
        if out_feats != embed_feats:
            self.output_proj = nn.Linear(embed_dim, out_feats)
        else:
            self.output_proj = None

    def get_config(self):
        hid_act = AF.get_config(self.hid_act)
        config = {
            "pred_type": "conv",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "out_feats": self.out_feats,
            "context_size": self.context_size,
            "embed_dropout_rate": self.embed_dropout_rate,
            "blank_id": self.blank_id,
            "hid_act": hid_act,
        }
        return config

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          # need_pad:
          #   True to left pad the input. Should be True during training.
          #   False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        embed = self.embedding(y)
        if self.context > 1:
            embed = embed.transpose(1, 2)
            if states is None:
                embed = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                raise NotImplementedError()
            embed = self.conv(embed).transpose(1, 2)

        out = self.hid_act(embed)
        if self.output_proj:
            out = self.output_proj(out)

        return out, None

        # # this stuff about clamp() is a temporary fix for a mismatch
        # # at utterance start, we use negative ids in beam_search.py
        # if torch.jit.is_tracing():
        #     # This is for exporting to PNNX via ONNX
        #     embedding_out = self.embedding(y)
        # else:
        #     embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)
        # if self.context_size > 1:
        #     embedding_out = embedding_out.permute(0, 2, 1)
        #     if need_pad is True:
        #         embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
        #     else:
        #         # During inference time, there is no need to do extra padding
        #         # as we only need one output
        #         assert embedding_out.size(-1) == self.context_size
        #     embedding_out = self.conv(embedding_out)
        #     embedding_out = embedding_out.permute(0, 2, 1)
        # embedding_out = F.relu(embedding_out)
        # return embedding_out

    def change_config(
        self,
        override_dropouts=False,
        embed_dropout_rate: float = 0.0,
    ):
        logging.info("changing predictor config")

        if override_dropouts:
            logging.info("overriding predictor dropouts")
            self.embed_dropout_rate = embed_dropout_rate
            self.embed_dropout = nn.Dropout(self.embed_dropout_rate)
