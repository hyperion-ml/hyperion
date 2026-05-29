"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layers import ActivationFactory as AF


class TransducerRNNPredictor(nn.Module):
    """RNN-T prediction network with LSTM or GRU.

    Implementation based on:
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/decoder.py

    Attributes:
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the input embedding.
      num_layers: Number of LSTM layers.
      hid_feats: Hidden dimension of LSTM layers.
      out_feats: Output dimension of the predictor.
      embed_dropout_rate: Dropout rate for the embedding layer.
      rnn_dropout_rate: Dropout for recurrent layers.
      rnn_type: Recurrent cell type (`lstm` or `gru`).
      blank_id: The ID of the blank symbol.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        hid_feats: int,
        out_feats: Optional[int] = None,
        embed_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
        rnn_type: str = "lstm",
        blank_id: int = 0,
    ):
        """Build an RNN-based transducer predictor network.

        Args:
          vocab_size: Number of output symbols including blank.
          embed_dim: Token embedding dimension.
          num_layers: Number of recurrent layers.
          hid_feats: Recurrent hidden size.
          out_feats: Optional output projection size.
          embed_dropout_rate: Embedding dropout probability.
          rnn_dropout_rate: Recurrent dropout probability.
          rnn_type: Recurrent cell type (`"lstm"` or `"gru"`).
          blank_id: Blank/padding token id.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout_rate,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout_rate,
            )
        else:
            raise Exception(f"Unknown RNN type {rnn_type}")

        self.out_feats = out_feats
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hid_feats = hid_feats
        self.rnn_type = rnn_type
        self.embed_dropout_rate = embed_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        if out_feats is None:
            out_feats = hid_feats

        self.out_feats = out_feats
        if out_feats != hid_feats:
            self.output_proj = nn.Linear(hid_feats, out_feats)
        else:
            self.output_proj = None

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary for this predictor.

        Returns:
          Dictionary with predictor construction parameters.
        """
        config = {
            "pred_type": "rnn",
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute predictor outputs and recurrent states.

        Args:
          y: Previous tokens tensor of shape `(batch, steps)`, with `<sos>` prepended.
          states: Optional recurrent state tuple `(h, c)` for RNN layers.
            For GRU, only `h` is consumed.

        Returns:
          Tuple `(out, (h, c))` where:
          `out` has shape `(batch, steps, out_feats)` and `(h, c)` are recurrent
          states with shape `(num_layers, batch, hid_feats)`.
        """
        embed = self.embedding(y)
        embed = self.embed_dropout(embed)
        if self.rnn_type == "gru":
            h0 = states[0] if states is not None else None
            out, h = self.rnn(embed, h0)
            # Keep return signature consistent with LSTM by providing
            # a placeholder second state for GRU callers expecting `(h, c)`.
            c = torch.zeros_like(h)
        else:
            out, (h, c) = self.rnn(embed, states)
        if self.output_proj:
            out = self.output_proj(out)

        return out, (h, c)

    def change_config(
        self,
        override_dropouts: bool = False,
        embed_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ) -> None:
        """Update predictor dropout settings.

        Args:
          override_dropouts: If True, apply provided dropout values.
          embed_dropout_rate: New embedding dropout probability.
          rnn_dropout_rate: New recurrent dropout probability.
        """
        logging.info("changing decoder config")

        if override_dropouts:
            logging.info("overriding decoder dropouts")
            self.rnn_dropout_rate = rnn_dropout_rate
            self.rnn.dropout = self.rnn_dropout_rate
            self.embed_dropout_rate = embed_dropout_rate
            self.embed_dropout = nn.Dropout(self.embed_dropout_rate)


class TransducerConvPredictor(nn.Module):
    """RNN-T prediction network based on convolutions.

    Implementation based on:
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
        out_feats: Optional[int] = None,
        context_size: int = 2,
        embed_dropout_rate: float = 0.0,
        hid_act: str = "relu",
        blank_id: int = 0,
    ):
        """Build a convolution-based transducer predictor network.

        Args:
          vocab_size: Number of output symbols including blank.
          embed_dim: Token embedding dimension.
          out_feats: Optional output projection size.
          context_size: Context window size for depthwise conv.
          embed_dropout_rate: Embedding dropout probability.
          hid_act: Hidden activation configuration.
          blank_id: Blank/padding token id.
        """
        super().__init__()
        if out_feats is None:
            out_feats = embed_dim

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
                groups=embed_dim,
                bias=False,
            )

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_dropout_rate = embed_dropout_rate
        self.context_size = context_size
        self.hid_act = AF.create(hid_act)

        self.out_feats = out_feats
        if out_feats != embed_dim:
            self.output_proj = nn.Linear(embed_dim, out_feats)
        else:
            self.output_proj = None

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary for this predictor.

        Returns:
          Dictionary with predictor construction parameters.
        """
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
        states: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Compute predictor outputs and updated convolution context.

        Args:
          y: Token ids tensor of shape `(batch, steps)`.
          states: Optional cached left context tuple with one tensor of shape
            `(batch, embed_dim, context_size - 1)`.

        Returns:
          Tuple `(out, (new_state,))` where:
          `out` has shape `(batch, steps, out_feats)` and `new_state` stores the
          last `context_size - 1` frames of left context in shape
          `(batch, embed_dim, context_size - 1)`.
        """
        y = y.to(torch.int64)
        embed = self.embedding(y)
        embed = self.embed_dropout(embed)
        if self.context_size > 1:
            embed = embed.transpose(1, 2)
            if states is None:
                embed = nn.functional.pad(embed, pad=(self.context_size - 1, 0))
            else:
                embed = torch.cat((states[0], embed), dim=-1)

            new_state = embed[:, :, -self.context_size + 1 :]
            embed = self.conv(embed).transpose(1, 2)
        else:
            new_state = embed.new_empty((embed.size(0), embed.size(2), 0))

        out = self.hid_act(embed)
        if self.output_proj:
            out = self.output_proj(out)

        return out, (new_state,)

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
        override_dropouts: bool = False,
        embed_dropout_rate: float = 0.0,
    ) -> None:
        """Update predictor dropout settings.

        Args:
          override_dropouts: If True, apply provided dropout values.
          embed_dropout_rate: New embedding dropout probability.
        """
        logging.info("changing predictor config")

        if override_dropouts:
            logging.info("overriding predictor dropouts")
            self.embed_dropout_rate = embed_dropout_rate
            self.embed_dropout = nn.Dropout(self.embed_dropout_rate)
