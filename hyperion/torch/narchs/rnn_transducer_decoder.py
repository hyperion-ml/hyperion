"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from jsonargparse import ActionParser, ArgumentParser

import torch
import torch.nn as nn

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

from ...utils import filter_func_args
from ..layer_blocks import TransducerPredictor as Predictor, TransducerJoiner as Joiner
from .net_arch import NetArch


class RNNTransducerDecoder(NetArch):
    """ RNN-T Decoder composed of Predictor and Joiner networks
    Implementation based on 
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/transducer.py

    Attributes:
      in_feats: input features dimension (encoder output)
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the predictor input embedding.
      blank_id: The ID of the blank symbol.
      num_layers: Number of LSTM layers.
      hid_feats: Hidden dimension for predictor layers.
      embed_dropout_rate: Dropout rate for the embedding layer.
      rnn_dropout_rate: Dropout for LSTM layers.

    """

    def __init__(self,
                 in_feats: int,
                 vocab_size: int,
                 embed_dim: int,
                 num_pred_layers: int,
                 pred_hid_feats: int,
                 embed_dropout_rate: float = 0.0,
                 rnn_dropout_rate: float = 0.0,
                 rnn_type: str = "lstm",
                 blank_id: int = 0):

        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_pred_layers = num_pred_layers
        self.pred_hid_feats = pred_hid_feats
        self.embed_dropout_rate = embed_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.rnn_type = rnn_type
        self.blank_id = blank_id

        pred_args = filter_func_args(Predictor.__init__, locals())
        pred_args["num_layers"] = num_pred_layers
        pred_args["hid_feats"] = pred_hid_feats
        pred_args["out_feats"] = in_feats
        self.predictor = Predictor(**pred_args)
        self.joiner = Joiner(in_feats, vocab_size)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                y: k2.RaggedTensor) -> torch.Tensor:

        # get y_lengths
        row_splits = y.shape.row_splits(1)
        y_lengths = row_splits[1:] - row_splits[:-1]

        # shift y adding <sos> token
        sos_y = add_sos(y, sos_id=self.blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        # apply predictor and joiner
        pred_out, _ = self.predictor(sos_y_padded)
        logits = self.joiner(x, pred_out)

        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)
        x_lengths = x_lengths.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded.to(torch.int32),
            logit_lengths=x_lengths,
            target_lengths=y_lengths,
            blank=blank_id,
            reduction="sum",
        )
        return loss
