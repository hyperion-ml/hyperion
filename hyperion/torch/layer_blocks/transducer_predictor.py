"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TransducerPredictor(nn.Module):
    """ RNN-T prediction network.
    Implmentation  based on:
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/decoder.py

    Attributes:
      vocab_size: Number of tokens of the modeling unit including blank.
      embed_dim: Dimension of the input embedding.
      blank_id: The ID of the blank symbol.
      num_layers: Number of LSTM layers.
      hid_feats: Hidden dimension of LSTM layers.
      out_feats: Output dimension of the predictor.
      embed_dropout_rate: Dropout rate for the embedding layer.
      rnn_dropout_rate: Dropout for LSTM layers.
           
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_layers: int,
                 hid_feats: int,
                 out_feats: int,
                 embed_dropout_rate: float = 0.0,
                 rnn_dropout_rate: float = 0.0,
                 rnn_type: str = "lstm",
                 blank_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embed_dim=embed_dim,
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
        self.embed_dropout_rate = embed_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.output = nn.Linear(hid_feats, in_feats)

    def forward(
        self,
        y: torch.Tensor,
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
        rnn_out, (h, c) = self.rnn(embed, states)
        out = self.output(rnn_out)

        return out, (h, c)

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "hid_feats": self.hid_feats,
            "embed_dropout_rate": self.embed_dropout_rate,
            "rnn_dropout_rate": self.rnn_dropout_rate,
        }

        # base_config = super().get_config()
        return dict(list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "in_feats",
            "blank_id",
            "vocab_size",
            "embed_dim",
            "num_layers",
            "hid_feats",
            "embed_dropout_rate",
            "rnn_dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def filter_finetune_args(**kwargs):
        valid_args = (
            "embed_dropout_rate",
            "rnn_dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser,
                       prefix=None,
                       skip=set(["in_feats", "blank_id", "vocab_size"])):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument("--in-feats",
                                type=int,
                                required=True,
                                help=("input feature dimension"))
        if "blank_id" not in skip:
            parser.add_argument("--blank-id",
                                type=int,
                                required=True,
                                help=("blank id from sp model"))
        if "vocab_size" not in skip:
            parser.add_argument("--vocab-size",
                                type=int,
                                required=True,
                                help=("output prediction dimension"))
        parser.add_argument("--embedding-dim",
                            default=1024,
                            type=int,
                            help=("feature dimension"))
        parser.add_argument("--embedding-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder input embeddings"))
        parser.add_argument("--rnn-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder RNN "))

        parser.add_argument("--num-layers", default=2, type=int, help=(""))

        parser.add_argument("--hidden-dim", default=512, type=int, help=(""))

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

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

    @staticmethod
    def add_finetune_args(parser,
                          prefix=None,
                          skip=set(["in_feats", "blank_id", "vocab_size"])):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--override-dropouts",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to use the dropout probabilities passed in the "
                "arguments instead of the defaults in the pretrained model."))
        parser.add_argument("--embedding-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder input embeddings"))
        parser.add_argument("--rnn-dropout-rate",
                            default=0.0,
                            type=float,
                            help=("dropout prob for decoder RNN "))

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
