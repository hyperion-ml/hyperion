# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser


# TODO(fangjun): Support switching between LSTM and GRU
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_id: int,
        num_layers: int,
        hidden_dim: int,
        in_feats: int,
        embedding_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          embedding_dim:
            Dimension of the input embedding.
          blank_id:
            The ID of the blank symbol.
          num_layers:
            Number of LSTM layers.
          hidden_dim:
            Hidden dimension of LSTM layers.
          output_dim:
            Output dimension of the decoder.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)
        # TODO(fangjun): Use layer normalized LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout_rate,
        )

        self.in_feats = in_feats
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dropout_rate = embedding_dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.output_linear = nn.Linear(hidden_dim, in_feats)

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U) with BOS prepended.
          states:
            A tuple of two tensors containing the states information of
            LSTM layers in this decoder.
        Returns:
          Return a tuple containing:

            - rnn_output, a tensor of shape (N, U, C)
            - (h, c), containing the state information for LSTM layers.
              Both are of shape (num_layers, N, C)
        """
        embedding_out = self.embedding(y)
        embedding_out = self.embedding_dropout(embedding_out)
        #print("yy", y.shape, embedding_out.shape, y)
        rnn_out, (h, c) = self.rnn(embedding_out, states)
        out = self.output_linear(rnn_out)

        return out, (h, c)

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "embedding_dropout_rate": self.embedding_dropout_rate,
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
            "embedding_dim",
            "num_layers",
            "hidden_dim",
            "embedding_dropout_rate",
            "rnn_dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args


    @staticmethod
    def filter_finetune_args(**kwargs):
        valid_args = (
            "embedding_dropout_rate",
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
        embedding_dropout_rate: float = 0.0,
        rnn_dropout_rate: float = 0.0,
    ):
        logging.info("changing decoder config")

        if override_dropouts:
            logging.info("overriding decoder dropouts")

            # for module in self.modules():
            #     if isinstance(module, DropConnect1d):
            #         module.p *= drop_connect_rate / self.drop_connect_rate

            self.rnn_dropout_rate = rnn_dropout_rate
            self.rnn.p = self.rnn_dropout_rate
            
            self.embedding_dropout_rate = embedding_dropout_rate
            self.embedding_dropout = nn.Dropout(self.embedding_dropout_rate)



    @staticmethod
    def add_finetune_args(parser,
                       prefix=None,
                       skip=set(["in_feats", "blank_id", "vocab_size"])):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--override-dropouts",
                            default=False,
                            action=ActionYesNo,
                            help=(
                                "whether to use the dropout probabilities passed in the "
                                "arguments instead of the defaults in the pretrained model."
                            ))
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
