"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..layer_blocks import TransformerConv2dSubsampler as Subsampler
from ..layers import ActivationFactory as AF
#from ..layers import NormLayer1dFactory as NLF
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


class RNNEncoder(NetArch):
    """ RNN Encoder network

    Attributeds:
      in_feats: input features
      hid_feats: hidden features in RNN layers
      out_feats: output features, if 0 we remove last projection layer
      num_layers: number of RNN layers
      proj_feats: projection features in LSTM layers
      rnn_type: type of RNN in [lstm, gru]
      bidirectional: whether RNN layers are bidirectional
      dropout_rate: dropout rate
      subsample_input: whether to subsample the input features time dimension x4
      subsampling_act: activation function of the subsampling block
    """

    def __init__(self,
                 in_feats: int,
                 hid_feats: int,
                 out_feats: int,
                 num_layers: int,
                 proj_feats: int = 0,
                 rnn_type: str = "lstm",
                 bidirectional: bool = False,
                 dropout_rate: float = 0.0,
                 subsample_input: bool = False,
                 subsampling_act: str = "relu6"):
        super().__init__()
        if rnn_type != "lstm":
            proj_feats = 0

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.proj_feats = proj_feats
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.subsample_input = subsample_input
        self.subsampling_act = subsampling_act

        rnn_feats = hid_feats if proj_feats == 0 else proj_feats
        if subsample_input:
            subsamplinb_act = AF.create(subsampling_act)
            self.subsampler = Subsampler(in_feats,
                                         hid_feats,
                                         hid_act=subsampling_act)
            lstm_in_dim = hid_feats
        else:
            self.subsampler = None
            lstm_in_dim = in_feats

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hid_feats,
                hidden_size=hid_feats,
                num_layers=num_layers,
                bias=True,
                proj_size=proj_feats,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.GRU(
                input_size=hid_feats,
                hidden_size=hid_feats,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional,
            )

        if out_feats > 0:
            self.output = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(rnn_feats, out_feats),
            )

    def forward(self, x: torch.Tensor,
                x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.subsample_input:
            t1 = x.size(1)
            x = self.subsampler(x)
            t2 = x.size(2)
            x_lengths = torch.div(t2 * x_lengths, t1, rounding_mode="floor")

        x = pack_padded_sequence(input=x,
                                 lengths=x_lengths.cpu(),
                                 batch_first=True,
                                 enforce_sorted=True)
        x, _ = self.rnn(x)
        x = pad_packed_sequence(x, batch_first=True)
        if self.out_feats > 0:
            x = self.output(x)

        return x, x_lengths

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, None, self.in_feats)

    def out_shape(self, in_shape=None):
        out_feats = self.out_feats if self.out_feats > 0 else (
            self.proj_feats if self.proj_feats > 0 else self.hid_feats)

        if in_shape is None:
            return (None, None, out_feats)

        assert len(in_shape) == 3
        return (*in_shape, out_feats)

    def get_config(self):
        config = filter_func_args(RNNEncoder.__init__, self.__dict__)
        base_config = super().get_config()
        base_config.update(config)
        return base_config
        #return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, override_dropouts, dropout_rate):
        if override_dropouts:
            logging.info("changing RNNEncoder dropouts")
            self.change_dropouts(dropout_rate)

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(RNNEncoder.__init__, **kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument("--in-feats",
                                type=int,
                                required=True,
                                help=("input feature dimension"))

        parser.add_argument(
            "--hid-feats",
            default=512,
            type=int,
            help=("num of hidden dimensions of RNN layers"),
        )

        parser.add_argument(
            "--out-feats",
            default=512,
            type=int,
            help=
            ("number of output dimensions of the encoder, if 0 output projection is removed"
             ),
        )

        parser.add_argument(
            "--proj-feats",
            default=512,
            type=int,
            help=("projection features of LSTM layers"),
        )

        parser.add_argument(
            "--num-layers",
            default=5,
            type=int,
            help=("number of RNN layers"),
        )

        parser.add_argument(
            "--in-kernel-size",
            default=3,
            type=int,
            help=("kernel size of input convolution"),
        )

        parser.add_argument(
            "--rnn-type",
            default="lstm",
            choices=[
                "lstm",
                "gru",
            ],
            help=("RNN type in [lstm, gru]"),
        )

        parser.add_argument(
            "--bidirectional",
            default=False,
            action=ActionYesNo,
            help="whether to use bidirectional RNN",
        )

        parser.add_argument(
            "--subsample-input",
            default=False,
            action=ActionYesNo,
            help="whether to subsaple input features x4",
        )
        parser.add_argument("--subsampling-act",
                            default="relu6",
                            help="activation for subsampler block")

        if "dropout_rate" not in skip:
            parser.add_argument("--dropout-rate",
                                default=0,
                                type=float,
                                help="dropout probability")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):

        valid_args = (
            "override_dropouts",
            "dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None, skip=set([])):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        try:
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        except:
            pass

        try:
            parser.add_argument("--dropout-rate",
                                default=0,
                                type=float,
                                help="dropout probability")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
