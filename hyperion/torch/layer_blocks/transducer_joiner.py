"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba, Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TransducerJoiner(nn.Module):
    """ RNN-T Joiner network.
    Implementation based on 
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/joiner.py

    Attributes:
      in_feats: input feature dimension.
      vocab_size: vocabulary size
    """

    def __init__(self, in_feats: int, vocab_size: int):
        super().__init__()
        self.in_feats = in_feats
        self.vocab_size = vocab_size

        self.output = nn.Linear(in_feats, vocab_size)

    def forward(self, encoder_out: torch.Tensor,
                pred_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
          encoder_out: Output from the encoder with shape = (N, T, C).
          pred_out: Output from the predictor with shape = (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        assert encoder_out.ndim == pred_out.ndim == 3
        assert encoder_out.size(0) == pred_out.size(0)
        assert encoder_out.size(2) == pred_out.size(2)

        encoder_out = encoder_out.unsqueeze(2)
        # Now encoder_out is (N, T, 1, C)
        pred_out = pred_out.unsqueeze(1)
        # Now pred_out is (N, 1, U, C)
        x = torch.tanh(encoder_out + pred_out)

        logits = self.output(x)
        return logits

    # def get_config(self):
    #     config = {
    #         "in_feats": self.in_feats,
    #         "out_dims": self.out_dims,
    #         "num_layers": self.num_layers,
    #     }

    #     # base_config = super().get_config()
    #     return dict(list(config.items()))

    # @staticmethod
    # def filter_args(**kwargs):
    #     valid_args = (
    #         "in_feats",
    #         "out_dims",
    #         "num_layers",
    #     )
    #     args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    #     return args

    # @staticmethod
    # def add_class_args(parser,
    #                    prefix=None,
    #                    skip=set(["in_feats", "out_dims"])):
    #     if prefix is not None:
    #         outer_parser = parser
    #         parser = ArgumentParser(prog="")

    #     if "in_feats" not in skip:
    #         parser.add_argument("--in-feats",
    #                             type=int,
    #                             required=True,
    #                             help=("input feature dimension"))

    #     if "out_dims" not in skip:
    #         parser.add_argument("--out-dims",
    #                             type=int,
    #                             required=True,
    #                             help=("output feature dimension (vocab size)"))
    #     parser.add_argument("--num-layers",
    #                         default=1,
    #                         type=int,
    #                         help=("layers of the joiner"))

    #     if prefix is not None:
    #         outer_parser.add_argument("--" + prefix,
    #                                   action=ActionParser(parser=parser))
