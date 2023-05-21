"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Dict, Optional, Tuple, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch

from ...narchs import RNNEncoder
from .rnn_transducer import RNNTransducer


class RNNRNNTransducer(RNNTransducer):
    """RNN-T with RNN Encoder

    Attributes:
      encoder: dictionary of options to initialize RNNEncoder class or RNNEncoder object
      decoder: RNN-T Decoder config. dictionary or module.

    """

    def __init__(self, encoder, decoder):
        if isinstance(encoder, dict):
            encoder = RNNEncoder(**encoder)
        else:
            assert isinstance(encoder, RNNEncoder)

        super().__init__(encoder, decoder)

    @staticmethod
    def filter_args(**kwargs):
        args = RNNTransducer.filter_args(**kwargs)
        encoder_args = RNNEncoder.filter_args(**kwargs["encoder"])
        args["encoder"] = encoder_args
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNEncoder.add_class_args(parser, prefix="encoder", skip=skip)
        RNNTransducer.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    def change_config(
        self,
        encoder,
        decoder,
    ):
        logging.info("changing transducer encoder config")
        self.encoder.change_config(**encoder)
        super().chage_config(**decoder)

    @staticmethod
    def filter_finetune_args(**kwargs):
        args = RNNTransducer.filter_finetune_args(**kwargs)
        encoder_args = RNNEncoder.filter_finetune_args(**kwargs["encoder"])
        args["encoder"] = encoder_args
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNEncoder.add_finetune_args(parser, prefix="encoder")
        RNNTransducer.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
