"""
 Copyright 2024 Johns Hopkins University  (Author: Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...tpm import HFWav2Vec2
from ..transducer import RNNRNNTransducer
from .wav2rnn_transducer import Wav2RNNTransducer


class Wav2RNNRNNTransducer(Wav2RNNTransducer):
    """Class for RNN-T LSTM encoder and acoustic feature input

    Attributes:
      Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      transducer: Transducer configuration dictionary or object.
    """

    def __init__(
        self,
        feats: Union[Dict, HFWav2Vec2],
        transducer: Union[Dict, RNNRNNTransducer],
    ):

        if isinstance(transducer, dict):
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer = RNNRNNTransducer(**transducer)
        else:
            assert isinstance(transducer, RNNRNNTransducer)

        super().__init__(feats, transducer)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2RNNTransducer.add_class_args(parser)
        RNNRNNTransducer.add_class_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = RNNRNNTransducer.filter_finetune_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNRNNTransducer.add_finetune_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
