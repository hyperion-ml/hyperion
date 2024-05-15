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
from ..transducer import ConformerV1RNNTransducer
from .wav2rnn_transducer import Wav2RNNTransducer


class Wav2ConformerV1RNNTransducer(Wav2RNNTransducer):
    """Class for RNN-T with ConformerV1 Encoder and acoustic feature input

    Attributes:
      Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      transducer: Transducer configuration dictionary or object.
    """

    def __init__(
        self,
        feats: Union[Dict, HFWav2Vec2],
        transducer: Union[Dict, ConformerV1RNNTransducer],
    ):

        if isinstance(transducer, dict):
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer = ConformerV1RNNTransducer(**transducer)
        else:
            assert isinstance(transducer, ConformerV1RNNTransducer)

        super().__init__(feats, transducer)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2RNNTransducer.add_class_args(parser)
        ConformerV1RNNTransducer.add_class_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = ConformerV1RNNTransducer.filter_finetune_args(
            **kwargs["transducer"]
        )
        base_args["transducer"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConformerV1RNNTransducer.add_finetune_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
