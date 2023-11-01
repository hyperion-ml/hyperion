"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..xvectors import ConformerV1XVector
from .wav2xvector import Wav2XVector


class Wav2ConformerV1XVector(Wav2XVector):
    """Class extracting ConformerV1 x-vectors from waveform.
    It contains acoustic feature extraction, feature normalization and
    ConformerV1XVector extractor.

    Attributes:
      Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      xvector: ConformerV1XVector configuration dictionary or object.
    """

    def __init__(self, feats, xvector):
        if isinstance(xvector, dict):
            xvector = ConformerV1XVector.filter_args(**xvector)
            xvector = ConformerV1XVector(**xvector)
        else:
            assert isinstance(xvector, ConformerV1XVector)

        super().__init__(feats, xvector)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds Wav2ConformerV1XVector options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2XVector.add_class_args(parser)
        ConformerV1XVector.add_class_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = ConformerV1XVector.filter_finetune_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConformerV1XVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
