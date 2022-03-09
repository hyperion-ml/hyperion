"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from .wav2xvector import Wav2XVector
from ..xvectors import ResNet1dXVector


class Wav2ResNet1dXVector(Wav2XVector):
    """Class extracting ResNet1d x-vectors from waveform.
    It contains acoustic feature extraction, feature normalization and
    ResNet1dXVector extractor.

    Attributes:
      Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      xvector: ResNet1dXVector configuration dictionary or object.
    """

    def __init__(self, feats, xvector):

        if isinstance(xvector, dict):
            xvector = ResNet1dXVector.filter_args(**xvector)
            xvector = ResNet1dXVector(**xvector)
        else:
            assert isinstance(xvector, ResNet1dXVector)

        super().__init__(feats, xvector)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds Wav2ResNet1dXVector options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2XVector.add_class_args(parser)
        ResNet1dXVector.add_class_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
