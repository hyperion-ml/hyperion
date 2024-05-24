"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..xvectors import ConvNext2dXVector
from .wav2xvector import Wav2XVector


class Wav2ConvNext2dXVector(Wav2XVector):
    """Class extracting ConvNext2d x-vectors from waveform.
    It contains acoustic feature extraction, feature normalization and
    ConvNext2dXVector extractor.

    Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      xvector: ConvNext2dXVector configuration dictionary or object.
    """

    def __init__(self, feats, xvector):
        if isinstance(xvector, dict):
            xvector = ConvNext2dXVector.filter_args(**xvector)
            xvector = ConvNext2dXVector(**xvector)
        else:
            assert isinstance(xvector, ConvNext2dXVector)

        super().__init__(feats, xvector)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds Wav2ConvNext2dXVector options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2XVector.add_class_args(parser)
        ConvNext2dXVector.add_class_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = ConvNext2dXVector.filter_finetune_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConvNext2dXVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs):
        base_args = {}
        child_args = ConvNext2dXVector.filter_dino_teacher_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConvNext2dXVector.add_dino_teacher_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
