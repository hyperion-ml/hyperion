"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..xvectors import ConformerV1XVector
from .wav2xvector import Wav2XVector


class Wav2ConformerV1XVector(Wav2XVector):
    """Wrapper that combines waveform features with a Conformer x-vector backend.

    Attributes:
      feats: Acoustic feature extractor or configuration dictionary.
      xvector: Conformer backend or configuration dictionary.
    """

    def __init__(
        self,
        feats: Any,
        xvector: Union[Dict[str, Any], ConformerV1XVector],
    ) -> None:
        """Initializes the wrapper.

        Args:
          feats: Acoustic feature extractor instance or configuration dictionary.
          xvector: Conformer x-vector backend instance or configuration dictionary.
        """
        if isinstance(xvector, dict):
            xvector = ConformerV1XVector.filter_args(**xvector)
            xvector = ConformerV1XVector(**xvector)
        else:
            assert isinstance(xvector, ConformerV1XVector)

        super().__init__(feats, xvector)

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2XVector.add_class_args(parser)
        ConformerV1XVector.add_class_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters fine-tuning configuration for this wrapper.

        Args:
          kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
        base_args: Dict[str, Any] = {}
        child_args = ConformerV1XVector.filter_finetune_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds fine-tuning CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConformerV1XVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters DINO-teacher configuration for this wrapper.

        Args:
          kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
        base_args: Dict[str, Any] = {}
        child_args = ConformerV1XVector.filter_dino_teacher_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds DINO-teacher CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConformerV1XVector.add_dino_teacher_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
