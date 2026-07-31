"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..xvectors import TransformerV2XVector
from .wav2xvector import Wav2XVector


class Wav2TransformerV2XVector(Wav2XVector):
    """Wrapper that combines waveform features with a Transformer V2 x-vector backend.

    Attributes:
      feats: Acoustic feature extractor or configuration dictionary.
      xvector: Transformer V2 backend or configuration dictionary.
    """

    def __init__(
        self,
        feats: Any,
        xvector: Union[Dict[str, Any], TransformerV2XVector],
    ) -> None:
        """Initializes the wrapper.

        Args:
          feats: Acoustic feature extractor instance or configuration dictionary.
          xvector: Transformer V2 backend instance or configuration dictionary.
        """
        if isinstance(xvector, dict):
            xvector = TransformerV2XVector.filter_args(**xvector)
            xvector = TransformerV2XVector(**xvector)
        else:
            assert isinstance(xvector, TransformerV2XVector)

        super().__init__(feats, xvector)

    @staticmethod
    def add_class_args(
        parser: Any,
        prefix: Optional[str] = None,
    ) -> None:
        """Adds CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2XVector.add_class_args(parser)
        TransformerV2XVector.add_class_args(parser, prefix="xvector")

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
        child_args = TransformerV2XVector.filter_finetune_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(
        parser: Any,
        prefix: Optional[str] = None,
    ) -> None:
        """Adds fine-tuning CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        TransformerV2XVector.add_finetune_args(parser, prefix="xvector")

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
        child_args = TransformerV2XVector.filter_dino_teacher_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(
        parser: Any,
        prefix: Optional[str] = None,
    ) -> None:
        """Adds DINO-teacher CLI arguments for this wrapper.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        TransformerV2XVector.add_dino_teacher_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
