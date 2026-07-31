"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Set, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch

from ...narchs import RNNEncoder, RNNTransducerDecoder
from .rnn_transducer import RNNTransducer


class RNNRNNTransducer(RNNTransducer):
    """RNN-T with an RNN encoder.

    Attributes:
      encoder: RNN encoder module.
      rnnt_decoder: RNN-T decoder module.
    """

    def __init__(
        self,
        encoder: Union[Dict[str, Any], RNNEncoder],
        rnnt_decoder: Union[Dict[str, Any], RNNTransducerDecoder],
    ) -> None:
        """Initializes the RNN-RNN transducer.

        Args:
          encoder: Encoder configuration dictionary or module instance.
          rnnt_decoder: Decoder configuration dictionary or module instance.
        """
        if isinstance(encoder, dict):
            encoder = RNNEncoder(**encoder)
        else:
            assert isinstance(encoder, RNNEncoder)

        super().__init__(encoder, rnnt_decoder)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters nested configuration dictionaries for construction.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Filtered configuration dictionary.
        """
        args = RNNTransducer.filter_args(**kwargs)
        encoder_args = RNNEncoder.filter_args(**kwargs["encoder"])
        args["encoder"] = encoder_args
        return args

    @staticmethod
    def add_class_args(
        parser: Any, prefix: Optional[str] = None, skip: Optional[Set[str]] = None
    ) -> None:
        """Registers CLI arguments for this class.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
          skip: Optional set of arguments to skip in nested registration.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNEncoder.add_class_args(parser, prefix="encoder", skip=skip)
        RNNTransducer.add_class_args(parser, skip=skip)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    def change_config(
        self,
        encoder: Dict[str, Any],
        rnnt_decoder: Dict[str, Any],
    ) -> None:
        """Applies runtime configuration changes to both submodules.

        Args:
          encoder: Encoder configuration updates.
          rnnt_decoder: Decoder configuration updates.
        """
        logging.info("changing transducer encoder config")
        self.encoder.change_config(**encoder)
        super().change_config(rnnt_decoder)

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters fine-tuning configuration dictionaries.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Filtered configuration dictionary.
        """
        args = RNNTransducer.filter_finetune_args(**kwargs)
        encoder_args = RNNEncoder.filter_finetune_args(**kwargs["encoder"])
        args["encoder"] = encoder_args
        return args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Registers fine-tuning CLI arguments.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNEncoder.add_finetune_args(parser, prefix="encoder")
        RNNTransducer.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
