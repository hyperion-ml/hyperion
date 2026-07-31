"""
Copyright 2024 Johns Hopkins University  (Author: Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Union

from jsonargparse import ActionParser, ArgumentParser

from ...narchs import AudioFeatsMVN
from ..transducer import ConformerV1RNNTransducer
from .wav2rnn_transducer import Wav2RNNTransducer


class Wav2ConformerV1RNNTransducer(Wav2RNNTransducer):
    """Class for RNN-T with a ConformerV1 encoder and acoustic feature input.

    Attributes:
      feats: Audio feature extractor object or configuration dictionary.
      transducer: Transducer configuration dictionary or object.
    """

    def __init__(
        self,
        feats: Union[Dict[str, Any], AudioFeatsMVN],
        transducer: Union[Dict[str, Any], ConformerV1RNNTransducer],
    ) -> None:
        """Initializes the wrapper.

        Args:
          feats: Audio feature extractor instance or configuration dictionary.
          transducer: Backend transducer instance or configuration dictionary.
        """

        if isinstance(transducer, dict):
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer = ConformerV1RNNTransducer(**transducer)
        else:
            assert isinstance(transducer, ConformerV1RNNTransducer)

        super().__init__(feats, transducer)

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds wrapper CLI arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Wav2RNNTransducer.add_class_args(parser)
        ConformerV1RNNTransducer.add_class_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters fine-tuning arguments from a configuration dictionary.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Subset of fine-tuning arguments accepted by this wrapper.
        """
        base_args = {}
        child_args = ConformerV1RNNTransducer.filter_finetune_args(
            **kwargs["transducer"]
        )
        base_args["transducer"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds fine-tuning CLI arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        ConformerV1RNNTransducer.add_finetune_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
