"""
Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Union

from jsonargparse import ActionParser, ArgumentParser

from ...tpm import HFWav2Vec2
from ..transducer import RNNTransducer
from .hf_wav2rnn_transducer import HFWav2RNNTransducer


class HFWav2Vec2RNNTransducer(HFWav2RNNTransducer):
    """RNN-T wrapper with Wav2Vec2 front-end features.

    Attributes:
      hf_feats: Hugging Face Wav2Vec2 wrapper or configuration dictionary.
      transducer: Transducer configuration dictionary or object.
      feat_fusion_start: First hidden-state layer used for fusion.
      feat_fusion_method: Hidden-state fusion method.
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFWav2Vec2],
        transducer: Union[Dict[str, Any], RNNTransducer],
        feat_fusion_start: int = 0,
        feat_fusion_method: str = "weighted-avg",
    ) -> None:
        """Initializes the wrapper.

        Args:
          hf_feats: Hugging Face Wav2Vec2 wrapper or configuration dictionary.
          transducer: Backend transducer instance or configuration dictionary.
          feat_fusion_start: First hidden-state layer used for fusion.
          feat_fusion_method: Hidden-state fusion method.
        """

        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        super().__init__(hf_feats, transducer, feat_fusion_start, feat_fusion_method)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments from a configuration dictionary.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Subset of arguments accepted by this wrapper.
        """
        base_args = HFWav2RNNTransducer.filter_args(**kwargs)
        child_args = HFWav2Vec2.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNTransducer.filter_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        return base_args

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

        HFWav2Vec2.add_class_args(parser, prefix="hf_feats")
        RNNTransducer.add_class_args(parser, prefix="transducer")
        HFWav2RNNTransducer.add_class_args(parser)

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
        child_args = HFWav2Vec2.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNTransducer.filter_finetune_args(**kwargs["transducer"])
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

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats")
        RNNTransducer.add_finetune_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
