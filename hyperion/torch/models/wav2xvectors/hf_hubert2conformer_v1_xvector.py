"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import FeatFuserMVN
from ...tpm import HFHubert
from ..xvectors import ConformerV1XVector
from .hf_wav2xvector import HFWav2XVector


class HFHubert2ConformerV1XVector(HFWav2XVector):
    """Wrapper that combines HuBERT features with a Conformer backend.

    Attributes:
      hf_feats: HuBERT feature extractor or configuration dictionary.
      feat_fuser: Feature-fusion configuration dictionary.
      xvector: Conformer backend or configuration dictionary.
      feat_fusion_start: First HuBERT layer used by the feature fuser.
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFHubert],
        feat_fuser: Union[Dict[str, Any], FeatFuserMVN],
        xvector: Union[Dict[str, Any], ConformerV1XVector],
        feat_fusion_start: int = 0,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initializes the wrapper.

        Args:
          hf_feats: HuBERT feature extractor instance or configuration dictionary.
          feat_fuser: Feature-fusion configuration dictionary or object.
          xvector: Conformer backend instance or configuration dictionary.
          feat_fusion_start: First HuBERT layer used by the feature fuser.
        """
        if isinstance(hf_feats, dict):
            hf_feats = HFHubert(**hf_feats)
        else:
            assert isinstance(hf_feats, HFHubert)

        if isinstance(xvector, dict):
            xvector["encoder"]["in_feats"] = hf_feats.hidden_size
            xvector = ConformerV1XVector(**xvector)
        else:
            assert isinstance(xvector, ConformerV1XVector)
            assert xvector.encoder_net.in_feats == hf_feats.hidden_size

        super().__init__(
            hf_feats, feat_fuser, xvector, feat_fusion_start, bias_weight_decay
        )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments for this wrapper.

        Args:
          kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
        base_args = HFWav2XVector.filter_args(**kwargs)
        child_args = HFHubert.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ConformerV1XVector.filter_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

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

        HFHubert.add_class_args(parser, prefix="hf_feats")
        ConformerV1XVector.add_class_args(parser, prefix="xvector")
        HFWav2XVector.add_class_args(parser)

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
        base_args = {}
        child_args = HFHubert.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
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

        HFHubert.add_finetune_args(parser, prefix="hf_feats")
        ConformerV1XVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
