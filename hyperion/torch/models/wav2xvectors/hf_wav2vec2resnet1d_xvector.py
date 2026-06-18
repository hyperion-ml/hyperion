"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Any, Dict, Optional, Union
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import FeatFuserMVN
from ...tpm import HFWav2Vec2
from ..xvectors import ResNet1dXVector
from .hf_wav2xvector import HFWav2XVector


class HFWav2Vec2ResNet1dXVector(HFWav2XVector):
    """Wrapper that combines Wav2Vec2 features with a ResNet1d x-vector backend.

    Attributes:
      hf_feats: Wav2Vec2 feature extractor or configuration dictionary.
      feat_fuser: Feature-fusion configuration dictionary.
      xvector: ResNet1d backend or configuration dictionary.
      feat_fusion_start: First Wav2Vec2 layer used by the feature fuser.
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFWav2Vec2],
        feat_fuser: Union[Dict[str, Any], FeatFuserMVN],
        xvector: Union[Dict[str, Any], ResNet1dXVector],
        feat_fusion_start: int = 0,
    ) -> None:
        """Initializes the wrapper.

        Args:
          hf_feats: Wav2Vec2 feature extractor instance or configuration dictionary.
          feat_fuser: Feature-fusion configuration dictionary or object.
          xvector: ResNet1d backend instance or configuration dictionary.
          feat_fusion_start: First Wav2Vec2 layer used by the feature fuser.
        """
        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        if isinstance(xvector, dict):
            xvector["resnet_enc"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in xvector:
                del xvector["class_name"]
            xvector = ResNet1dXVector(**xvector)
        else:
            assert isinstance(xvector, ResNet1dXVector)
            assert xvector.encoder_net.in_feats == hf_feats.hidden_size

        super().__init__(hf_feats, feat_fuser, xvector, feat_fusion_start)
        # feat_fusion_method)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments for this wrapper.

        Args:
          kwargs: Candidate keyword arguments.

        Returns:
          Filtered configuration dictionary.
        """
        base_args = HFWav2XVector.filter_args(**kwargs)
        child_args = HFWav2Vec2.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dXVector.filter_args(**kwargs["xvector"])
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

        HFWav2Vec2.add_class_args(parser, prefix="hf_feats")
        ResNet1dXVector.add_class_args(parser, prefix="xvector")
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
        base_args: Dict[str, Any] = {}
        child_args = HFWav2Vec2.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dXVector.filter_finetune_args(**kwargs["xvector"])
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

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats")
        ResNet1dXVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
