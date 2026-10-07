"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Set, Union

import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs import FeatFuserMVN, HydraHead
from ...tpm import HFWav2Vec2
from .hf_wav2xvectorp import HFWav2XVectorP


class HFWav2Vec2XVectorP(HFWav2XVectorP):
    """X-vector+ model backed by a Hugging Face Wav2Vec2 encoder.

    Attributes:
        hf_feats: Wav2Vec2 feature extractor and Transformer encoder.
        feat_fuser: Optional fuser for selected Wav2Vec2 hidden states.
        feat_fusion_start: First hidden-state index used by the fuser.
        pooling: Global pooling module (inherited).
        proj_head: Projection head (inherited).
        head: Optional downstream classification or regression head (inherited).
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFWav2Vec2],
        pooling: Union[str, Dict[str, Any], nn.Module],
        xvector_dim: int,
        head: Optional[Union[Dict[str, Any], HydraHead]],
        feat_fuser: Optional[Union[Dict[str, Any], FeatFuserMVN]] = None,
        feat_fusion_start: int = 0,
        proj_norm_layer: Optional[str] = None,
        proj_use_norm: bool = True,
        proj_norm_before: bool = True,
        enable_xvector_sig_reg: bool = False,
        xvector_sig_reg: Optional[Dict[str, Any]] = None,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        pooling_weight_decay: Optional[float] = None,
        proj_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialize the Wav2Vec2 x-vector+ model.

        Args:
            hf_feats: Wav2Vec2 module or its constructor configuration.
            pooling: Global-pooling type, configuration, or module.
            xvector_dim: Dimension of the final embedding.
            head: Hydra head configuration or module.
            feat_fuser: Optional hidden-state fusion configuration or module.
            feat_fusion_start: First hidden-state index used by the fuser.
            proj_norm_layer: Projection normalization type; ``None`` selects batch norm.
            proj_use_norm: Whether projection normalization is enabled.
            proj_norm_before: Whether normalization precedes the projection.
            backbone_feats_lr: Optional feature-extractor learning-rate override.
            backbone_feats_weight_decay: Optional feature-extractor weight decay.
            backbone_lr: Optional Transformer encoder learning-rate override.
            backbone_weight_decay: Optional Transformer encoder weight decay.
            enable_xvector_sig_reg: Whether to calculate SIGReg on projected embeddings.
            xvector_sig_reg: SIGReg constructor arguments.
            pooling_weight_decay: Optional weight decay for global pooling.
            proj_weight_decay: Optional weight decay for the projection head.
            head_weight_decay: Optional weight decay for the downstream head.
            bias_weight_decay: Optional weight decay for bias and normalization parameters.
        """
        if isinstance(hf_feats, dict):
            hf_config = dict(hf_feats)
            hf_config.pop("class_name", None)
            hf_feats = HFWav2Vec2(**HFWav2Vec2.filter_args(**hf_config))
        elif not isinstance(hf_feats, HFWav2Vec2):
            raise TypeError("hf_feats must be an HFWav2Vec2 module or configuration")

        super_args = filter_func_args(HFWav2XVectorP.__init__, locals())
        super().__init__(**super_args)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter configuration arguments for this wrapper.

        Args:
            **kwargs: Candidate model configuration values.

        Returns:
            Constructor-compatible keyword arguments with nested Wav2Vec2 options.
        """
        args = HFWav2XVectorP.filter_args(**kwargs)
        hf_feats = kwargs.get("hf_feats")
        if isinstance(hf_feats, dict):
            hf_config = dict(hf_feats)
            hf_config.pop("class_name", None)
            args["hf_feats"] = HFWav2Vec2.filter_args(**hf_config)
        elif hf_feats is not None:
            args["hf_feats"] = hf_feats
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register Wav2Vec2 and x-vector+ model arguments.

        Args:
            parser: Parser receiving the arguments.
            prefix: Optional namespace prefix.
            skip: Constructor argument names to omit.
        """
        skip = set(skip or ())
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None
        if "hf_feats" not in skip:
            HFWav2Vec2.add_class_args(parser, prefix="hf_feats", skip=skip)
        HFWav2XVectorP.add_class_args(parser, skip=skip)
        if outer_parser is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter fine-tuning settings for this wrapper.

        Args:
            **kwargs: Candidate fine-tuning configuration values.

        Returns:
            Model fine-tuning options with filtered Wav2Vec2 overrides.
        """
        args = HFWav2XVectorP.filter_finetune_args(**kwargs)
        hf_feats = kwargs.get("hf_feats")
        if isinstance(hf_feats, dict):
            args["hf_feats"] = HFWav2Vec2.filter_finetune_args(**hf_feats)
        elif hf_feats is not None:
            args["hf_feats"] = hf_feats
        return args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register Wav2Vec2 and x-vector+ fine-tuning options.

        Args:
            parser: Parser receiving the fine-tuning arguments.
            prefix: Optional namespace prefix.
            skip: Constructor argument names to omit.
        """
        skip = set(skip or ())
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None
        if "hf_feats" not in skip:
            HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats", skip=skip)
        HFWav2XVectorP.add_finetune_args(parser, skip=skip)
        if outer_parser is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
