"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Union

from jsonargparse import ActionParser, ArgumentParser

from ...narchs import FeatFuserMVN
from ...tpm import HFWav2Vec2Bert
from ..xvectors import ConformerV1XVector
from .hf_wav2xvector import HFWav2XVector


class HFWav2Vec2Bert2ConformerV1XVector(HFWav2XVector):
    """Wrapper that combines Wav2Vec2-BERT features with a Conformer backend.

    Attributes:
      hf_feats: Wav2Vec2-BERT feature extractor or configuration dictionary.
      feat_fuser: Feature-fusion configuration dictionary.
      xvector: Conformer backend or configuration dictionary.
      feat_fusion_start: First Wav2Vec2-BERT layer used by the feature fuser.
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFWav2Vec2Bert],
        feat_fuser: Union[Dict[str, Any], FeatFuserMVN],
        xvector: Union[Dict[str, Any], ConformerV1XVector],
        feat_fusion_start: int = 0,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initializes the wrapper."""
        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2Bert(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2Bert)

        if isinstance(xvector, dict):
            xvector["encoder"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in xvector:
                del xvector["class_name"]
            xvector = ConformerV1XVector(**xvector)
        else:
            assert isinstance(xvector, ConformerV1XVector)
            assert xvector.encoder_net.in_feats == hf_feats.hidden_size

        super().__init__(
            hf_feats, feat_fuser, xvector, feat_fusion_start, bias_weight_decay
        )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        base_args = HFWav2XVector.filter_args(**kwargs)
        base_args["hf_feats"] = HFWav2Vec2Bert.filter_args(**kwargs["hf_feats"])
        base_args["xvector"] = ConformerV1XVector.filter_args(**kwargs["xvector"])
        return base_args

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2Bert.add_class_args(parser, prefix="hf_feats")
        ConformerV1XVector.add_class_args(parser, prefix="xvector")
        HFWav2XVector.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        base_args: Dict[str, Any] = {}
        base_args["hf_feats"] = HFWav2Vec2Bert.filter_finetune_args(
            **kwargs["hf_feats"]
        )
        base_args["xvector"] = ConformerV1XVector.filter_finetune_args(
            **kwargs["xvector"]
        )
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2Bert.add_finetune_args(parser, prefix="hf_feats")
        ConformerV1XVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
