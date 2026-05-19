"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Set, Union

from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs import FeatFuserMVN, HydraHead, QFormerV2
from ...tpm import HFWav2Vec2
from .hf_wav2qvector import HFWav2QVector


class HFWav2Vec2QVector(HFWav2QVector):
    """Q-vector with Wav2Vec2 backbone.

    Attributes:
        hf_feats: Front-end/backbone module exposing HF-like hidden states.
        feat_fuser: Optional hidden-state fusion/mvn module configuration.
        hidden_feats_fusion_start: Index of the first hidden state used by the
            feature fuser when output features are aggregated.
        hidden_feats_agg_start: Index of the first hidden state used for hidden
            feature aggregation.
        hidden_feats_shared_adapters: Whether hidden-feature adapters are shared
            across selected hidden layers.
        backbone_layers: Indices of intermediate backbone layers returned during
            hidden-feature aggregation, or ``None`` when not requested.
        backbone_return_output: Flag indicating whether ``forward_hid_feats`` also
            returns the final backbone output tensor.
        hidden_feats_adapter: Optional projection layers that align backbone hidden
            features with the hidden Q-former input dimensionality.
        output_feats_adapter: Optional projection mapping backbone outputs to the
            output Q-former input space.
        backbone_feats_lr: Optional learning-rate override for feature-extractor
            parameters from ``hf_feats``.
        backbone_feats_weight_decay: Optional weight-decay override for feature-
            extractor parameters from ``hf_feats``.
        backbone_lr: Optional learning-rate override for encoder/backbone
            parameters from ``hf_feats``.
        backbone_weight_decay: Optional weight-decay override for encoder/backbone
            parameters from ``hf_feats``.
        adapter_lr: Optional learning-rate override for adapter parameters.
        adapter_weight_decay: Optional weight-decay override for adapter
            parameters.
        num_hidden_feats_queries: Number of learned queries used for hidden feature
            aggregation (inherited from ``QVector``).
        num_output_feats_queries: Number of learned queries used for output feature
            aggregation (inherited from ``QVector``).
        qvector_dim: Dimensionality of the flattened q-vector embedding (inherited).
        hidden_feats_agg_qformer: Q-former module that attends to intermediate
            backbone activations (inherited).
        output_feats_agg_qformer: Q-former operating on backbone outputs (inherited).
        hidden_feats_queries: Learnable queries feeding the hidden Q-former
            (inherited).
        output_feats_queries: Learnable queries feeding the output Q-former
            (inherited).
        proj_head: Projection head that flattens the concatenated Q-former outputs
            (inherited).
        head: Downstream Hydra head that produces logits or regression estimates
            (inherited).
    """

    def __init__(
        self,
        hf_feats: Union[Dict[str, Any], HFWav2Vec2],
        feat_fuser: Optional[Dict[str, Any]],
        hidden_feats_agg_qformer: Union[Dict[str, Any], QFormerV2, None],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Union[Dict[str, Any], None],
        num_output_feats_queries: int,
        qvector_dim: int,
        head: Union[Dict[str, Any], HydraHead],
        hidden_feats_fusion_start: int = 0,
        hidden_feats_agg_start: int = 0,
        hidden_feats_shared_adapters: bool = True,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        adapter_lr: Optional[float] = None,
        adapter_weight_decay: Optional[float] = None,
        qformer_weight_decay: Optional[float] = None,
        proj_head_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialise a Wav2Vec2-backed q-vector model.

        Args:
            hf_feats: Wav2Vec2 feature/backbone module or its config dictionary.
            feat_fuser: Optional configuration used to build
                :class:`FeatFuserMVN`. Required when
                ``output_feats_agg_qformer`` is provided.
            hidden_feats_agg_qformer: Hidden Q-former configuration or module.
            num_hidden_feats_queries: Number of hidden queries used for aggregation.
            output_feats_agg_qformer: Output Q-former configuration or module.
            num_output_feats_queries: Number of output queries.
            qvector_dim: Size of the final q-vector embedding.
            head: Hydra head configuration or module.
            hidden_feats_fusion_start: First hidden-state index used for output
                feature fusion.
            hidden_feats_agg_start: First hidden-state index used for hidden
                feature aggregation.
            hidden_feats_shared_adapters: If ``True``, hidden feature adapters
                share weights across selected layers.
            backbone_feats_lr: Optional learning-rate override for feature/backbone
                front-end parameters.
            backbone_feats_weight_decay: Optional weight-decay override for
                feature/backbone front-end parameters.
            backbone_lr: Optional learning-rate override for fused-backbone
                parameters.
            backbone_weight_decay: Optional weight-decay override for
                fused-backbone parameters.
            adapter_lr: Optional learning-rate override for adapter parameters
                (``hidden_feats_adapter`` and ``output_feats_adapter``).
            adapter_weight_decay: Optional weight-decay override for adapter
                parameters (``hidden_feats_adapter`` and ``output_feats_adapter``).
            qformer_weight_decay: Optional weight-decay override applied to both
                hidden/output Q-former parameters.
            proj_head_weight_decay: Optional weight-decay override applied to
                projection-head parameters.
            head_weight_decay: Optional weight-decay override applied to downstream
                head parameters.
            bias_weight_decay: Optional weight decay applied only to bias parameters.
        """
        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        super_args = filter_func_args(HFWav2QVector.__init__, locals())
        super().__init__(**super_args)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Return only keyword args matching this constructor signature."""
        base_args = HFWav2QVector.filter_args(**kwargs)
        hf_feats = kwargs.get("hf_feats", None)
        if isinstance(hf_feats, dict):
            child_args = HFWav2Vec2.filter_args(**hf_feats)
            base_args["hf_feats"] = child_args
        elif hf_feats is not None:
            base_args["hf_feats"] = hf_feats
        return base_args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI/configuration arguments for this model.

        Args:
            parser: ``ArgumentParser`` that receives the class arguments.
            prefix: Optional namespace prefix for grouped argument registration.
            skip: Optional set of argument names to exclude from registration.
        """
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        skip |= {
            "encoder_lr",
            "feat_extractor_lr",
            "bias_weight_decay",
            "encoder_weight_decay",
            "feat_extractor_weight_decay",
        }
        HFWav2Vec2.add_class_args(parser, prefix="hf_feats", skip=skip)
        HFWav2QVector.add_class_args(parser, skip=skip)

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Return fine-tuning keyword args relevant to this model."""
        base_args = HFWav2QVector.filter_finetune_args(**kwargs)
        hf_feats = kwargs.get("hf_feats", None)
        if isinstance(hf_feats, dict):
            child_args = HFWav2Vec2.filter_finetune_args(**hf_feats)
            base_args["hf_feats"] = child_args
        elif hf_feats is not None:
            base_args["hf_feats"] = hf_feats
        return base_args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register fine-tuning CLI/configuration arguments for this model.

        Args:
            parser: ``ArgumentParser`` that receives fine-tuning arguments.
            prefix: Optional namespace prefix for grouped argument registration.
            skip: Optional set of argument names to exclude from registration.
        """
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        skip |= {
            "encoder_lr",
            "feat_extractor_lr",
            "bias_weight_decay",
            "encoder_weight_decay",
            "feat_extractor_weight_decay",
        }

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats", skip=skip)
        HFWav2QVector.add_finetune_args(parser, skip=skip)
        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
