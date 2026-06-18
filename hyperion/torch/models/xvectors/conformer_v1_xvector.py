"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import ConformerEncoderV1 as Encoder
from .xvector import XVector


class ConformerV1XVector(XVector):
    """x-vector model that wraps a Conformer encoder.

    Attributes:
        encoder_net: Encoder network inherited from ``XVector``.
        in_feats: Input feature dimension inherited from ``XVector``.
        proj: Optional encoder-to-pooling projection inherited from ``XVector``.
        proj_feats: Projection feature dimension inherited from ``XVector``.
        pool_net: Temporal pooling module inherited from ``XVector``.
        classif_net: Classification head inherited from ``XVector``.
        proj_head_net: Optional projection head inherited from ``XVector``.
        head_type: Head type inherited from ``XVector``.
        embed_dim: Embedding dimension inherited from ``XVector``.
        num_embed_layers: Number of embedding layers inherited from ``XVector``.
        dropout_rate: Head dropout inherited from ``XVector``.
        encoder: Saved encoder configuration when serialized.
        enc_d_model: Transformer/Conformer model dimension inherited from the encoder.
        num_enc_heads: Number of encoder heads inherited from the encoder.
        num_enc_blocks: Number of encoder blocks inherited from the encoder.
        enc_att_type: Attention type inherited from the encoder.
        enc_att_context: Local attention context inherited from the encoder.
        enc_ff_type: Feed-forward type inherited from the encoder.
        enc_d_ff: Feed-forward hidden dimension inherited from the encoder.
        enc_ff_kernel_size: Feed-forward kernel size inherited from the encoder.
        in_layer_type: Input layer type inherited from the encoder.
        enc_concat_after: Whether attention output is concatenated inherited from the encoder.
        pos_dropout_rate: Positional dropout inherited from the encoder.
        att_dropout_rate: Attention dropout inherited from the encoder.
        norm_layer: Normalization layer inherited from the encoder.
        head_norm_layer: Head normalization layer inherited from ``XVector``.
        use_norm: Whether normalization is enabled inherited from ``XVector``.
        norm_before: Whether normalization is applied before activation inherited from ``XVector``.
        head_use_norm: Whether the head uses normalization inherited from ``XVector``.
        head_use_in_norm: Whether the head normalizes inputs inherited from ``XVector``.
        head_hid_dim: DINO head hidden dimension inherited from ``XVector``.
        head_bottleneck_dim: DINO head bottleneck dimension inherited from ``XVector``.
        proj_head_use_norm: Whether the projection head uses normalization inherited from ``XVector``.
        proj_head_norm_before: Whether projection-head normalization is before activation inherited from ``XVector``.
    """

    def __init__(
        self,
        encoder: Union[Dict[str, Any], Encoder],
        num_classes: int,
        pool_net: Union[str, Dict[str, Any], nn.Module] = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[str, Dict[str, Any], Callable[..., nn.Module]] = {
            "name": "relu",
            "inplace": True,
        },
        loss_type: str = "arc-softmax",
        cos_scale: int = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        dropout_rate: float = 0,
        norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        head_norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        head_use_norm: bool = True,
        head_use_in_norm: bool = False,
        head_hid_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        proj_head_use_norm: bool = True,
        proj_head_norm_before: bool = True,
        embed_layer: int = 0,
        proj_feats: Optional[int] = None,
        head_type: str = "x-vector",
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Build a Conformer x-vector model.

        Args:
            encoder: Encoder module or configuration dictionary.
            num_classes: Number of output classes.
            pool_net: Pooling configuration.
            embed_dim: X-vector embedding dimension.
            num_embed_layers: Number of hidden layers in the head.
            hid_act: Hidden activation configuration.
            loss_type: Classification loss type.
            cos_scale: Scaling factor for angular-margin losses.
            margin: Margin for angular-margin losses.
            margin_warmup_epochs: Margin warmup duration in epochs.
            intertop_k: InterTopK penalty parameter.
            intertop_margin: InterTopK margin parameter.
            num_subcenters: Number of subcenters for subcenter losses.
            dropout_rate: Dropout rate used in the head.
            norm_layer: Normalization layer configuration.
            head_norm_layer: Normalization layer configuration for the head.
            use_norm: Whether to use normalization in auxiliary blocks.
            norm_before: Whether normalization is applied before activation.
            head_use_norm: Whether to use normalization in the head.
            head_use_in_norm: Whether to normalize head inputs.
            head_hid_dim: Hidden dimension for DINO heads.
            head_bottleneck_dim: Bottleneck dimension for DINO heads.
            proj_head_use_norm: Whether to normalize the projection head.
            proj_head_norm_before: Whether projection head normalization happens before activation.
            embed_layer: Head layer index used for embeddings.
            proj_feats: Optional projection feature dimension after the encoder.
            head_type: Classification head type.
            bias_weight_decay: Optional weight decay for bias parameters.
        """
        if isinstance(encoder, dict):
            logging.info(f"making conformer encoder network={encoder}")
            encoder["in_time_dim"] = 2
            encoder["out_time_dim"] = 2
            encoder = Encoder(**encoder)
        else:
            encoder.in_time_dim = 2
            encoder.out_time_dim = 2

        super().__init__(
            encoder,
            num_classes,
            pool_net=pool_net,
            embed_dim=embed_dim,
            num_embed_layers=num_embed_layers,
            hid_act=hid_act,
            loss_type=loss_type,
            cos_scale=cos_scale,
            margin=margin,
            margin_warmup_epochs=margin_warmup_epochs,
            intertop_k=intertop_k,
            intertop_margin=intertop_margin,
            num_subcenters=num_subcenters,
            norm_layer=norm_layer,
            head_norm_layer=head_norm_layer,
            use_norm=use_norm,
            norm_before=norm_before,
            head_use_norm=head_use_norm,
            head_use_in_norm=head_use_in_norm,
            head_hid_dim=head_hid_dim,
            head_bottleneck_dim=head_bottleneck_dim,
            proj_head_use_norm=proj_head_use_norm,
            proj_head_norm_before=proj_head_norm_before,
            dropout_rate=dropout_rate,
            embed_layer=embed_layer,
            proj_feats=proj_feats,
            head_type=head_type,
            bias_weight_decay=bias_weight_decay,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Model configuration dictionary.
        """
        base_config = super().get_config()
        del base_config["encoder_cfg"]
        del base_config["in_feats"]

        encoder_cfg = self.encoder_net.get_config()
        del encoder_cfg["class_name"]
        config = {
            "encoder": encoder_cfg,
        }

        config.update(base_config)
        return config

    def change_config(
        self,
        encoder: Dict[str, Any],
        override_output: bool = False,
        override_dropouts: bool = False,
        dropout_rate: float = 0,
        num_classes: Optional[int] = None,
        loss_type: str = "arc-softmax",
        cos_scale: int = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 10,
        intertop_k: int = 5,
        intertop_margin: float = 0,
        num_subcenters: int = 2,
    ) -> None:
        """Update the model configuration in place.

        Args:
            encoder: New encoder configuration.
            override_output: Whether to rebuild the output layer.
            override_dropouts: Whether to override dropout settings.
            dropout_rate: New dropout rate.
            num_classes: New number of classes.
            loss_type: New loss type.
            cos_scale: New scale value for angular-margin losses.
            margin: New margin value.
            margin_warmup_epochs: New margin warmup duration.
            intertop_k: New InterTopK parameter.
            intertop_margin: New InterTopK margin.
            num_subcenters: New number of subcenters.
        """
        super().change_config(
            override_output,
            False,
            dropout_rate,
            num_classes,
            loss_type,
            cos_scale,
            margin,
            margin_warmup_epochs,
            intertop_k,
            intertop_margin,
            num_subcenters,
        )
        if override_dropouts:
            logging.info("chaning x-vector head dropouts")
            self.classif_net.change_dropouts(dropout_rate)

        self.encoder_net.change_config(**encoder)

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "ConformerV1XVector":
        """Load a model from a config, state dict, or saved file.

        Args:
            file_path: Optional file path to load from.
            cfg: Optional configuration dictionary.
            state_dict: Optional state dictionary.

        Returns:
            Loaded model instance.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        try:
            del cfg["in_feats"]
        except:
            pass

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter constructor arguments for this model.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        base_args = XVector.filter_args(**kwargs)
        child_args = Encoder.filter_args(**kwargs["encoder"])

        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add constructor arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser, skip=set(["in_feats"]))
        Encoder.add_class_args(parser, prefix="encoder", skip=set())
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter arguments used for finetuning.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["encoder"])
        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add finetuning arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        Encoder.add_finetune_args(parser, prefix="encoder", skip=set())

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter arguments used for DINO teacher configuration.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        base_args = XVector.filter_dino_teacher_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["encoder"])
        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add DINO teacher arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_dino_teacher_args(parser)
        Encoder.add_finetune_args(parser, prefix="encoder", skip=set())

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
