"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import TDNNFactory as TF
from .xvector import XVector


class TDNNXVector(XVector):
    """x-vector model that wraps a TDNN encoder.

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
        tdnn_type: Encoder factory key used to build the model.
    """

    def __init__(
        self,
        tdnn_type: str,
        num_enc_blocks: int,
        in_feats: int,
        num_classes: int,
        enc_hid_units: Any,
        enc_expand_units: Optional[Any] = None,
        kernel_size: int = 3,
        dilation: int = 1,
        dilation_factor: int = 1,
        pool_net: Union[str, Dict[str, Any], nn.Module] = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[str, Dict[str, Any], Callable[..., nn.Module]] = {
            "name": "relu6",
            "inplace": True,
        },
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        dropout_rate: float = 0,
        norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        head_norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        use_norm: bool = True,
        norm_before: bool = False,
        in_norm: bool = False,
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
        """Build a TDNN x-vector model.

        Args:
            tdnn_type: TDNN factory key.
            num_enc_blocks: Number of encoder blocks.
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            enc_hid_units: Encoder hidden unit specification.
            enc_expand_units: Optional encoder expansion specification.
            kernel_size: TDNN kernel size.
            dilation: TDNN dilation.
            dilation_factor: TDNN dilation factor.
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
            in_norm: Whether to normalize encoder input features.
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
        logging.info("making %s encoder network", tdnn_type)
        encoder_net = TF.create(
            tdnn_type,
            num_enc_blocks,
            in_feats,
            enc_hid_units,
            enc_expand_units,
            kernel_size=kernel_size,
            dilation=dilation,
            dilation_factor=dilation_factor,
            hid_act=hid_act,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            use_norm=use_norm,
            norm_before=norm_before,
            in_norm=in_norm,
        )

        super().__init__(
            encoder_net,
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
            in_feats=None,
            proj_feats=proj_feats,
            head_type=head_type,
            bias_weight_decay=bias_weight_decay,
        )

        self.tdnn_type = tdnn_type

    @property
    def num_enc_blocks(self):
        """Return the number of encoder blocks.

        Returns:
            Number of encoder blocks.
        """
        return self.encoder_net.num_blocks

    @property
    def enc_hid_units(self):
        """Return the encoder hidden unit specification.

        Returns:
            Encoder hidden unit specification.
        """
        return self.encoder_net.hid_units

    @property
    def enc_expand_units(self):
        """Return the encoder expansion specification.

        Returns:
            Encoder expansion specification, if available.
        """
        try:
            return self.encoder_net.expand_units
        except:
            return None

    @property
    def kernel_size(self):
        """Return the TDNN kernel size.

        Returns:
            TDNN kernel size.
        """
        return self.encoder_net.kernel_size

    @property
    def dilation(self):
        """Return the TDNN dilation.

        Returns:
            TDNN dilation.
        """
        return self.encoder_net.dilation

    @property
    def dilation_factor(self):
        """Return the TDNN dilation factor.

        Returns:
            TDNN dilation factor.
        """
        return self.encoder_net.dilation_factor

    @property
    def in_norm(self):
        """Return whether the input is normalized.

        Returns:
            ``True`` when input normalization is enabled.
        """
        return self.encoder_net.in_norm

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Model configuration dictionary.
        """
        base_config = super().get_config()
        del base_config["encoder_cfg"]

        pool_cfg = self.pool_net.get_config()

        config = {
            "tdnn_type": self.tdnn_type,
            "num_enc_blocks": self.num_enc_blocks,
            "in_feats": self.in_feats,
            "enc_hid_units": self.enc_hid_units,
            "enc_expand_units": self.enc_expand_units,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "dilation_factor": self.dilation_factor,
            "in_norm": self.in_norm,
        }

        config.update(base_config)
        return config

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "TDNNXVector":
        """Load a model from a config, state dict, or saved file.

        Args:
            file_path: Optional file path to load from.
            cfg: Optional configuration dictionary.
            state_dict: Optional state dictionary.

        Returns:
            Loaded model instance.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

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
        child_args = TF.filter_args(**kwargs)

        base_args.update(child_args)
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

        XVector.add_class_args(parser)
        TF.add_class_args(parser)

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
        child_args = TF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
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
        TF.add_finetune_args(parser)

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
        child_args = TF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
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
        TF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
