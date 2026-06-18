f"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import TransformerEncoderV1 as TE
from .xvector import XVector


class TransformerV1XVector(XVector):
    """x-Vector with Transformer encoder.

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
        enc_d_model: Transformer block feature dimension.
        num_enc_heads: Number of attention heads.
        num_enc_blocks: Number of self-attention blocks.
        enc_att_type: Attention type.
        enc_att_context: Local attention context size.
        enc_ff_type: Feed-forward block type.
        enc_d_ff: Feed-forward hidden dimension.
        enc_ff_kernel_size: Feed-forward convolution kernel size.
        in_layer_type: Input layer type.
        enc_concat_after: Whether attention input/output are concatenated.
        pos_dropout_rate: Positional dropout rate.
        att_dropout_rate: Attention dropout rate.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        enc_d_model: int = 512,
        num_enc_heads: int = 4,
        num_enc_blocks: int = 6,
        enc_att_type: str = "scaled-dot-prod-v1",
        enc_att_context: int = 25,
        enc_ff_type: str = "linear",
        enc_d_ff: int = 2048,
        enc_ff_kernel_size: int = 1,
        in_layer_type: str = "conv2d-sub",
        enc_concat_after: bool = False,
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
        dropout_rate: float = 0.1,
        pos_dropout_rate: float = 0.1,
        att_dropout_rate: float = 0.0,
        norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        head_norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        use_norm: bool = True,
        norm_before: bool = False,
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
        """Build a Transformer V1 x-vector model.

        Args:
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            enc_d_model: Transformer model dimension.
            num_enc_heads: Number of attention heads.
            num_enc_blocks: Number of encoder blocks.
            enc_att_type: Attention type.
            enc_att_context: Local attention context size.
            enc_ff_type: Feed-forward block type.
            enc_d_ff: Feed-forward hidden dimension.
            enc_ff_kernel_size: Feed-forward convolution kernel size.
            in_layer_type: Input layer type.
            enc_concat_after: Whether to concatenate attention input/output.
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
            pos_dropout_rate: Positional encoder dropout rate.
            att_dropout_rate: Attention dropout rate.
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
        logging.info("making transformer-v1 encoder network")
        encoder_net = TE(
            in_feats,
            enc_d_model,
            num_enc_heads,
            num_enc_blocks,
            att_type=enc_att_type,
            att_context=enc_att_context,
            ff_type=enc_ff_type,
            d_ff=enc_d_ff,
            ff_kernel_size=enc_ff_kernel_size,
            ff_dropout_rate=dropout_rate,
            pos_dropout_rate=pos_dropout_rate,
            att_dropout_rate=att_dropout_rate,
            in_layer_type=in_layer_type,
            norm_before=norm_before,
            concat_after=enc_concat_after,
            in_time_dim=-1,
            out_time_dim=-1,
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

    @property
    def enc_d_model(self):
        """Return the Transformer model dimension.

        Returns:
            Transformer model dimension.
        """
        return self.encoder_net.d_model

    @property
    def num_enc_heads(self):
        """Return the number of attention heads.

        Returns:
            Number of attention heads.
        """
        return self.encoder_net.num_heads

    @property
    def num_enc_blocks(self):
        """Return the number of encoder blocks.

        Returns:
            Number of encoder blocks.
        """
        return self.encoder_net.num_blocks

    @property
    def enc_att_type(self):
        """Return the attention type.

        Returns:
            Attention type string.
        """
        return self.encoder_net.att_type

    @property
    def enc_att_context(self):
        """Return the local attention context.

        Returns:
            Attention context size.
        """
        return self.encoder_net.att_context

    @property
    def enc_ff_type(self):
        """Return the feed-forward block type.

        Returns:
            Feed-forward block type string.
        """
        return self.encoder_net.ff_type

    @property
    def enc_d_ff(self):
        """Return the feed-forward hidden dimension.

        Returns:
            Feed-forward hidden dimension.
        """
        return self.encoder_net.d_ff

    @property
    def enc_ff_kernel_size(self):
        """Return the feed-forward kernel size.

        Returns:
            Feed-forward kernel size.
        """
        return self.encoder_net.ff_kernel_size

    @property
    def pos_dropout_rate(self):
        """Return the positional dropout rate.

        Returns:
            Positional dropout rate.
        """
        return self.encoder_net.pos_dropout_rate

    @property
    def att_dropout_rate(self):
        """Return the attention dropout rate.

        Returns:
            Attention dropout rate.
        """
        return self.encoder_net.att_dropout_rate

    @property
    def in_layer_type(self):
        """Return the input layer type.

        Returns:
            Input layer type string.
        """
        return self.encoder_net.in_layer_type

    @property
    def enc_concat_after(self):
        """Return whether attention output is concatenated.

        Returns:
            ``True`` when attention input and output are concatenated.
        """
        return self.encoder_net.concat_after

    @property
    def enc_ff_type(self):
        """Return the feed-forward block type.

        Returns:
            Feed-forward block type string.
        """
        return self.encoder_net.ff_type

    def get_config(self) -> Dict[str, Any]:
        """Gets network config
        Returns:
           dictionary with config params
        """
        base_config = super().get_config()
        del base_config["encoder_cfg"]

        pool_cfg = self.pool_net.get_config()

        config = {
            "num_enc_blocks": self.num_enc_blocks,
            "in_feats": self.in_feats,
            "enc_d_model": self.enc_d_model,
            "num_enc_heads": self.num_enc_heads,
            "enc_att_type": self.enc_att_type,
            "enc_att_context": self.enc_att_context,
            "enc_ff_type": self.enc_ff_type,
            "enc_d_ff": self.enc_d_ff,
            "enc_ff_kernel_size": self.enc_ff_kernel_size,
            "pos_dropout_rate": self.pos_dropout_rate,
            "att_dropout_rate": self.att_dropout_rate,
            "in_layer_type": self.in_layer_type,
            "enc_concat_after": self.enc_concat_after,
        }

        config.update(base_config)
        return config

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "TransformerV1XVector":
        """Loads model from file.

        Args:
            file_path: Optional file path to load from.
            cfg: Optional configuration dictionary.
            state_dict: Optional state dictionary.

        Returns:
            Loaded model instance.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        # fix to load old model
        if "d_enc_ff" in cfg:
            cfg["enc_d_ff"] = cfg["d_enc_ff"]
            del cfg["d_enc_ff"]
        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          prefix: prefix string
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        base_args = XVector.filter_args(**kwargs)

        valid_args = (
            "num_enc_blocks",
            "in_feats",
            "enc_d_model",
            "num_enc_heads",
            "enc_att_type",
            "enc_att_context",
            "enc_ff_type",
            "enc_d_ff",
            "enc_ff_kernel_size",
            "pos_dropout_rate",
            "att_dropout_rate",
            "in_layer_type",
            "enc_concat_after",
        )

        child_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds TransformerXVector config parameters to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser)
        parser.add_argument(
            "--num-enc-blocks",
            default=6,
            type=int,
            help=("number of tranformer blocks"),
        )

        parser.add_argument(
            "--enc-d-model", default=512, type=int, help=("encoder layer sizes")
        )

        parser.add_argument(
            "--num-enc-heads",
            default=4,
            type=int,
            help=("number of heads in self-attention layers"),
        )

        parser.add_argument(
            "--enc-att-type",
            default="scaled-dot-prod-v1",
            choices=["scaled-dot-prod-v1", "local-scaled-dot-prod-v1"],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--enc-att-context",
            default=25,
            type=int,
            help=("context size when using local attention"),
        )

        parser.add_argument(
            "--enc-ff-type",
            default="linear",
            choices=["linear", "conv1dx2", "conv1dlinear"],
            help=("type of feed forward layers in transformer block"),
        )

        parser.add_argument(
            "--enc-d-ff",
            default=2048,
            type=int,
            help=("size middle layer in feed forward block"),
        )

        parser.add_argument(
            "--enc-ff-kernel-size",
            default=3,
            type=int,
            help=("kernel size in convolutional feed forward block"),
        )

        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )

        parser.add_argument(
            "--in-layer-type",
            default="linear",
            choices=["linear", "conv2d-sub"],
            help=("type of input layer"),
        )

        parser.add_argument(
            "--enc-concat-after",
            default=False,
            action="store_true",
            help="concatenate attention input and output instead of adding",
        )

        # parser.add_argument('--in-norm', default=False, action='store_true',
        #                     help='batch normalization at the input')
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='xvector options')

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        base_args = XVector.filter_finetune_args(**kwargs)

        valid_args = (
            "pos_dropout_rate",
            "att_dropout_rate",
        )

        child_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds TransformerXVector config parameters for finetuning to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        base_args = XVector.filter_dino_teacher_args(**kwargs)

        valid_args = (
            "pos_dropout_rate",
            "att_dropout_rate",
        )

        child_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds TransformerXVector config parameters for finetuning to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_dino_teacher_args(parser)
        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
