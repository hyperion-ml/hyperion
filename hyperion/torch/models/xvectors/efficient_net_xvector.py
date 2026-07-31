"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import EfficientNet as EN
from .xvector import XVector


class EfficientNetXVector(XVector):
    """x-vector model that wraps an EfficientNet encoder.

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
        effnet_type: Encoder factory key used to build the model.
        in_channels: Number of encoder input channels.
        in_conv_channels: Stem convolution width.
        in_kernel_size: Stem kernel size.
        in_stride: Stem stride.
        mbconv_repeats: MBConv repeat schedule.
        mbconv_channels: MBConv channel schedule.
        mbconv_kernel_sizes: MBConv kernel schedule.
        mbconv_strides: MBConv stride schedule.
        mbconv_expansions: MBConv expansion schedule.
        head_channels: Encoder head channel count.
        width_scale: EfficientNet width multiplier.
        depth_scale: EfficientNet depth multiplier.
        fix_stem_head: Whether stem/head widths are fixed.
        drop_connect_rate: Drop connect probability.
        se_r: Squeeze-excitation reduction ratio.
        time_se: Whether time squeeze-excitation is enabled.
    """

    def __init__(
        self,
        effnet_type: str,
        in_feats: int,
        num_classes: int,
        in_channels: int = 1,
        in_conv_channels: int = 32,
        in_kernel_size: int = 3,
        in_stride: int = 2,
        mbconv_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        mbconv_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        mbconv_kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        mbconv_strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        mbconv_expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
        head_channels: int = 1280,
        width_scale: Optional[float] = None,
        depth_scale: Optional[float] = None,
        fix_stem_head: bool = False,
        se_r: int = 4,
        time_se: bool = False,
        pool_net: Union[str, Dict[str, Any], nn.Module] = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[str, Dict[str, Any], Callable[..., nn.Module]] = "swish",
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        drop_connect_rate: float = 0.2,
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
        """Build an EfficientNet x-vector model.

        Args:
            effnet_type: EfficientNet factory key.
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            in_conv_channels: Stem convolution width.
            in_kernel_size: Stem kernel size.
            in_stride: Stem stride.
            mbconv_repeats: Number of repeats for each MBConv stage.
            mbconv_channels: Channel widths for each MBConv stage.
            mbconv_kernel_sizes: Kernel sizes for each MBConv stage.
            mbconv_strides: Strides for each MBConv stage.
            mbconv_expansions: Expansion factors for each MBConv stage.
            head_channels: Encoder head channels.
            width_scale: EfficientNet width multiplier.
            depth_scale: EfficientNet depth multiplier.
            fix_stem_head: Whether to keep stem/head widths fixed.
            se_r: Squeeze-excitation reduction ratio.
            time_se: Whether to use time squeeze-excitation.
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
            drop_connect_rate: Drop connect probability.
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
        logging.info("making %s encoder network", effnet_type)
        encoder_net = EN(
            effnet_type,
            in_channels,
            in_conv_channels,
            in_kernel_size,
            in_stride,
            mbconv_repeats,
            mbconv_channels,
            mbconv_kernel_sizes,
            mbconv_strides,
            mbconv_expansions,
            head_channels,
            width_scale=width_scale,
            depth_scale=depth_scale,
            fix_stem_head=fix_stem_head,
            hid_act=hid_act,
            drop_connect_rate=drop_connect_rate,
            norm_layer=norm_layer,
            se_r=se_r,
            time_se=time_se,
            in_feats=in_feats,
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
            in_feats=in_feats,
            proj_feats=proj_feats,
            head_type=head_type,
            bias_weight_decay=bias_weight_decay,
        )

    @property
    def effnet_type(self):
        """Return the EfficientNet variant name.

        Returns:
            EfficientNet variant name.
        """
        return self.encoder_net.effnet_type

    @property
    def in_channels(self):
        """Return the number of input channels.

        Returns:
            Number of input channels.
        """
        return self.encoder_net.in_channels

    @property
    def in_conv_channels(self):
        """Return the stem convolution width.

        Returns:
            Stem convolution channel count.
        """
        return self.encoder_net.in_conv_channels

    @property
    def in_kernel_size(self):
        """Return the stem kernel size.

        Returns:
            Stem kernel size.
        """
        return self.encoder_net.in_kernel_size

    @property
    def in_stride(self):
        """Return the stem stride.

        Returns:
            Stem stride.
        """
        return self.encoder_net.in_stride

    @property
    def mbconv_repeats(self):
        """Return the MBConv repeat schedule.

        Returns:
            MBConv repeat schedule.
        """
        return self.encoder_net.mbconv_repeats

    @property
    def mbconv_channels(self):
        """Return the MBConv channel schedule.

        Returns:
            MBConv channel schedule.
        """
        return self.encoder_net.mbconv_channels

    @property
    def mbconv_kernel_sizes(self):
        """Return the MBConv kernel schedule.

        Returns:
            MBConv kernel schedule.
        """
        return self.encoder_net.mbconv_kernel_sizes

    @property
    def mbconv_strides(self):
        """Return the MBConv stride schedule.

        Returns:
            MBConv stride schedule.
        """
        return self.encoder_net.mbconv_strides

    @property
    def mbconv_expansions(self):
        """Return the MBConv expansion schedule.

        Returns:
            MBConv expansion schedule.
        """
        return self.encoder_net.mbconv_expansions

    @property
    def head_channels(self):
        """Return the encoder head channels.

        Returns:
            Encoder head channels.
        """
        return self.encoder_net.head_channels

    @property
    def width_scale(self):
        """Return the configured width multiplier.

        Returns:
            Width multiplier.
        """
        return self.encoder_net.width_scale

    @property
    def depth_scale(self):
        """Return the configured depth multiplier.

        Returns:
            Depth multiplier.
        """
        return self.encoder_net.depth_scale

    @property
    def fix_stem_head(self):
        """Return whether the stem/head widths are fixed.

        Returns:
            ``True`` if the stem/head widths are fixed.
        """
        return self.encoder_net.fix_stem_head

    @property
    def drop_connect_rate(self):
        """Return the drop connect rate.

        Returns:
            Drop connect rate.
        """
        return self.encoder_net.drop_connect_rate

    @property
    def se_r(self):
        """Return the squeeze-excitation reduction ratio.

        Returns:
            Squeeze-excitation reduction ratio.
        """
        return self.encoder_net.se_r

    @property
    def time_se(self):
        """Return whether time squeeze-excitation is enabled.

        Returns:
            ``True`` if time squeeze-excitation is enabled.
        """
        return self.encoder_net.time_se

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Model configuration dictionary.
        """
        base_config = super().get_config()
        del base_config["encoder_cfg"]

        config = {
            "effnet_type": self.effnet_type,
            "in_channels": self.in_channels,
            "in_conv_channels": self.encoder_net.b0_in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "mbconv_repeats": self.encoder_net.b0_mbconv_repeats,
            "mbconv_channels": self.encoder_net.b0_mbconv_channels,
            "mbconv_kernel_sizes": self.mbconv_kernel_sizes,
            "mbconv_strides": self.mbconv_strides,
            "mbconv_expansions": self.mbconv_expansions,
            "head_channels": self.head_channels,
            "width_scale": self.encoder_net.cfg_width_scale,
            "depth_scale": self.encoder_net.cfg_depth_scale,
            "fix_stem_head": self.fix_stem_head,
            "drop_connect_rate": self.drop_connect_rate,
            "se_r": self.se_r,
            "time_se": self.time_se,
        }

        config.update(base_config)
        return config

    def change_config(
        self,
        override_output: bool = False,
        override_dropouts: bool = False,
        dropout_rate: float = 0,
        drop_connect_rate: float = 0,
        bias_weight_decay: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update the model configuration in place.

        Args:
            override_output: Whether to rebuild the output layer.
            override_dropouts: Whether to override dropout settings.
            dropout_rate: New dropout rate.
            drop_connect_rate: New drop connect probability.
            bias_weight_decay: New weight decay for bias parameters.
            kwargs: Remaining encoder/head configuration values.
        """
        xvec_args = XVector.filter_finetune_args(**kwargs)
        xvec_args["override_dropouts"] = False
        xvec_args["bias_weight_decay"] = bias_weight_decay
        super().change_config(**xvec_args)

        if override_dropouts:
            self.encoder_net.change_dropouts(dropout_rate, drop_connect_rate)
            self.classif_net.change_dropouts(dropout_rate)

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "EfficientNetXVector":
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
        child_args = EN.filter_args(**kwargs)

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

        # we put args of EfficientNet first so it get swish as
        # default activation instead of relu
        EN.add_class_args(parser)
        XVector.add_class_args(parser)

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
        child_args = EN.filter_finetune_args(**kwargs)

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

        EN.add_finetune_args(parser)
        XVector.add_finetune_args(parser)

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
        child_args = EN.filter_finetune_args(**kwargs)

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

        EN.add_finetune_args(parser)
        XVector.add_dino_teacher_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
