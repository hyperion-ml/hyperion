"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import ResNetFactory as RNF
from .xvector import XVector


class ResNetXVector(XVector):
    """x-vector model that wraps a generic ResNet encoder.

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
        resnet_type: Encoder factory key used to build the model.
        in_channels: Number of input channels for the encoder.
        conv_channels: Encoder convolution channel width.
        base_channels: Encoder base channel width.
        in_kernel_size: Input kernel size.
        in_stride: Input stride.
        zero_init_residual: Whether residual branches are zero-initialized.
        groups: Number of convolution groups.
        replace_stride_with_dilation: Stride-to-dilation replacement policy.
        do_maxpool: Whether the stem uses max-pooling.
        in_norm: Whether the encoder normalizes inputs.
        se_r: Squeeze-excitation reduction ratio.
        res2net_scale: Res2Net scale factor.
        res2net_width_factor: Res2Net width factor.
        freq_pos_enc: Whether frequency positional encoding is enabled.
    """

    def __init__(
        self,
        resnet_type: str,
        in_feats: int,
        num_classes: int,
        in_channels: int,
        conv_channels: int = 64,
        base_channels: int = 64,
        in_kernel_size: int = 7,
        in_stride: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        replace_stride_with_dilation: Optional[Any] = None,
        do_maxpool: bool = False,
        pool_net: Union[str, Dict[str, Any], nn.Module] = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[str, Dict[str, Any], Callable[..., nn.Module]] = {
            "name": "relu",
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
        norm_before: bool = True,
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
        se_r: int = 16,
        res2net_scale: int = 4,
        res2net_width_factor: int = 1,
        freq_pos_enc: bool = False,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Build a ResNet x-vector model.

        Args:
            resnet_type: ResNet factory key.
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            in_channels: Number of input channels for the encoder.
            conv_channels: Encoder convolution channel width.
            base_channels: Encoder base channel width.
            in_kernel_size: Input kernel size.
            in_stride: Input stride.
            zero_init_residual: Whether to zero-initialize residual branches.
            groups: Number of convolution groups.
            replace_stride_with_dilation: Optional stride-to-dilation replacement.
            do_maxpool: Whether to use max-pooling in the stem.
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
            se_r: Squeeze-excitation reduction ratio.
            res2net_scale: Res2Net scale factor.
            res2net_width_factor: Res2Net width factor.
            freq_pos_enc: Whether to use frequency positional encoding.
            bias_weight_decay: Optional weight decay for bias parameters.
        """
        logging.info("making %s encoder network", resnet_type)
        encoder_net = RNF.create(
            resnet_type,
            in_channels,
            conv_channels=conv_channels,
            base_channels=base_channels,
            hid_act=hid_act,
            in_kernel_size=in_kernel_size,
            in_stride=in_stride,
            zero_init_residual=zero_init_residual,
            groups=groups,
            replace_stride_with_dilation=replace_stride_with_dilation,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
            in_norm=in_norm,
            se_r=se_r,
            in_feats=in_feats,
            res2net_scale=res2net_scale,
            res2net_width_factor=res2net_width_factor,
            freq_pos_enc=freq_pos_enc,
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

        self.resnet_type = resnet_type

    @property
    def in_channels(self):
        """Return the number of input channels.

        Returns:
            Number of input channels.
        """
        return self.encoder_net.in_channels

    @property
    def conv_channels(self):
        """Return the convolution channel width.

        Returns:
            Convolution channel width.
        """
        return self.encoder_net.conv_channels

    @property
    def base_channels(self):
        """Return the base channel width.

        Returns:
            Base channel width.
        """
        return self.encoder_net.base_channels

    @property
    def in_kernel_size(self):
        """Return the input kernel size.

        Returns:
            Input kernel size.
        """
        return self.encoder_net.in_kernel_size

    @property
    def in_stride(self):
        """Return the input stride.

        Returns:
            Input stride.
        """
        return self.encoder_net.in_stride

    @property
    def zero_init_residual(self):
        """Return whether residual branches are zero-initialized.

        Returns:
            ``True`` when residual branches are zero-initialized.
        """
        return self.encoder_net.zero_init_residual

    @property
    def groups(self):
        """Return the convolution group count.

        Returns:
            Number of convolution groups.
        """
        return self.encoder_net.groups

    @property
    def replace_stride_with_dilation(self):
        """Return the stride-to-dilation replacement policy.

        Returns:
            Replacement policy value.
        """
        return self.encoder_net.replace_stride_with_dilation

    @property
    def do_maxpool(self):
        """Return whether max-pooling is used in the stem.

        Returns:
            ``True`` when max-pooling is enabled.
        """
        return self.encoder_net.do_maxpool

    @property
    def in_norm(self):
        """Return whether the input is normalized.

        Returns:
            ``True`` when input normalization is enabled.
        """
        return self.encoder_net.in_norm

    @property
    def se_r(self):
        """Return the squeeze-excitation reduction ratio.

        Returns:
            Squeeze-excitation reduction ratio.
        """
        return self.encoder_net.se_r

    @property
    def res2net_scale(self):
        """Return the Res2Net scale factor.

        Returns:
            Res2Net scale factor.
        """
        return self.encoder_net.res2net_scale

    @property
    def res2net_width_factor(self):
        """Return the Res2Net width factor.

        Returns:
            Res2Net width factor.
        """
        return self.encoder_net.res2net_width_factor

    @property
    def freq_pos_enc(self):
        """Return whether frequency positional encoding is enabled.

        Returns:
            ``True`` when frequency positional encoding is enabled.
        """
        return self.encoder_net.freq_pos_enc

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Model configuration dictionary.
        """
        base_config = super().get_config()
        del base_config["encoder_cfg"]
        config = {
            "resnet_type": self.resnet_type,
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "base_channels": self.base_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "zero_init_residual": self.zero_init_residual,
            "groups": self.groups,
            "replace_stride_with_dilation": self.replace_stride_with_dilation,
            "do_maxpool": self.do_maxpool,
            "in_norm": self.in_norm,
            "se_r": self.se_r,
            "res2net_scale": self.res2net_scale,
            "res2net_width_factor": self.res2net_width_factor,
            "freq_pos_enc": self.freq_pos_enc,
        }

        config.update(base_config)
        return config

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "ResNetXVector":
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
        child_args = RNF.filter_args(**kwargs)

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
        RNF.add_class_args(parser)

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
        child_args = RNF.filter_finetune_args(**kwargs)

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
        RNF.add_finetune_args(parser)

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
        child_args = RNF.filter_finetune_args(**kwargs)

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
        RNF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
