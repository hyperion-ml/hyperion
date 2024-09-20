"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs.transformer_encoder_v2 import TransformerV2Encoder as Encoder
from ...utils import scale_seq_lengths
from .xvector import XVector


class TransformerV2XVector(XVector):
    def __init__(
        self,
        transformer_enc,
        num_classes,
        pool_net="mean+stddev",
        embed_dim=256,
        num_embed_layers=1,
        hid_act={"name": "relu", "inplace": True},
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
        dropout_rate=0,
        norm_layer=None,
        head_norm_layer=None,
        use_norm=True,
        norm_before=True,
        head_use_norm=True,
        head_use_in_norm=False,
        head_hid_dim=2048,
        head_bottleneck_dim=256,
        proj_head_use_norm=True,
        proj_head_norm_before=True,
        embed_layer=0,
        proj_feats=None,
        head_type="x-vector",
        bias_weight_decay=None,
    ):
        if isinstance(transformer_enc, dict):
            logging.info("making transformer v2 encoder network")
            transformer_enc = Encoder(**transformer_enc)

        super().__init__(
            transformer_enc,
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

    def get_config(self):
        base_config = super().get_config()
        del base_config["encoder_cfg"]
        del base_config["in_feats"]

        encoder_cfg = self.encoder_net.get_config()
        del encoder_cfg["class_name"]
        config = {
            "transformer_enc": encoder_cfg,
        }

        config.update(base_config)
        return config

    def change_config(
        self,
        transformer_enc,
        override_output=False,
        override_dropouts=False,
        dropout_rate=0,
        num_classes=None,
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=10,
        intertop_k=5,
        intertop_margin=0,
        num_subcenters=2,
    ):
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

        self.encoder_net.change_config(**transformer_enc)

    def _pre_enc(self, x):
        return x.transpose(1, 2).contiguous()

    def _post_enc(self, x, in_lengths=None, max_in_length=None):
        x = x.transpose(1, 2).contiguous()

        if self.proj is not None:
            x = self.proj(x)

        if in_lengths is not None:
            out_lengths = scale_seq_lengths(in_lengths, x.size(-1), max_in_length)
        else:
            out_lengths = None

        return x, out_lengths

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
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
    def filter_args(**kwargs):
        base_args = XVector.filter_args(**kwargs)
        child_args = Encoder.filter_args(**kwargs["transformer_enc"])
        if "in_feats" in base_args:
            del base_args["in_feats"]
        base_args["transformer_enc"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser, skip=set(["in_feats"]))
        Encoder.add_class_args(
            parser, prefix="transformer_enc", skip=set(["head_channels"])
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["transformer_enc"])
        base_args["transformer_enc"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        Encoder.add_finetune_args(
            parser, prefix="transformer_enc", skip=set(["head_channels"])
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs):
        base_args = XVector.filter_dinoteacher_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["transformer_enc"])
        base_args["transformer_enc"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_dino_teacher_args(parser)
        Encoder.add_finetune_args(
            parser, prefix="transformer_enc", skip=set(["head_channels"])
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
