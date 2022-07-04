"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from .xvector import XVector
from ...narchs import ResNet1dEncoder as Encoder


class ResNet1dXVector(XVector):
    def __init__(
        self,
        resnet_enc,
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
        in_norm=False,
        embed_layer=0,
        proj_feats=None,
    ):

        if isinstance(resnet_enc, dict):
            logging.info("making %s resnet1d encoder network", resnet_enc["resb_type"])
            resnet_enc = Encoder(**resnet_enc)

        super().__init__(
            resnet_enc,
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
            dropout_rate=dropout_rate,
            embed_layer=embed_layer,
            proj_feats=proj_feats,
        )

    def get_config(self):

        base_config = super().get_config()
        del base_config["encoder_cfg"]
        del base_config["in_feats"]

        encoder_cfg = self.encoder_net.get_config()
        del encoder_cfg["class_name"]
        config = {
            "resnet_enc": encoder_cfg,
        }

        config.update(base_config)
        return config

    def change_config(
        self,
        resnet_enc,
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

        self.encoder_net.change_config(**resnet_enc)

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
        child_args = Encoder.filter_args(**kwargs["resnet_enc"])

        base_args["resnet_enc"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser, skip=set(["in_feats"]))
        Encoder.add_class_args(parser, prefix="resnet_enc", skip=set(["head_channels"]))
        # parser.link_arguments("in_feats", "resnet_enc.in_feats", apply_on="parse")
        # parser.link_arguments("norm_layer", "encoder_net.norm_layer", apply_on="parse")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["resnet_enc"])
        base_args["resnet_enc"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        Encoder.add_finetune_args(
            parser, prefix="resnet_enc", skip=set(["head_channels"])
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
