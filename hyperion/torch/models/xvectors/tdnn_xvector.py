"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from .xvector import XVector
from ...narchs import TDNNFactory as TF


class TDNNXVector(XVector):
    def __init__(
        self,
        tdnn_type,
        num_enc_blocks,
        in_feats,
        num_classes,
        enc_hid_units,
        enc_expand_units=None,
        kernel_size=3,
        dilation=1,
        dilation_factor=1,
        pool_net="mean+stddev",
        embed_dim=256,
        num_embed_layers=1,
        hid_act={"name": "relu6", "inplace": True},
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
        norm_before=False,
        in_norm=False,
        embed_layer=0,
        proj_feats=None,
    ):

        logging.info("making %s encoder network" % (tdnn_type))
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
            dropout_rate=dropout_rate,
            embed_layer=embed_layer,
            in_feats=None,
            proj_feats=proj_feats,
        )

        self.tdnn_type = tdnn_type

    @property
    def num_enc_blocks(self):
        return self.encoder_net.num_blocks

    @property
    def enc_hid_units(self):
        return self.encoder_net.hid_units

    @property
    def enc_expand_units(self):
        try:
            return self.encoder_net.expand_units
        except:
            return None

    @property
    def kernel_size(self):
        return self.encoder_net.kernel_size

    @property
    def dilation(self):
        return self.encoder_net.dilation

    @property
    def dilation_factor(self):
        return self.encoder_net.dilation_factor

    @property
    def in_norm(self):
        return self.encoder_net.in_norm

    def get_config(self):

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
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs):
        base_args = XVector.filter_args(**kwargs)
        child_args = TF.filter_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser)
        TF.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = TF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        TF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
