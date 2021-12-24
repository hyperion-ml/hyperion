"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from argparse import Namespace

import torch
import torch.nn as nn

from .xvector import XVector
from ..narchs import ResNetFactory as RNF


class ResNetXVector(XVector):
    def __init__(
        self,
        in_feats,
        num_classes,
        resnet_cfg=Namespace(
            resnet_type="resnet34",
            in_channels=1,
            conv_channels=64,
            base_channels=64,
            in_kernel_size=7,
            in_stride=1,
            zero_init_residual=False,
            groups=1,
            replace_stride_with_dilation=None,
            do_maxpool=False,
            hid_act={"name": "relu", "inplace": True},
            dropout_rate=0,
            norm_layer=None,
            use_norm=True,
            norm_before=True,
            in_norm=False,
            se_r=16,
            res2net_scale=4,
            res2net_width_factor=1,
        ),
        conformer_cfg=Namespace(
            d_model=256,
            num_heads=4,
            num_blocks=6,
            attype="scaled-dot-prod-v1",
            atcontext=25,
            conv_repeats=1,
            conv_kernel_sizes=31,
            conv_strides=1,
            ff_type="linear",
            d_ff=2048,
            ff_kernel_size=1,
            dropourate=0.1,
            pos_dropourate=0.1,
            att_dropout_rate=0.0,
            in_layer_type="conv2d-sub",
            rel_pos_enc=True,
            causal_pos_enc=False,
            no_pos_enc=False,
            hid_act="swish",
            conv_norm_layer=None,
            se_r=None,
            ff_macaron=True,
            red_lnorms=False,
            concat_after=False,
        ),
        pool_net="mean+stddev",
        head_cfg=Namespace(
            embed_dim=256,
            num_embed_layers=1,
            head_hid_act={"name": "relu", "inplace": True},
            loss_type="arc-softmax",
            s=64,
            margin=0.3,
            margin_warmup_epochs=0,
            num_subcenters=2,
            norm_layer=None,
            use_norm=True,
            norm_before=True,
            dropout_rate=0,
            embed_layer=0,
        ),
    ):

        logging.info("making %s encoder network" % (resnet_type))
        if isinstance(resnet_cfg, Namespace):
            resnet_cfg = var(resnet_cfg)

        self.resnet_type = resnet_cfg["resnet_type"]
        encoder_net = RNF.create(**resnet_cfg)

        super().__init__(
            encoder_net,
            num_classes,
            conformer_cfg=conformer_cfg,
            pool_net=pool_net,
            head_cfg=head_cfg,
            in_feats=in_feats,
            proj_feats=None,
        )

    @property
    def in_channels(self):
        return self.encoder_net.in_channels

    @property
    def conv_channels(self):
        return self.encoder_net.conv_channels

    @property
    def base_channels(self):
        return self.encoder_net.base_channels

    @property
    def in_kernel_size(self):
        return self.encoder_net.in_kernel_size

    @property
    def in_stride(self):
        return self.encoder_net.in_stride

    @property
    def zero_init_residual(self):
        return self.encoder_net.zero_init_residual

    @property
    def groups(self):
        return self.encoder_net.groups

    @property
    def replace_stride_with_dilation(self):
        return self.encoder_net.replace_stride_with_dilation

    @property
    def do_maxpool(self):
        return self.encoder_net.do_maxpool

    @property
    def in_norm(self):
        return self.encoder_net.in_norm

    @property
    def se_r(self):
        return self.encoder_net.se_r

    @property
    def res2net_scale(self):
        return self.encoder_net.res2net_scale

    @property
    def res2net_width_factor(self):
        return self.encoder_net.res2net_width_factor

    def get_config(self):

        base_config = super().get_config()
        del base_config["encoder_cfg"]
        enc_cfg = self.encoder_net.get_config()
        del enc_cfg["block"]
        del enc_cfg["out_units"]
        del enc_cfg["out_act"]
        enc_cfg["resnet_type"] = self.resnet_type

        base_config["resnet_cfg"] = enc_cfg

        return base_config

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):

        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    def filter_args(prefix=None, **kwargs):

        base_args = XVector.filter_args(prefix, **kwargs)
        child_args = RNF.filter_args(prefix, **kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_argparse_args(parser, prefix=None):

        XVector.add_argparse_args(parser, prefix)
        if prefix is None:
            prefix = "resnet"
        else:
            prefix = prefix + "-resnet"
        RNF.add_argparse_args(parser, prefix)
