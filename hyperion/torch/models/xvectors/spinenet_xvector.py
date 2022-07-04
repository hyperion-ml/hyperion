"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Copyright 2020 Magdalena Rybicka
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser
import logging

import torch
import torch.nn as nn

from .xvector import XVector
from ...narchs import SpineNetFactory as SNF


class SpineNetXVector(XVector):
    def __init__(
        self,
        spinenet_type,
        in_feats,
        num_classes,
        in_channels,
        output_levels=[3, 4, 5, 6, 7],
        endpoints_num_filters=256,
        resample_alpha=0.5,
        block_repeats=1,
        filter_size_scale=1.0,
        conv_channels=64,
        base_channels=64,
        in_kernel_size=7,
        in_stride=1,
        zero_init_residual=False,
        groups=1,
        do_maxpool=False,
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
        se_r=16,
        res2net_scale=4,
        res2net_width_factor=1,
    ):

        logging.info("making %s encoder network", spinenet_type)
        encoder_net = SNF.create(
            spinenet_type,
            in_channels,
            output_levels=output_levels,
            endpoints_num_filters=endpoints_num_filters,
            resample_alpha=resample_alpha,
            block_repeats=block_repeats,
            filter_size_scale=filter_size_scale,
            conv_channels=conv_channels,
            base_channels=base_channels,
            hid_act=hid_act,
            in_kernel_size=in_kernel_size,
            in_stride=in_stride,
            zero_init_residual=zero_init_residual,
            groups=groups,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
            in_norm=in_norm,
            se_r=se_r,
            in_feats=in_feats,
            res2net_scale=res2net_scale,
            res2net_width_factor=res2net_width_factor,
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
            in_feats=in_feats,
            proj_feats=proj_feats,
        )

        self.spinenet_type = spinenet_type

    @property
    def in_channels(self):
        return self.encoder_net.in_channels

    @property
    def output_levels(self):
        return self.encoder_net.output_levels

    @property
    def endpoints_num_filters(self):
        return self.encoder_net.endpoints_num_filters

    @property
    def resample_alpha(self):
        return self.encoder_net.resample_alpha

    @property
    def block_repeats(self):
        return self.encoder_net.block_repeats

    @property
    def filter_size_scale(self):
        return self.encoder_net.filter_size_scale

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

        pool_cfg = self.pool_net.get_config()

        config = {
            "spinenet_type": self.spinenet_type,
            "in_channels": self.in_channels,
            "output_levels": self.output_levels,
            "endpoints_num_filters": self.endpoints_num_filters,
            "resample_alpha": self.resample_alpha,
            "block_repeats": self.block_repeats,
            "filter_size_scale": self.filter_size_scale,
            "conv_channels": self.conv_channels,
            "base_channels": self.base_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "zero_init_residual": self.zero_init_residual,
            "groups": self.groups,
            "do_maxpool": self.do_maxpool,
            "in_norm": self.in_norm,
            "res2net_scale": self.res2net_scale,
            "res2net_width_factor": self.res2net_width_factor,
            "se_r": self.se_r,
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
        child_args = SNF.filter_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser)
        SNF.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = SNF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        SNF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
