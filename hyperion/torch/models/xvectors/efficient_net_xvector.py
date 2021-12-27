"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from .xvector import XVector
from ...narchs import EfficientNet as EN


class EfficientNetXVector(XVector):
    def __init__(
        self,
        effnet_type,
        in_feats,
        num_classes,
        in_channels=1,
        in_conv_channels=32,
        in_kernel_size=3,
        in_stride=2,
        mbconv_repeats=[1, 2, 2, 3, 3, 4, 1],
        mbconv_channels=[16, 24, 40, 80, 112, 192, 320],
        mbconv_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
        mbconv_strides=[1, 2, 2, 2, 1, 2, 1],
        mbconv_expansions=[1, 6, 6, 6, 6, 6, 6],
        head_channels=1280,
        width_scale=None,
        depth_scale=None,
        fix_stem_head=False,
        se_r=4,
        time_se=False,
        pool_net="mean+stddev",
        embed_dim=256,
        num_embed_layers=1,
        hid_act="swish",
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        num_subcenters=2,
        drop_connect_rate=0.2,
        dropout_rate=0,
        norm_layer=None,
        head_norm_layer=None,
        use_norm=True,
        norm_before=True,
        embed_layer=0,
        proj_feats=None,
    ):

        logging.info("making %s encoder network" % (effnet_type))
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

    @property
    def effnet_type(self):
        return self.encoder_net.effnet_type

    @property
    def in_channels(self):
        return self.encoder_net.in_channels

    @property
    def in_conv_channels(self):
        return self.encoder_net.in_conv_channels

    @property
    def in_kernel_size(self):
        return self.encoder_net.in_kernel_size

    @property
    def in_stride(self):
        return self.encoder_net.in_stride

    @property
    def mbconv_repeats(self):
        return self.encoder_net.mbconv_repeats

    @property
    def mbconv_channels(self):
        return self.encoder_net.mbconv_channels

    @property
    def mbconv_kernel_sizes(self):
        return self.encoder_net.mbconv_kernel_sizes

    @property
    def mbconv_strides(self):
        return self.encoder_net.mbconv_strides

    @property
    def mbconv_expansions(self):
        return self.encoder_net.mbconv_expansions

    @property
    def head_channels(self):
        return self.encoder_net.head_channels

    @property
    def width_scale(self):
        return self.encoder_net.width_scale

    @property
    def depth_scale(self):
        return self.encoder_net.depth_scale

    @property
    def depth_scale(self):
        return self.encoder_net.depth_scale

    @property
    def fix_stem_head(self):
        return self.encoder_net.fix_stem_head

    @property
    def drop_connect_rate(self):
        return self.encoder_net.drop_connect_rate

    @property
    def se_r(self):
        return self.encoder_net.se_r

    @property
    def time_se(self):
        return self.encoder_net.time_se

    def get_config(self):

        base_config = super().get_config()
        del base_config["encoder_cfg"]

        pool_cfg = self.pool_net.get_config()
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
            "depth_scale": self.encoder_net.cfg_width_scale,
            "fix_stem_head": self.fix_stem_head,
            "drop_connect_rate": self.drop_connect_rate,
            "se_r": self.se_r,
            "time_se": self.time_se,
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

    def filter_args(**kwargs):

        base_args = XVector.filter_args(**kwargs)
        child_args = EN.filter_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        # we put args of EfficientNet first so it get swish as
        # default activation instead of relu
        EN.add_class_args(parser)
        XVector.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='xvector options')

    add_argparse_args = add_class_args
