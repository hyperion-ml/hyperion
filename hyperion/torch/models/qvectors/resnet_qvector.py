"""
 Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Dict, Optional, Union, Any


import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import ResNetFactory as RNF
from .qvector import QVector


class ResNetQVector(QVector):
    def __init__(
        self,
        in_feats: int,
        resnet_enc: Dict[str, AnyType],
        hidden_feats_agg_qformer: Dict[str, Any],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Dict[str, Any],
        num_output_feats_queries: int,
        qvector_dim: int,
        classif_head: Dict[str, Any],
        bias_weight_decay=None,
    ):
        logging.info("making %s encoder network", resnet_type)
        encoder_net = RNF.create(**resnet_enc)
        self.in_feats = in_feats

        super().__init__(
            encoder_net,
            hidden_feats_agg_qformer,
            num_hidden_feats_queries,
            output_feats_agg_qformer,
            num_output_feats_queries,
            qvector_dim,
            classif_head,
            bias_weight_decay=bias_weight_decay,
        )

    def _infer_enc_layers_indeces_and_dims(self, qformer_cfg):
        return_

    def get_config(self):
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
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs):
        base_args = XVector.filter_args(**kwargs)
        child_args = RNF.filter_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser)
        RNF.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = RNF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        RNF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs):
        base_args = XVector.filter_dino_teacher_args(**kwargs)
        child_args = RNF.filter_finetune_args(**kwargs)

        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_dino_teacher_args(parser)
        RNF.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
