"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser
import torch.nn as nn

from .global_pool import *


class GlobalPool1dFactory(object):
    @staticmethod
    def create(
        pool_type,
        in_feats=None,
        inner_feats=128,
        num_comp=64,
        dist_pow=2,
        use_bias=False,
        num_heads=8,
        d_k=256,
        d_v=256,
        bin_attn=False,
        use_global_context=True,
        norm_layer=None,
        dim=-1,
        keepdim=False,
        **kwargs
    ):

        if pool_type == "avg":
            return GlobalAvgPool1d(dim=dim, keepdim=keepdim)

        if pool_type == "mean+stddev":
            return GlobalMeanStdPool1d(dim=dim, keepdim=keepdim)

        if pool_type == "mean+logvar":
            return GlobalMeanLogVarPool1d(dim=dim, keepdim=keepdim)

        if pool_type == "lde":
            return LDEPool1d(
                in_feats,
                num_comp=num_comp,
                dist_pow=dist_pow,
                use_bias=use_bias,
                dim=dim,
                keepdim=keepdim,
            )

        if pool_type == "scaled-dot-prod-att-v1":
            return ScaledDotProdAttV1Pool1d(
                in_feats,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                bin_attn=bin_attn,
                dim=dim,
                keepdim=keepdim,
            )

        if pool_type in ["ch-wise-att-mean+stddev", "ch-wise-att-mean-stddev"]:
            return GlobalChWiseAttMeanStdPool1d(
                in_feats,
                inner_feats,
                bin_attn,
                use_global_context=use_global_context,
                norm_layer=norm_layer,
                dim=dim,
                keepdim=keepdim,
            )

    @staticmethod
    def filter_args(**kwargs):

        if "wo_bias" in kwargs:
            kwargs["use_bias"] = not kwargs["wo_bias"]
            del kwargs["wo_bias"]

        valid_args = (
            "pool_type",
            "dim",
            "keepdim",
            "in_feats",
            "num_comp",
            "use_bias",
            "dist_pow",
            "num_heads",
            "d_k",
            "d_v",
            "bin_attn",
            "inner_feats",
            "use_global_context",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=[]):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--pool-type",
            type=str.lower,
            default="mean+stddev",
            choices=[
                "avg",
                "mean+stddev",
                "mean+logvar",
                "lde",
                "scaled-dot-prod-att-v1",
                "ch-wise-att-mean+stddev",
            ],
            help=(
                "Pooling methods: Avg, Mean+Std, Mean+logVar, LDE, "
                "scaled-dot-product-attention-v1, Attentive-Mean+Std"
            ),
        )

        if "dim" not in skip:
            parser.add_argument(
                "--dim",
                default=-1,
                type=int,
                help=("Pooling dimension, usually time dimension"),
            )

        if "keepdim" not in skip:
            parser.add_argument(
                "--keepdim",
                default=False,
                action="store_true",
                help=("keeps the pooling dimension as singletone"),
            )

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats",
                default=0,
                type=int,
                help=("feature size for LDE/Att pooling"),
            )

        parser.add_argument(
            "--inner-feats",
            default=0,
            type=int,
            help=("inner feature size for attentive pooling"),
        )

        parser.add_argument(
            "--num-comp",
            default=8,
            type=int,
            help=("number of components for LDE pooling"),
        )

        parser.add_argument(
            "--dist-pow", default=2, type=int, help=("Distace power for LDE pooling")
        )

        parser.add_argument(
            "--wo-bias",
            default=False,
            action="store_true",
            help=("Don't use bias in LDE"),
        )

        parser.add_argument(
            "--num-heads", default=4, type=int, help=("number of attention heads")
        )

        parser.add_argument(
            "--d-k", default=256, type=int, help=("key dimension for attention")
        )

        parser.add_argument(
            "--d-v", default=256, type=int, help=("value dimension for attention")
        )

        parser.add_argument(
            "--bin-attn",
            default=False,
            action="store_true",
            help=("Use binary attention, i.e. sigmoid instead of softmax"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='pool options')

    @staticmethod
    def get_config(layer):

        config = layer.get_config()
        if isinstance(layer, GlobalAvgPool1d):
            config["pool_type"] = "avg"

        if isinstance(layer, GlobalMeanStdPool1d):
            config["pool_type"] = "mean+stddev"

        if isinstance(layer, GlobalMeanLogVarPool1d):
            config["pool_type"] = "mean+logvar"

        if isinstance(layer, LDEPool1d):
            config["pool_type"] = "lde"

        if isinstance(layer, ScaledDotProdAttV1Pool1d):
            config["pool_type"] = "scaled-dot-prod-att-v1"

        if isinstance(layer, GlobalChWiseAttMeanStdPool1d):
            config["pool_type"] = "ch-wise-att-mean+stddev"

        return config

    add_argparse_args = add_class_args
