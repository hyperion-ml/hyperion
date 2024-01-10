"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn import Linear

from ...utils.misc import filter_func_args
from ..layer_blocks import FCBlock
from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch


class ProjHead(NetArch):
    """Classification Head for x-vector style networks

    Attributes:
       in_feats: input features
       num_classes: number of output classes
       out_feats: dimension of embedding layer
       num_embed_layers: number of hidden layers
       hid_act: str or dict hidden activation type in ['relu', 'relu6', 'swish', ... ]
       loss_type: type of loss function that will be used with the x-vector in ['softmax', 'cos-softmax', 'arc-softmax'],
                  corresponding to standard cross-entorpy, additive margin softmax or additive angular margin softmax.
       cos_scale: scale parameter for cos-softmax and arc-softmax
       margin: margin parameter for cos-softmax and arc-softmax
       margin_warmup_epochs: number of epochs to anneal the margin from 0 to margin
       intertop_k: adds negative angular penalty to k largest negative scores.
       intertop_margin: inter-top-k penalty.
       num_subcenters: number of subcenters in subcenter losses
       norm_layer: norm_layer object or str indicating type norm layer, if None it uses BatchNorm1d
       use_norm: it True it uses layer/batch-normalization
       norm_before: if True, layer-norm is before the activation function
       use_in_norm: put batchnorm at the input
    """

    def __init__(
        self, in_feats, out_feats=256, norm_layer=None, use_norm=True, norm_before=True,
    ):
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before

        if use_norm:
            norm_groups = None
            if norm_layer == "group-norm":
                norm_groups = min(out_feats // 8, 32)
            _norm_layer = NLF.create(norm_layer, norm_groups)
            if norm_before:
                self._norm_layer = _norm_layer(in_feats)
            else:
                self._norm_layer = _norm_layer(out_feats)
        else:
            self._norm_layer = None

        self.proj = nn.Linear(in_feats, out_feats)

    def forward(self, x, y=None):
        if self.use_norm and self.norm_before:
            x = self._norm_layer(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x before proj is nan {x.size()} {torch.sum(torch.isnan(x))}"
        x = self.proj(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x after proj is nan {x.size()} {torch.sum(torch.isnan(x))}"
        if self.use_norm and not self.norm_before:
            x = self._norm_layer(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x after bn is nan {x.size()} {torch.sum(torch.isnan(x))}"
        return x

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(ProjHead.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--out-feats", default=256, type=int, help=("projection dimension")
        )

        try:
            parser.add_argument(
                "--norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help="type of normalization layer for all components of x-vector network",
            )
        except:
            pass

        parser.add_argument(
            "--use-norm",
            default=False,
            action=ActionYesNo,
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton after activation",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
