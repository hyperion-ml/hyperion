"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn
from torch.nn import Linear

from ..layers import ActivationFactory as AF
from ..layers import CosLossOutput, ArcLossOutput, SubCenterArcLossOutput
from ..layers import NormLayer1dFactory as NLF
from ..layer_blocks import FCBlock
from .net_arch import NetArch


class ClassifHead(NetArch):
    """Classification Head for x-vector style networks

    Attributes:
       in_feats: input features
       num_classes: number of output classes
       embed_dim: dimension of embedding layer
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
    """

    def __init__(
        self,
        in_feats,
        num_classes,
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
        norm_layer=None,
        use_norm=True,
        norm_before=True,
        dropout_rate=0,
    ):

        super().__init__()
        assert num_embed_layers >= 1, "num_embed_layers (%d < 1)" % num_embed_layers

        self.num_embed_layers = num_embed_layers
        self.in_feats = in_feats
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        if use_norm:
            norm_groups = None
            if norm_layer == "group-norm":
                norm_groups = min(embed_dim // 8, 32)
            self._norm_layer = NLF.create(norm_layer, norm_groups)
        else:
            self._norm_layer = None

        self.use_norm = use_norm
        self.norm_before = norm_before

        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        self.num_subcenters = num_subcenters

        prev_feats = in_feats
        fc_blocks = []
        for i in range(num_embed_layers - 1):
            fc_blocks.append(
                FCBlock(
                    prev_feats,
                    embed_dim,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )
            prev_feats = embed_dim

        if loss_type != "softmax":
            act = None
        else:
            act = hid_act

        fc_blocks.append(
            FCBlock(
                prev_feats,
                embed_dim,
                activation=act,
                norm_layer=self._norm_layer,
                use_norm=use_norm,
                norm_before=norm_before,
            )
        )

        self.fc_blocks = nn.ModuleList(fc_blocks)

        # output layer
        if loss_type == "softmax":
            self.output = Linear(embed_dim, num_classes)
        elif loss_type == "cos-softmax":
            self.output = CosLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "arc-softmax":
            self.output = ArcLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "subcenter-arc-softmax":
            self.output = SubCenterArcLossOutput(
                embed_dim,
                num_classes,
                num_subcenters,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )

    def rebuild_output_layer(
        self,
        num_classes,
        loss_type,
        cos_scale,
        margin,
        margin_warmup_epochs,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
    ):

        embed_dim = self.embed_dim
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_margin = intertop_margin
        self.num_subcenters = num_subcenters
        self.num_subcenters = num_subcenters

        if loss_type == "softmax":
            self.output = Linear(embed_dim, num_classes)
        elif loss_type == "cos-softmax":
            self.output = CosLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "arc-softmax":
            self.output = ArcLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "subcenter-arc-softmax":
            self.output = SubCenterArcLossOutput(
                embed_dim,
                num_classes,
                num_subcenters,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )

    def set_margin(self, margin):
        if self.loss_type == "softmax":
            return

        self.margin = margin
        self.output.margin = margin

    def set_margin_warmup_epochs(self, margin_warmup_epochs):
        if self.loss_type == "softmax":
            return

        self.margin_warmup_epochs = margin_warmup_epochs
        self.output.margin_warmup_epochs = margin_warmup_epochs

    def set_cos_scale(self, cos_scale):
        if self.loss_type == "softmax":
            return

        self.cos_scale = cos_scale
        self.output.cos_scale = cos_scale

    def set_intertop_k(self, intertop_k):
        if self.loss_type == "softmax":
            return

        self.intertop_k = intertop_k
        self.output.intertop_k = intertop_k

    def set_intertop_margin(self, intertop_margin):
        if self.loss_type == "softmax":
            return

        self.intertop_margin = intertop_margin
        self.output.intertop_margin = intertop_margin

    def set_num_subcenters(self, num_subcenters):
        if not self.loss_type == "subcenter-arc-softmax":
            return

        self.num_subcenters = num_subcenters
        self.output.num_subcenters = num_subcenters

    def update_margin(self, epoch):
        if hasattr(self.output, "update_margin"):
            self.output.update_margin(epoch)

    def freeze_layers(self, layer_list):
        for l in layer_list:
            for param in self.fc_blocks[l].parameters():
                param.requires_grad = False

    def put_layers_in_eval_mode(self, layer_list):
        for l in layer_list:
            self.fc_blocks[l].eval()

    def forward(self, x, y=None):

        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)

        if self.loss_type == "softmax" or isinstance(self.output,nn.modules.linear.Identity):
            y = self.output(x)
        else:
            y = self.output(x, y)

        return y

    def forward_hid_feats(self, x, y=None, return_layers=None, return_logits=False):

        assert return_layers is not None or return_logits
        if return_layers is None:
            return_layers = []

        h = []
        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)
            if l in return_layers:
                h.append(x)

        if self.loss_type == "softmax":
            y = self.output(x)
        else:
            y = self.output(x, y)

        if return_logits:
            return h, y
        return h, None

    def extract_embed(self, x, embed_layer=0):

        for l in range(embed_layer):
            x = self.fc_blocks[l](x)

        if self.loss_type == "softmax" or embed_layer < self.num_embed_layers:
            y = self.fc_blocks[embed_layer].forward_linear(x)
        else:
            y = self.fc_blocks[l](x)
        return y

    def compute_prototype_affinity(self):
        if self.loss_type != "softmax":
            return self.output.compute_prototype_affinity()

        kernel = self.output.weight  # (num_classes, feat_dim)
        kernel = kernel / torch.linalg.norm(kernel, 2, dim=1, keepdim=True)
        return torch.mm(kernel, kernel.transpose(0, 1))

    def get_config(self):

        hid_act = AF.get_config(self.fc_blocks[0].activation)

        config = {
            "in_feats": self.in_feats,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_embed_layers": self.num_embed_layers,
            "hid_act": hid_act,
            "lost_type": self.lost_type,
            "cos_scale": self.cos_scale,
            "margin": self.margin,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "intertop_k": self.intertop_k,
            "intertop_margin": self.intertop_margin,
            "num_subcenters": self.num_subcenters,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "dropout_rate": self.dropout_rate,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):

        if "wo_norm" in kwargs:
            kwargs["use_norm"] = not kwargs["wo_norm"]
            del kwargs["wo_norm"]

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        valid_args = (
            "num_classes",
            "embed_dim",
            "num_embed_layers",
            "hid_act",
            "loss_type",
            "s",
            "margin",
            "margin_warmup_epochs",
            "intertop_k",
            "intertop_margin",
            "num_subcenters",
            "use_norm",
            "norm_before",
            "dropout_rate",
            "norm_layer",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--embed-dim", default=256, type=int, help=("x-vector dimension")
        )

        parser.add_argument(
            "--num-embed-layers",
            default=1,
            type=int,
            help=("number of layers in the classif head"),
        )

        try:
            parser.add_argument("--hid-act", default="relu6", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--loss-type",
            default="arc-softmax",
            choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
            help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
        )

        parser.add_argument("--s", default=64, type=float, help="scale for arcface")

        parser.add_argument(
            "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
        )

        parser.add_argument(
            "--margin-warmup-epochs",
            default=10,
            type=float,
            help="number of epoch until we set the final margin",
        )

        parser.add_argument(
            "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
        )
        parser.add_argument(
            "--intertop-margin",
            default=0.0,
            type=float,
            help="margin for InterTopK penalty",
        )

        parser.add_argument(
            "--num-subcenters",
            default=2,
            type=int,
            help="number of subcenters in subcenter losses",
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
            "--wo-norm",
            default=False,
            action="store_true",
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-after",
            default=False,
            action="store_true",
            help="batch normalizaton after activation",
        )

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='classification head options')

    add_argparse_args = add_class_args
