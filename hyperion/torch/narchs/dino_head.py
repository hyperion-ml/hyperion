"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import FCBlock
from ..layers import ActivationFactory as AF
from ..layers import CosLossOutput
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch


class DINOHead(NetArch):
    """Classification Head for DINO x-vector style networks

    Attributes:
       in_feats: input features
       num_classes: number of output classes
       hid_feats: dimension of hidding layer
       bottleneck_feats: dimension of bottleneck layer before output
       num_hid_layers: number of hidden layers
       hid_act: str or dict hidden activation type in ['relu', 'relu6', 'swish', ... ]
       output_type: type of output layer that will be used with the x-vector in ['softmax', 'cos-softmax'],
                  corresponding to standard cross-entorpy, cosine scoring
       norm_layer: norm_layer object or str indicating type norm layer, if None it uses BatchNorm1d
       use_norm: it True it uses layer/batch-normalization
       norm_before: if True, layer-norm is before the activation function
       use_in_norm: put batchnorm at the input
    """

    def __init__(
        self,
        in_feats,
        num_classes,
        hid_feats=2048,
        bottleneck_feats=256,
        num_hid_layers=3,
        hid_act="gelu",
        output_type="softmax",
        norm_layer=None,
        use_norm=False,
        norm_before=True,
        dropout_rate=0,
        use_in_norm=False,
    ):
        super().__init__()
        assert num_hid_layers >= 1, "num_layers (%d < 1)" % num_hid_layers

        self.num_hid_ayers = num_hid_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.bottleneck_feats = bottleneck_feats
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.use_in_norm = use_in_norm

        if use_norm:
            norm_groups = None
            if norm_layer == "group-norm":
                norm_groups = min(hid_feats // 8, 32)
            self._norm_layer = NLF.create(norm_layer, norm_groups)
        else:
            self._norm_layer = None

        self.use_norm = use_norm
        self.norm_before = norm_before

        self.dropout_rate = dropout_rate
        self.output_type = output_type
        if use_in_norm:
            assert not self.norm_before
            self.in_norm = self._norm_layer(in_feats)

        if num_hid_layers == 1:
            self.fc_layers = nn.Linear(in_feats, bottleneck_feats)
        else:
            use_bias = False if use_norm and norm_before else True
            layers = [nn.Linear(in_feats, hid_feats, bias=use_bias)]
            if use_norm and norm_before:
                layers.append(self._norm_layer(hid_feats))
            layers.append(AF.create(hid_act))
            if use_norm and not norm_before:
                layers.append(self._norm_layer(hid_feats))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            for _ in range(num_hid_layers - 2):
                layers.append(nn.Linear(hid_feats, hid_feats, bias=use_bias))
                if use_norm and norm_before:
                    layers.append(self._norm_layer(hid_feats))
                layers.append(AF.create(hid_act))
                if use_norm and not norm_before:
                    layers.append(self._norm_layer(hid_feats))
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))

            layers.append(nn.Linear(hid_feats, bottleneck_feats))
            self.hid_layers = nn.Sequential(*layers)

        self.apply(self._init_weights)
        if output_type == "softmax":
            output = nn.Linear(bottleneck_feats, num_classes, bias=False)
            with torch.no_grad():
                self.output = nn.utils.weight_norm(output)
            self.output.weight_g.data.fill_(1)
            self.output.weight_g.requires_grad = False
        elif output_type == "cos-softmax":
            self.output = CosLossOutput(
                hid_feats,
                num_classes,
                cos_scale=1,
                margin=0,
                margin_warmup_epochs=0,
                intertop_k=0,
                intertop_margin=0,
            )
        else:
            raise ValueError(f"wrong loss_type={output_type}")

    # def before_cloning(self):
    #     if self.output_type == "cos-softmax":
    #         return None, None

    #     torch.nn.utils.remove_weight_norm(self.output)
    #     return None, None
    #     cloned_output = self._clone_output()
    #     output = self.output
    #     self.output = None
    #     return output, cloned_output

    # def after_cloning(self, output: nn.Module):
    #     if self.output_type == "cos-softmax":
    #         return

    #     self.output = nn.utils.weight_norm(self.output)
    #     self.output.weight_g.data.fill_(1)
    #     self.output.weight_g.requires_grad = False

    # def _clone_output(self):
    #     output = nn.utils.weight_norm(
    #         nn.Linear(self.bottleneck_feats, self.num_classes, bias=False)
    #     )
    #     output.weight_g.data.fill_(1)
    #     output.weight_v.data = self.output_v.data.detach()
    #     output.weight_g.requires_grad = False
    #     return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.use_in_norm:
            x = self.in_norm(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x is nan {x.size()} {torch.sum(torch.isnan(x))}"
        x = self.hid_layers(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x_hid is nan {x.size()} {torch.sum(torch.isnan(x))}"
        x = nn.functional.normalize(x, dim=-1, p=2)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"x_l2 is nan  {x.size()} {torch.sum(torch.isnan(x))}"
        x = self.output(x)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"out is nan  {x.size()} {torch.sum(torch.isnan(x))}"
        return x

    def get_config(self):
        hid_act = AF.get_config(self.fc_blocks[0].activation)

        config = {
            "in_feats": self.in_feats,
            "num_classes": self.num_classes,
            "hid_feats": self.hid_feats,
            "bottleneck_feats": self.bottleneck_feats,
            "num_hid_layers": self.num_hid_layers,
            "hid_act": hid_act,
            "output_type": self.output_type,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "dropout_rate": self.dropout_rate,
            "use_in_norm": self.use_in_norm,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        # if "wo_norm" in kwargs:
        #     kwargs["use_norm"] = not kwargs["wo_norm"]
        #     del kwargs["wo_norm"]

        # if "norm_after" in kwargs:
        #     kwargs["norm_before"] = not kwargs["norm_after"]
        #     del kwargs["norm_after"]

        args = filter_func_args(DINOHead.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--botteneck-feats",
            default=256,
            type=int,
            help=("bottleneck dimension before output layer"),
        )

        parser.add_argument(
            "--num-hid-layers",
            default=3,
            type=int,
            help=("number of hidden layers in the classif head"),
        )

        try:
            parser.add_argument("--hid-act", default="gelu", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--output-layer",
            default="softmax",
            choices=["softmax", "cos-softmax"],
            help="loss type: softmax, cos-softmax",
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
            default=True,
            action=ActionYesNo,
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton before activation",
        )

        parser.add_argument(
            "--use-in-norm",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton in the classif head input",
        )

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
