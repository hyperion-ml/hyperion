"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layers import ActivationFactory as AF
from ..layers import CosLossOutput
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch


class DINOHead(NetArch):
    """Classification head for DINO x-vector style networks.

    Attributes:
        in_feats: Input feature dimension.
        num_classes: Number of output classes.
        hid_feats: Hidden-layer dimension.
        bottleneck_feats: Bottleneck dimension before the output layer.
        num_hid_layers: Number of hidden layers.
        hid_act: Hidden activation type or configuration.
        output_type: Output layer type, either ``softmax`` or
            ``cos-softmax``.
        norm_layer: Normalization-layer type. If ``None``, BatchNorm1d is
            used.
        use_norm: Whether to apply normalization layers in the hidden stack.
        norm_before: If ``True``, normalization is applied before activation.
        dropout_rate: Dropout rate applied between hidden layers.
        use_in_norm: If ``True``, apply normalization at the input.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        hid_feats: int = 2048,
        bottleneck_feats: int = 256,
        num_hid_layers: int = 3,
        hid_act: Any = "gelu",
        output_type: str = "softmax",
        norm_layer: Optional[str] = None,
        use_norm: bool = False,
        norm_before: bool = True,
        dropout_rate: float = 0,
        use_in_norm: bool = False,
    ) -> None:
        """Build the DINO classification head.

        Args:
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            hid_feats: Hidden-layer dimension.
            bottleneck_feats: Bottleneck dimension before the output layer.
            num_hid_layers: Number of hidden layers.
            hid_act: Hidden activation type or configuration.
            output_type: Output layer type, either ``softmax`` or
                ``cos-softmax``.
            norm_layer: Normalization-layer type.
            use_norm: Whether to apply normalization layers in the hidden
                stack.
            norm_before: If ``True``, normalization is applied before
                activation.
            dropout_rate: Dropout rate applied between hidden layers.
            use_in_norm: If ``True``, apply normalization at the input.
        """
        super().__init__()
        assert num_hid_layers >= 1, "num_hid_layers (%d < 1)" % num_hid_layers

        self.num_hid_layers = num_hid_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.bottleneck_feats = bottleneck_feats
        self.num_classes = num_classes
        self.hid_act = hid_act
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
            assert self._norm_layer is not None, "use_in_norm requires use_norm"
            self.in_norm = self._norm_layer(in_feats)

        layers = []
        if num_hid_layers == 1:
            layers.append(nn.Linear(in_feats, bottleneck_feats))
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
                bottleneck_feats,
                num_classes,
                cos_scale=1,
                margin=0,
                margin_warmup_epochs=0,
                intertop_k=0,
                intertop_margin=0,
            )
        else:
            raise ValueError(f"wrong output_type={output_type}")

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize linear-layer weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute output logits for an input batch.

        Args:
            x: Input tensor with shape ``(batch, in_feats)``.
            y: Optional target labels required by cosine-based outputs during
                training.

        Returns:
            Output tensor with shape ``(batch, num_classes)``.
        """
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
        if self.output_type == "softmax":
            x = self.output(x)
        else:
            x = self.output(x, y)
        # assert not torch.any(
        #     torch.isnan(x)
        # ), f"out is nan  {x.size()} {torch.sum(torch.isnan(x))}"
        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration for the head.

        Args:
            no_class_name: If ``True``, omit the class name from the config.

        Returns:
            Dictionary containing the constructor arguments.
        """
        hid_act = self.hid_act

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

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter constructor keyword arguments to the supported subset.

        Args:
            **kwargs: Candidate constructor keyword arguments.

        Returns:
            Dictionary with unsupported keys removed and aliases normalized.
        """
        # if "wo_norm" in kwargs:
        #     kwargs["use_norm"] = not kwargs["wo_norm"]
        #     del kwargs["wo_norm"]

        # if "norm_after" in kwargs:
        #     kwargs["norm_before"] = not kwargs["norm_after"]
        #     del kwargs["norm_after"]

        if "botteneck_feats" in kwargs:
            kwargs["bottleneck_feats"] = kwargs["botteneck_feats"]
            del kwargs["botteneck_feats"]

        if "output_layer" in kwargs:
            kwargs["output_type"] = kwargs["output_layer"]
            del kwargs["output_layer"]

        return filter_func_args(DINOHead.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register CLI arguments for configuring the head.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used to nest the arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--bottleneck-feats",
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
            "--output-type",
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
