"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Set, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks.transformer_v2 import TransformerV2NormLayerType
from .net_arch import NetArch


class QProjHead(NetArch):
    """Linear projection used to collapse a flattened q-matrix into a q-vector.

    Attributes:
        in_feats: Input feature dimensionality.
        out_feats: Output q-vector dimensionality.
        norm_layer: Normalization layer type applied before the projection.
        use_norm: Flag indicating whether the normalization layer is active.
        bias: Whether the projection layer uses a bias term.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        norm_layer: Union[str, TransformerV2NormLayerType] = (
            TransformerV2NormLayerType.LAYERNORM
        ),
        use_norm: bool = False,
        bias: bool = True,
    ) -> None:
        """Initialize the projection block.

        Args:
            in_feats: Size of the flattened input feature vector.
            out_feats: Target q-vector dimensionality.
            norm_layer: Normalization layer type applied before the linear
                projection when ``use_norm=True``.
            use_norm: Enable/disable the normalization layer.
            bias: Whether the linear projection uses a bias term.
        """
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        if isinstance(norm_layer, str):
            norm_layer = TransformerV2NormLayerType(norm_layer)
        self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.bias = bias

        self._norm_layer: Union[nn.Module, None] = None
        if use_norm:
            norm_cls = TransformerV2NormLayerType.to_class(self.norm_layer)
            self._norm_layer = norm_cls(in_feats)

        self.proj = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the flattened q-matrix into a q-vector embedding.

        Args:
            x: Input tensor of shape ``(..., in_feats)``.

        Returns:
            torch.Tensor: Projected tensor of shape ``(..., out_feats)``.
        """
        if self._norm_layer is not None:
            x = self._norm_layer(x)

        x = self.proj(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the constructor arguments."""
        config = {
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "norm_layer": self.norm_layer.value,
            "use_norm": self.use_norm,
            "bias": self.bias,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments so only constructor parameters remain."""
        return filter_func_args(QProjHead.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI/configuration arguments for ``QProjHead``.

        Args:
            parser: Destination parser.
            prefix: Optional nested namespace.
            skip: Optional set of argument names to omit.
        """
        skip = set(skip) if skip else set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        if "out_feats" not in skip:
            parser.add_argument(
                "--out-feats",
                default=256,
                type=int,
                help="dimensionality of the projected q-vector",
            )

        if "norm_layer" not in skip:
            parser.add_argument(
                "--norm-layer",
                default=TransformerV2NormLayerType.LAYERNORM.value,
                choices=TransformerV2NormLayerType.choices(),
                help="type of normalization layer applied before the projection",
            )

        if "use_norm" not in skip:
            parser.add_argument(
                "--use-norm",
                default=False,
                action=ActionYesNo,
                help="enable the selected normalization layer before projecting",
            )

        if "bias" not in skip:
            parser.add_argument(
                "--bias",
                default=True,
                action=ActionYesNo,
                help="include a bias term in the projection layer",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
