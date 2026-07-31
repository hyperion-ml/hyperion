"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

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
    """Projection head for x-vector style networks.

    Attributes:
       in_feats: Input feature dimension.
       out_feats: Output projection dimension.
       norm_layer: Normalization-layer specification passed to
           ``NormLayer1dFactory.create``.
       use_norm: Whether to apply normalization.
       norm_before: Whether normalization is applied before the projection.
       proj: Linear projection layer.
       _norm_layer: Normalization module when ``use_norm`` is enabled.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int = 256,
        norm_layer: Optional[Any] = None,
        use_norm: bool = True,
        norm_before: bool = True,
    ) -> None:
        """Initialize the projection head.

        Args:
            in_feats: Input feature dimension.
            out_feats: Output projection dimension.
            norm_layer: Normalization-layer specification passed to
                ``NormLayer1dFactory.create``.
            use_norm: Whether to apply normalization.
            norm_before: Whether normalization is applied before the projection.
        """
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        use_bias = True
        if use_norm:
            norm_groups = None
            if norm_layer == "group-norm":
                norm_feats = in_feats if norm_before else out_feats
                norm_groups = max(1, min(norm_feats // 8, 32))
            _norm_layer = NLF.create(norm_layer, norm_groups)
            if norm_before:
                self._norm_layer = _norm_layer(in_feats)
            else:
                self._norm_layer = _norm_layer(out_feats)
                use_bias = False
        else:
            self._norm_layer = None

        self.proj = nn.Linear(in_feats, out_feats, bias=use_bias)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the projection head.

        Args:
            x: Input tensor.
            y: Unused optional target tensor, accepted for interface compatibility.

        Returns:
            Projected tensor.
        """
        if self.use_norm and self.norm_before:
            x = self._norm_layer(x)

        x = self.proj(x)

        if self.use_norm and not self.norm_before:
            x = self._norm_layer(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the base class name from the config.

        Returns:
            A dictionary with the constructor arguments.
        """
        config = {
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments to the subset accepted by ``__init__``.

        Returns:
            Keyword arguments compatible with ``ProjHead.__init__``.
        """
        args = filter_func_args(ProjHead.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register CLI arguments for building a projection head.

        Args:
            parser: Parser to extend.
            prefix: Optional nested prefix used to group arguments.
        """
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
                help="type of normalization layer",
            )
        except:
            pass

        parser.add_argument(
            "--use-norm",
            default=True,
            action=ActionYesNo,
            help="apply normalization before or after the projection",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="apply normalization before the projection",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
