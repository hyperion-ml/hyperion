"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Any

import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from .global_pool import (
    GlobalAvgPool1d,
    GlobalChWiseAttMeanStdPool1d,
    GlobalMeanLogVarPool1d,
    GlobalMeanStdPool1d,
    LDEPool1d,
    ScaledDotProdAttV1Pool1d,
)


class PoolType(str, Enum):
    """Supported global pooling types."""

    AVG = "avg"
    MEAN_STDDEV = "mean+stddev"
    MEAN_LOGVAR = "mean+logvar"
    LDE = "lde"
    SCALED_DOT_PROD_ATT_V1 = "scaled-dot-prod-att-v1"
    CH_WISE_ATT_MEAN_STDDEV = "ch-wise-att-mean+stddev"
    CH_WISE_ATT_MEAN_STDDEV_ALT = "ch-wise-att-mean-stddev"

    @staticmethod
    def choices() -> list[str]:
        """Returns the list of pooling type choices."""
        return [e.value for e in PoolType]


class GlobalPool1dFactory(object):
    """Factory class to create 1-D global pooling modules.

    Examples:
        >>> pool = GlobalPool1dFactory.create(PoolType.MEAN_STDDEV)
        >>> isinstance(pool, GlobalMeanStdPool1d)
        True

        >>> attn_pool = GlobalPool1dFactory.create(
        ...     PoolType.SCALED_DOT_PROD_ATT_V1,
        ...     in_feats=256,
        ...     num_heads=4,
        ...     d_k=64,
        ...     d_v=64,
        ... )
        >>> isinstance(attn_pool, ScaledDotProdAttV1Pool1d)
        True
    """

    @staticmethod
    def create(
        pool_type: PoolType,
        in_feats: int | None = None,
        inner_feats: int = 128,
        num_comp: int = 64,
        dist_pow: int = 2,
        use_bias: bool = False,
        num_heads: int = 8,
        d_k: int = 256,
        d_v: int = 256,
        bin_attn: bool = False,
        use_global_context: bool = True,
        norm_layer: type[nn.Module] | None = None,
        dim: int = -1,
        keepdim: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Creates a global pooling layer from arguments.

        Args:
          pool_type: Pooling type.
          in_feats: Input feature dimension for pooling layers that require it.
          inner_feats: Hidden feature dimension for channel-wise attentive pooling.
          num_comp: Number of LDE components.
          dist_pow: Distance power in LDE pooling.
          use_bias: If ``True``, uses bias in LDE pooling.
          num_heads: Number of attention heads.
          d_k: Key dimension in scaled dot-product attention pooling.
          d_v: Value dimension in scaled dot-product attention pooling.
          bin_attn: If ``True``, uses sigmoid (binary) attention.
          use_global_context: If ``True``, uses global context in channel-wise
            attentive pooling.
          norm_layer: Normalization layer constructor for channel-wise attentive
            pooling.
          dim: Pooling dimension.
          keepdim: If ``True``, keeps the pooling dimension.
          **kwargs: Unused extra keyword arguments kept for compatibility.

        Returns:
          Instantiated pooling module.
        """
        del kwargs

        if pool_type == PoolType.AVG:
            return GlobalAvgPool1d(dim=dim, keepdim=keepdim)

        if pool_type == PoolType.MEAN_STDDEV:
            return GlobalMeanStdPool1d(dim=dim, keepdim=keepdim)

        if pool_type == PoolType.MEAN_LOGVAR:
            return GlobalMeanLogVarPool1d(dim=dim, keepdim=keepdim)

        if pool_type == PoolType.LDE:
            return LDEPool1d(
                in_feats,
                num_comp=num_comp,
                dist_pow=dist_pow,
                use_bias=use_bias,
                dim=dim,
                keepdim=keepdim,
            )

        if pool_type == PoolType.SCALED_DOT_PROD_ATT_V1:
            return ScaledDotProdAttV1Pool1d(
                in_feats,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                bin_attn=bin_attn,
                dim=dim,
                keepdim=keepdim,
            )

        if pool_type in (
            PoolType.CH_WISE_ATT_MEAN_STDDEV,
            PoolType.CH_WISE_ATT_MEAN_STDDEV_ALT,
        ):
            return GlobalChWiseAttMeanStdPool1d(
                in_feats,
                inner_feats,
                bin_attn,
                use_global_context=use_global_context,
                norm_layer=norm_layer,
                dim=dim,
                keepdim=keepdim,
            )

        raise ValueError(f"Invalid pooling type {pool_type}")

    @staticmethod
    def filter_args(**kwargs: Any) -> dict[str, Any]:
        """Filters the arguments corresponding to the creation of a pooling layer.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with the pooling layer options.
        """

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

        return {k: kwargs[k] for k in valid_args if k in kwargs}

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: str | None = None,
        skip: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Adds pooling-layer command-line options to an argument parser.

        Args:
          parser: Target parser.
          prefix: Optional nested prefix name.
          skip: Optional argument names to skip.

        Returns:
          ``None``.
        """
        if skip is None:
            skip = []

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
                "Pooling methods: avg, mean+stddev, mean+logvar, lde, "
                "scaled-dot-prod-att-v1, ch-wise-att-mean+stddev"
            ),
        )

        if "dim" not in skip:
            parser.add_argument(
                "--dim",
                default=-1,
                type=int,
                help=("Pooling dimension (usually the time dimension)"),
            )

        if "keepdim" not in skip:
            parser.add_argument(
                "--keepdim",
                default=False,
                action="store_true",
                help=("Keeps the pooled dimension as a singleton axis"),
            )

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats",
                default=0,
                type=int,
                help=("Input feature size for LDE/attention pooling"),
            )

        parser.add_argument(
            "--inner-feats",
            default=0,
            type=int,
            help=("Hidden feature size for channel-wise attentive pooling"),
        )

        parser.add_argument(
            "--num-comp",
            default=8,
            type=int,
            help=("Number of components for LDE pooling"),
        )

        parser.add_argument(
            "--dist-pow",
            default=2,
            type=int,
            help=("Distance power for LDE pooling (typically 1 or 2)"),
        )

        parser.add_argument(
            "--wo-bias",
            default=False,
            action="store_true",
            help=("Disables bias in LDE pooling"),
        )

        parser.add_argument(
            "--num-heads", default=4, type=int, help=("Number of attention heads")
        )

        parser.add_argument(
            "--d-k", default=256, type=int, help=("Key dimension for attention")
        )

        parser.add_argument(
            "--d-v", default=256, type=int, help=("Value dimension for attention")
        )

        parser.add_argument(
            "--bin-attn",
            default=False,
            action="store_true",
            help=("Uses binary attention (sigmoid instead of softmax)"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def get_config(layer: nn.Module) -> dict[str, Any]:
        """Returns the serialized config for a pooling module.

        Args:
          layer: Pooling layer instance.

        Returns:
          Layer config with ``pool_type`` included.
        """

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
