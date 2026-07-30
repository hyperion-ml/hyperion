"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from .feat_fusers import (
    CatFeatFuser,
    FeatFuser,
    LastFeatFuser,
    LinearFeatFuser,
    WeightedAvgFeatFuser,
)

LAST_FUSER = "last"
WAVG_FUSER = "weighted-avg"
LINEAR_FUSER = "linear"
CAT_FUSER = "cat"

FUSER_TYPES = [LAST_FUSER, WAVG_FUSER, LINEAR_FUSER, CAT_FUSER]


class FeatFuserFactory:
    """Factory for creating feature fusers from configuration values.

    Examples:
      >>> fuser = FeatFuserFactory.create(
      ...     fuser_type="weighted-avg", num_feats=3, feat_dim=256, proj_dim=128
      ... )
      >>> fuser = FeatFuserFactory.create(
      ...     fuser_type="last", feat_dim=256, proj_dim=128, proj_bias=False
      ... )
      >>> fuser = FeatFuserFactory.create(
      ...     fuser_type="cat", num_feats=4, feat_dim=64, proj_dim=128
      ... )

    Attributes:
      None.
    """

    @staticmethod
    def create(
        fuser_type: str = WAVG_FUSER,
        num_feats: Optional[int] = None,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> FeatFuser:
        """Builds a feature fuser module from the requested type.

        Args:
          fuser_type: Fuser type string, one of ``FUSER_TYPES``.
          num_feats: Number of feature tensors to fuse for multi-input fusers.
          feat_dim: Input feature dimension per tensor.
          proj_dim: Optional output projection dimension.
          proj_bias: Whether projection layers include a bias term.

        Returns:
          Instantiated feature fuser module.
        """
        if fuser_type in (WAVG_FUSER, LINEAR_FUSER, CAT_FUSER) and num_feats is None:
            raise ValueError(f"num_feats is required for fuser_type={fuser_type}")

        if fuser_type == CAT_FUSER and feat_dim is None:
            raise ValueError("feat_dim is required for fuser_type=cat")

        if fuser_type == WAVG_FUSER:
            return WeightedAvgFeatFuser(
                num_feats, feat_dim=feat_dim, proj_dim=proj_dim, proj_bias=proj_bias
            )
        elif fuser_type == LAST_FUSER:
            return LastFeatFuser(
                feat_dim=feat_dim, proj_dim=proj_dim, proj_bias=proj_bias
            )
        elif fuser_type == LINEAR_FUSER:
            return LinearFeatFuser(
                num_feats, feat_dim=feat_dim, proj_dim=proj_dim, proj_bias=proj_bias
            )
        elif fuser_type == CAT_FUSER:
            return CatFeatFuser(
                num_feats, feat_dim=feat_dim, proj_dim=proj_dim, proj_bias=proj_bias
            )
        else:
            raise ValueError(f"unknown feature fuser type {fuser_type}")

    @staticmethod
    def filter_args(**kwargs):
        """Filters keyword arguments accepted by :meth:`create`.

        Args:
          kwargs: Candidate keyword arguments.

        Returns:
          Dictionary containing only keys supported by :meth:`create`.
        """
        args = filter_func_args(FeatFuserFactory.create, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds feature fuser options to an argument parser.

        Args:
          parser: Argument parser object to extend.
          prefix: Options prefix.

        Returns:
          None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--fuser-type",
            default=WAVG_FUSER,
            choices=FUSER_TYPES,
            help=f"One of {FUSER_TYPES}",
        )
        parser.add_argument(
            "--proj-dim",
            default=None,
            type=int,
            help="project features after fusion to proj_dim",
        )
        parser.add_argument(
            "--proj-bias",
            default=True,
            action=ActionYesNo,
            help="linear projection has bias",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
