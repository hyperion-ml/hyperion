"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Optional

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from .feat_fusers import (
    CatFeatFuser,
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
    """Factory class to create feature fusers for Wav2Vec style hidden features."""

    @staticmethod
    def create(
        fuser_type: str = WAVG_FUSER,
        num_feats: Optional[int] = None,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ):
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
        """Filters arguments correspondin to Feature Fuser
            from args dictionary

        Args:
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        args = filter_func_args(FeatFuserFactory.create, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds feature extractor options to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
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
