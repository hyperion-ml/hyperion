"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..layers import FeatFuserFactory as FFF
from ..layers import MeanVarianceNorm as MVN
from ..layers import SpecAugment
from .net_arch import NetArch


class FeatFuserMVN(NetArch):
    """FeatureFuser for Wav2Vec style hidden features + ST-MVN
    Optional SpecAugment
    """

    def __init__(
        self,
        feat_fuser: Dict[str, Any],
        mvn: Optional[Dict[str, Any]] = None,
        spec_augment: Optional[Dict[str, Any]] = None,
        trans: bool = False,
        aug_after_mvn: bool = False,
    ):
        super().__init__()

        feat_fuser = FFF.filter_args(**feat_fuser)
        self.feat_fuser_cfg = feat_fuser
        self.feat_fuser = FFF.create(**feat_fuser)

        self.mvn = None
        self.mvn_cfg = None
        if mvn is not None:
            mvn = MVN.filter_args(**mvn)
            self.mvn_cfg = mvn
            if (
                ("norm_mean" in mvn)
                and mvn["norm_mean"]
                or ("norm_var" in mvn)
                and mvn["norm_var"]
            ):
                self.mvn = MVN(**mvn)

        self.spec_augment = None
        self.spec_augment_cfg = None
        if spec_augment is not None:
            spec_augment = SpecAugment.filter_args(**spec_augment)
            self.spec_augment_cfg = spec_augment
            self.spec_augment = SpecAugment(**spec_augment)

        self.trans = trans
        self.aug_after_mvn = aug_after_mvn

    @property
    def fuser_type(self):
        return self.feat_fuser_cfg["fuser_type"]

    def forward(self, feats, feats_lengths=None):
        feats = self.feat_fuser(feats)
        if self.spec_augment is not None and not self.aug_after_mvn:
            feats = self.spec_augment(feats, feats_lengths)

        if self.mvn is not None:
            feats = self.mvn(feats, feats_lengths)

        if self.spec_augment is not None and self.aug_after_mvn:
            feats = self.spec_augment(feats, feats_lengths)

        if self.trans:
            feats = feats.transpose(1, 2).contiguous()

        return feats, feats_lengths

    def get_config(self):
        config = {
            "feat_fuser": self.feat_fuser_cfg,
            "mvn": self.mvn_cfg,
            "spec_augment": self.spec_augment_cfg,
            "trans": self.trans,
            "aug_after_mvn": self.aug_after_mvn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("feat_fuser", "mvn", "spec_augment", "trans", "aug_after_mvn")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        FFF.add_class_args(parser, prefix="feat_fuser")
        MVN.add_class_args(parser, prefix="mvn")
        SpecAugment.add_class_args(parser, prefix="spec_augment")
        parser.add_argument(
            "--aug-after-mvn",
            default=False,
            action="store_true",
            help=("do spec augment after st-mvn," "instead of before"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
