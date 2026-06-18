"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..layers import FeatFuserFactory as FFF
from ..layers import MeanVarianceNorm as MVN
from ..layers import SpecAugment
from .net_arch import NetArch


class FeatFuserMVN(NetArch):
    """Feature fuser frontend with optional short-term MVN and SpecAugment.

    The module applies:
    1. Feature fusion from a sequence of hidden representations.
    2. Optional SpecAugment before or after MVN.
    3. Optional short-term mean/variance normalization.
    4. Optional transpose to channel-first layout.

    Attributes:
        feat_fuser_cfg: Filtered constructor config used to build ``feat_fuser``.
        feat_fuser: Feature fusion module created by ``FeatFuserFactory``.
        mvn_cfg: Optional filtered constructor config used to build ``mvn``.
        mvn: Optional short-term mean/variance normalization layer.
        spec_augment_cfg: Optional filtered constructor config used to build ``spec_augment``.
        spec_augment: Optional SpecAugment module.
        trans: If ``True``, output layout is ``(B, C, T)``; otherwise ``(B, T, C)``.
        aug_after_mvn: If ``True``, applies SpecAugment after MVN; otherwise before MVN.
    """

    def __init__(
        self,
        feat_fuser: Dict[str, Any],
        mvn: Optional[Dict[str, Any]] = None,
        spec_augment: Optional[Dict[str, Any]] = None,
        trans: bool = False,
        aug_after_mvn: bool = False,
    ) -> None:
        """Build the feature-fusion frontend.

        Args:
            feat_fuser: Configuration dictionary for ``FeatFuserFactory``.
            mvn: Optional configuration dictionary for ``MeanVarianceNorm``.
            spec_augment: Optional configuration dictionary for ``SpecAugment``.
            trans: If ``True``, returns features as ``(B, C, T)`` instead of ``(B, T, C)``.
            aug_after_mvn: If ``True``, applies SpecAugment after MVN; otherwise before MVN.
        """
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
    def fuser_type(self) -> str:
        """Return the underlying feature fuser type.

        Returns:
            str: Fuser type string from the filtered fuser configuration.
        """
        return self.feat_fuser_cfg["fuser_type"]

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        feats_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse features and apply optional normalization/augmentation.

        Args:
            feats: Sequence of feature tensors to fuse.
            feats_lengths: Optional sequence lengths associated with ``feats``.

        Returns:
            Tuple ``(feats, feats_lengths)`` after fusion, optional augmentation,
            optional MVN, and optional transpose.
        """
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

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the class name from the base config.

        Returns:
            Dict[str, Any]: Model configuration.
        """
        config = {
            "feat_fuser": self.feat_fuser_cfg,
            "mvn": self.mvn_cfg,
            "spec_augment": self.spec_augment_cfg,
            "trans": self.trans,
            "aug_after_mvn": self.aug_after_mvn,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs to those accepted by the constructor.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing only supported constructor keys.
        """
        valid_args = ("feat_fuser", "mvn", "spec_augment", "trans", "aug_after_mvn")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Register CLI arguments for ``FeatFuserMVN``.

        Args:
            parser: Target argument parser.
            prefix: Optional top-level namespace to nest the arguments under.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        FFF.add_class_args(parser, prefix="feat_fuser")
        MVN.add_class_args(parser, prefix="mvn")
        SpecAugment.add_class_args(parser, prefix="spec_augment")
        parser.add_argument(
            "--aug-after-mvn",
            default=False,
            action=ActionYesNo,
            help=("do spec augment after st-mvn," "instead of before"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
