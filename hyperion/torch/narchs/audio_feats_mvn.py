"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

import torch.nn as nn

from ..layers import AudioFeatsFactory as AFF
from ..layers import MeanVarianceNorm as MVN
from ..layers import SpecAugment
from .net_arch import NetArch


class AudioFeatsMVN(NetArch):
    """Acoustic Feature Extractor + ST-MVN
    Optional SpecAugment
    """

    def __init__(
        self, audio_feats, mvn=None, spec_augment=None, trans=False, aug_after_mvn=False
    ):
        super().__init__()

        audio_feats = AFF.filter_args(**audio_feats)
        self.audio_feats_cfg = audio_feats
        self.audio_feats = AFF.create(**audio_feats)

        self.mvn = None
        self.mvn_cfg = None
        if mvn is not None:
            mvn = MVN.filter_args(**mvn)
            self.mvn_cfg = mvn
            if mvn["norm_mean"] or mvn["norm_var"]:
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
    def fs(self):
        return self.audio_feats.fs

    @property
    def frame_length(self):
        return self.audio_feats.frame_length

    @property
    def frame_shift(self):
        return self.audio_feats.frame_shift

    def forward(self, x, lengths=None):
        f = self.audio_feats(x)
        if self.spec_augment is not None and not self.aug_after_mvn:
            f = self.spec_augment(f, lengths)

        if self.mvn is not None:
            f = self.mvn(f)

        if self.spec_augment is not None and self.aug_after_mvn:
            f = self.spec_augment(f, lengths)

        if self.trans:
            f = f.transpose(1, 2).contiguous()

        return f

    def get_config(self):
        config = {
            "audio_feats": self.audio_feats_cfg,
            "mvn": self.mvn_cfg,
            "spec_augment": self.spec_augment_cfg,
            "trans": self.trans,
            "aug_after_mvn": self.aug_after_mvn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("audio_feats", "mvn", "spec_augment", "trans", "aug_after_mvn")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AFF.add_class_args(parser, prefix="audio_feats")
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
            # help='feature extraction options')
