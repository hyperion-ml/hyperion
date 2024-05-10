"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Dict, Optional, Tuple, Union

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import AudioFeatsMVN
from ...torch_model import TorchModel
from ...utils import remove_silence


class Wav2RNNTransducer(TorchModel):
    """Base class for models that integrate the acoustic feature extractor and and
    RNN-T Transducer that takes acoustic features as input

    Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      transducer: RNN-T transducer model
    """

    def __init__(self, feats, transducer):

        super().__init__()

        if isinstance(feats, dict):
            feats = AudioFeatsMVN.filter_args(**feats)
            feats["trans"] = False
            feats = AudioFeatsMVN(**feats)
        else:
            assert isinstance(feats, AudioFeatsMVN)

        self.feats = feats
        self.transducer = transducer

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: k2.RaggedTensor,
        vad_samples: Optional[torch.Tensor] = None,
        vad_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if vad_samples is not None:
            x, x_lengths = remove_silence(x, x_lengths)
        feats, feat_lengths = self.feats(x, x_lengths)
        if vad_feats is not None:
            feats, feat_lengths = remove_silence(feats, feat_lengths)

        return self.transducer(feats, feat_lengths, y)

    def set_train_mode(self, mode):
        self.transducer.set_train_mode(mode)

    def get_config(self):
        feat_cfg = self.feats.get_config()
        xvector_cfg = self.transducer.get_config()
        config = {
            "feats": feat_cfg,
            "transducer": xvector_cfg,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        """Filters Wav2XVector class arguments from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with SpecAugment options.
        """
        valid_args = (
            "feats",
            "transducer",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds Wav2XVector options common to all child classes to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AudioFeatsMVN.add_class_args(parser, prefix="feats")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
