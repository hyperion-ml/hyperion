"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from ..xvectors import ResNet1dXVector
from ...tpm import HFWav2Vec
from .hf_wav2xvector import HFWav2XVector


class HFWav2Vec2ResNet1dXVector(HFWav2XVector):
    """Class extracting ResNet1d x-vectors from waveform.
    It contains acoustic feature extraction, feature normalization and
    ResNet1dXVector extractor.

    Attributes:
      Attributes:
      hf_feats: HFWav2Vec configuration dictionary or object.
                This is a warpper over Hugging Face Wav2Vec model.
      xvector: ResNet1dXVector configuration dictionary or object.
    """

    def __init__(self, hf_feats, xvector):

        if isinstance(hf_feats, dict):
            hf_feats = HFWav2Vec(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec)

        if isinstance(xvector, dict):
            xvector = ResNet1dXVector(**xvector)
        else:
            assert isinstance(xvector, ResNet1dXVector)

        super().__init__(hf_feats, xvector)
