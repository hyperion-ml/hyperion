"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn


from ...torch_model import TorchModel


class HFWav2XVector(TorchModel):
    """Abstract Base class for x-vector models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       xvector: x-vector model object.
    """

    def __init__(self, hf_feats, xvector):

        self.hf_feats = hf_feats
        self.xvector = xvector
