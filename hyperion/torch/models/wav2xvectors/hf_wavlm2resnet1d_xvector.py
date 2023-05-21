"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Dict, Optional, Union

from jsonargparse import ActionParser, ArgumentParser

import torch
import torch.nn as nn

from ...tpm import HFWavLM
from ..xvectors import ResNet1dXVector
from .hf_wav2xvector import HFWav2XVector


class HFWavLM2ResNet1dXVector(HFWav2XVector):
    """Class extracting WavLM + ResNet1d x-vectors from waveform.

    Attributes:
      Attributes:
      hf_feats: HFWavLM configuration dictionary or object.
                This is a warpper over Hugging Face WavLM model.
      xvector: ResNet1dXVector configuration dictionary or object.
      feat_fusion_start: the input to x-vector model will fuse the WavLM layers from "feat_fusion_start" to
                         the WavLM "num_layers".
      feat_fusion_method: method to fuse the hidden layers from the WavLM model, when more
                           than one layer is used.
    """

    def __init__(
        self,
        hf_feats: Union[Dict, HFWavLM],
        xvector: Union[Dict, ResNet1dXVector],
        feat_fusion_start: int = 0,
        feat_fusion_method: str = "weighted-avg",
    ):

        if isinstance(hf_feats, dict):
            hf_feats = HFWavLM(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWavLM)

        if isinstance(xvector, dict):
            xvector["resnet_enc"]["in_feats"] = hf_feats.hidden_size
            xvector = ResNet1dXVector(**xvector)
        else:
            assert isinstance(xvector, ResNet1dXVector)
            assert xvector.encoder_net.in_feats == hf_feats.hidden_size

        super().__init__(hf_feats, xvector, feat_fusion_start, feat_fusion_method)

    @staticmethod
    def filter_args(**kwargs):

        base_args = HFWav2XVector.filter_args(**kwargs)
        child_args = HFWavLM.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dXVector.filter_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWavLM.add_class_args(parser, prefix="hf_feats")
        ResNet1dXVector.add_class_args(parser, prefix="xvector")
        HFWav2XVector.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = HFWavLM.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dXVector.filter_finetune_args(**kwargs["xvector"])
        base_args["xvector"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWavLM.add_finetune_args(parser, prefix="hf_feats")
        ResNet1dXVector.add_finetune_args(parser, prefix="xvector")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
