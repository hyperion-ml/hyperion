"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser
from typing import Union, Dict, Optional

import torch
import torch.nn as nn

from ..xvectors import ResNet1dXVector as ResNet1dLanguageID
from ...tpm import HFWav2Vec2
from .hf_wav2languageid import HFWav2LanguageID


class HFWav2Vec2ResNet1dLanguageID(HFWav2LanguageID):
    """Class extracting Wav2Vec2 + ResNet1d language identifications from waveform.

    Attributes:
      Attributes:
      hf_feats: HFWav2Vec configuration dictionary or object.
                This is a warpper over Hugging Face Wav2Vec model.
      languageid: ResNet1dLanguageID configuration dictionary or object.
      feat_fusion_start: the input to language identification model will fuse the wav2vec layers from "feat_fusion_start" to
                         the wav2vec "num_layers".
      feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used.
    """

    def __init__(
        self,
        hf_feats: Union[Dict, HFWav2Vec2],
        languageid: Union[Dict, ResNet1dLanguageID],
        feat_fusion_start: int = 0,
        feat_fusion_end: int = -1,
        feat_fusion_method: str = "weighted-avg",
    ):

        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        if isinstance(languageid, dict):
            languageid["resnet_enc"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in languageid:
                del languageid["class_name"]
            languageid = ResNet1dLanguageID(**languageid)
        else:
            assert isinstance(languageid, ResNet1dLanguageID)
            assert languageid.encoder_net.in_feats == hf_feats.hidden_size

        super().__init__(hf_feats, languageid, feat_fusion_start, feat_fusion_end, feat_fusion_method)

    @staticmethod
    def filter_args(**kwargs):

        base_args = HFWav2LanguageID.filter_args(**kwargs)
        child_args = HFWav2Vec2.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dLanguageID.filter_args(**kwargs["languageid"])
        base_args["languageid"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2.add_class_args(parser, prefix="hf_feats")
        ResNet1dLanguageID.add_class_args(parser, prefix="languageid")
        HFWav2LanguageID.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = HFWav2Vec2.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = ResNet1dLanguageID.filter_finetune_args(**kwargs["languageid"])
        base_args["languageid"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats")
        ResNet1dLanguageID.add_finetune_args(parser, prefix="languageid")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
