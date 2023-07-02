"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...tpm import HFWav2Vec2
from ..transducer import RNNTransducer
from ..xvectors import ResNet1dXVector as ResNet1dLanguageID
from ..wav2languageid import HFWav2Vec2ResNet1dLanguageID
from ..wav2transducer import HFWav2Vec2RNNTransducer


from .hf_wav2rnn_transducer_languageid import HFWav2RNNTransducerLanguageID


class HFWav2Vec2RNNTransducerResnet1D(HFWav2RNNTransducerLanguageID):
    """Class for RNN-T with Wav2Vec2 features

    Attributes:
      Attributes:
      hf_feats: HFWav2Vec configuration dictionary or object.
                This is a warpper over Hugging Face Wav2Vec model.
      transducer: Transducer configuration dictionary or object.
      feat_fusion_start: the input to x-vector model will fuse the wav2vec layers from "feat_fusion_start" to
                         the wav2vec "num_layers".
      feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used.
    """

    def __init__(
        self,
        hf_feats: Union[Dict, HFWav2Vec2],
        transducer: Union[Dict, RNNTransducer],
        languageid: Union[Dict, ResNet1dLanguageID],
        feat_fusion_start: int = 0,
        feat_fusion_method_transducer: str = "weighted-avg",
        feat_fusion_method_lid: str = "weighted-avg",
        loss_weight_transducer: float = 0.005,
        loss_weight_lid: float = 1.0,
        lid_length: float = 3.0,
    ):

        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        # if isinstance(languageid, dict):
        #     languageid["resnet_enc"]["in_feats"] = hf_feats.hidden_size
        #     if "class_name" in languageid:
        #         del languageid["class_name"]
        #     languageid = ResNet1dLanguageID(**languageid)
        # else:
        #     assert isinstance(languageid, ResNet1dLanguageID)
        #     assert languageid.encoder_net.in_feats == hf_feats.hidden_size

        # hf_feats = wav2transducer.hf_feats
        # transducer = wav2transducer.transducer
        # languageid = wav2languageid.languageid


        super().__init__(hf_feats, transducer, languageid, feat_fusion_start,
                         feat_fusion_method_transducer, feat_fusion_method_lid, loss_weight_transducer, loss_weight_lid, lid_length)

    @staticmethod
    def filter_args(**kwargs):
        base_args = HFWav2RNNTransducerLanguageID.filter_args(**kwargs)
        child_args = HFWav2Vec2.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNTransducer.filter_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        child_args = ResNet1dLanguageID.filter_args(**kwargs["languageid"])
        base_args["languageid"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2.add_class_args(parser, prefix="hf_feats")
        RNNTransducer.add_class_args(parser, prefix="transducer")
        # HFWav2RNNTransducer.add_class_args(parser)
        ResNet1dLanguageID.add_class_args(parser, prefix="languageid")
        HFWav2RNNTransducerLanguageID.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):

        valid_args = (
            "loss_weight_transducer",
            "loss_weight_lid",
            "lid_length",
        )

        base_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        child_args = HFWav2Vec2.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNTransducer.filter_finetune_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        child_args = ResNet1dLanguageID.filter_finetune_args(**kwargs["languageid"])
        base_args["languageid"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--loss-weight-transducer",
            default=0.005,
            type=float,
            help="""
            The weight of the transducer loss
            """,
        )

        parser.add_argument(
            "--loss-weight-lid",
            default=1.0,
            type=float,
            help="""
            The weight of the lid loss
            """,
        )

        parser.add_argument(
            "--lid-length",
            default=3.0,
            type=float,
            help="""
            The length of the chunks for language id
            """,
        )

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats")
        RNNTransducer.add_finetune_args(parser, prefix="transducer")
        ResNet1dLanguageID.add_finetune_args(parser, prefix="languageid")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
