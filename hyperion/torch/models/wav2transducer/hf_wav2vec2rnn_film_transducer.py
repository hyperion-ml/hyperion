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
from .hf_wav2rnn_film_transducer import HFWav2RNNFiLMTransducer
from ..transducer import RNNFiLMTransducer

class HFWav2Vec2RNNFiLMTransducer(HFWav2RNNFiLMTransducer):
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
        transducer: Union[Dict, RNNFiLMTransducer],
        feat_fusion_start: int = 0,
        feat_fusion_method: str = "weighted-avg",
    ):

        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWav2Vec2(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWav2Vec2)

        # if isinstance(transducer, dict):
        #     transducer["decoder"]["in_feats"] = hf_feats.hidden_size
        #     transducer["joiner"]["in_feats"] = hf_feats.hidden_size
        #     if "class_name" in transducer:
        #         del transducer["class_name"]
        #     transducer = Transducer(**transducer)
        # else:
        #     assert isinstance(transducer, Transducer)
        #     assert transducer.decoder.in_feats == hf_feats.hidden_size
        #     assert transducer.joiner.in_feats == hf_feats.hidden_size

        super().__init__(hf_feats, transducer, feat_fusion_start,
                         feat_fusion_method)



    @staticmethod
    def filter_args(**kwargs):
        base_args = HFWav2RNNFiLMTransducer.filter_args(**kwargs)
        child_args = HFWav2Vec2.filter_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNFiLMTransducer.filter_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2.add_class_args(parser, prefix="hf_feats")
        RNNFiLMTransducer.add_class_args(parser, prefix="transducer")
        HFWav2RNNFiLMTransducer.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = {}
        child_args = HFWav2Vec2.filter_finetune_args(**kwargs["hf_feats"])
        base_args["hf_feats"] = child_args
        child_args = RNNFiLMTransducer.filter_finetune_args(**kwargs["transducer"])
        base_args["transducer"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2Vec2.add_finetune_args(parser, prefix="hf_feats")
        RNNFiLMTransducer.add_finetune_args(parser, prefix="transducer")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))


