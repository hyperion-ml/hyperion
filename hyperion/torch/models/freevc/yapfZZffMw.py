"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs.audio_feats_mvn import AudioFeatsMVN
from ...narchs.hifi_generator import HiFiGenerator
from ...narchs.nvp_flow import WaveNetNVPFlow as NVPFlow
from ...narchs.wavenet_posterior_encoder import WaveNetPosteriorEncoder
from ...tpm import HFWavLM
from .freevc import FreeVC


class HFWavLMFreeVC(FreeVC):
    """
    FreeVC model for voice conversion using Hugging Face WavLM.
    This class extends the FreeVC model to utilize WavLM as the encoder.
    """
    def __init__(
        self,
        hf_feats: Union[Dict, HFWavLM],
        audio_feats: Union[Dict[str, Any], AudioFeatsMVN],
        prior_encoder: Union[Dict[str, Any], WaveNetPosteriorEncoder],
        prior_flow: Union[Dict[str, Any], NVPFlow],
        posterior_encoder: Union[Dict[str, Any], WaveNetPosteriorEncoder],
        decoder: Union[Dict[str, Any], HiFiGenerator],
        internal_feats: int = 192,
        speaker_feats: int = 192,
        l2_norm_speaker: bool = False,
    ):
        if isinstance(hf_feats, dict):
            if "class_name" in hf_feats:
                del hf_feats["class_name"]
            hf_feats = HFWavLM(**hf_feats)
        else:
            assert isinstance(hf_feats, HFWavLM)

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWavLM.add_class_args(parser, prefix="hf_feats")
        FreeVC.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
