"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:

    from src.model.voice_quality.whisper_voice_quality import (
        WhisperWrapper as VoxProfileVoiceQualityModel,
    )
except ImportError:

    VoxProfileVoiceQualityModel = None


# Label List

VOICE_QUALITY_LIST = [
    "shrill",
    "nasal",
    "deep",  # Pitch
    "silky",
    "husky",
    "raspy",
    "guttural",
    "vocal-fry",  # Texture
    "booming",
    "authoritative",
    "loud",
    "hushed",
    "soft",  # Volume
    "crisp",
    "slurred",
    "lisp",
    "stammering",  # Clarity
    "singsong",
    "pitchy",
    "flowing",
    "monotone",
    "staccato",
    "punctuated",
    "enunciated",
    "hesitant",  # Rhythm
]


class VoxProfileVoiceQualityEvaluator(VoxProfileEvaluator):

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-voice-quality",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_voice_quality",
        return_logits: bool = False,
    ):

        if VoxProfileVoiceQualityModel is None:
            raise ImportError(
                "VoxProfileVoiceQualityModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileVoiceQualityModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @property
    def classes(self) -> List[str]:
        return VOICE_QUALITY_LIST

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: List[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        prefix = self.output_prefix
        logits = []
        for audio_batch in audio_batches:
            logits_i = self.model(audio_batch, return_features=False)
            logits.append(logits_i)

        logits = torch.cat(logits, dim=0).mean(dim=0)
        probs = F.sigmoid(logits)

        result = {"id": audio_id}
        for label, prob in zip(self.classes, probs):
            result[f"{prefix}_{label}_prob"] = prob.item()

        if self.return_logits:
            for label, logit in zip(self.classes, logits):
                result[f"{prefix}_{label}_logit"] = logit.item()

        return result

    @staticmethod
    def add_class_args(
        parser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):
        """Register VoxProfileVoiceQualityEvaluator CLI arguments."""
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        super().add_class_args(parser, prefix=None, skip=skip)
        if "model_path" not in skip:
            parser.add_argument(
                "--model-path",
                type=str,
                default="tiantiaf/whisper-large-v3-voice-quality",
                help="Path to the pretrained VoxProfile voice quality model.",
            )
        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_voice_quality",
                help="Prefix for the output fields.",
            )
        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
