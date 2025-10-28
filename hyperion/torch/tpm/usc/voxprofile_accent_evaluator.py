"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:
    from src.model.accent.whisper_accent import WhisperWrapper as VoxProfileAccentModel

except ImportError:
    VoxProfileAccentModel = None


# Label List
NARROW_ENG_ACCENT_LIST = [
    "east-asia",
    "english",
    "germanic",
    "irish",
    "north-america",
    "northern-irish",
    "oceania",
    "other",
    "romance",
    "scottish",
    "semitic",
    "slavic",
    "south-african",
    "southeast-asia",
    "south-asia",
    "welsh",
]

BROAD_ENG_ACCENT_LIST = ["eng-gbr", "eng-usa", "eng-oth"]


class VoxProfileNarrowAccentEvaluator(VoxProfileEvaluator):

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-narrow-accent",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_narrow_accent",
        return_logits: bool = False,
    ):

        if VoxProfileAccentModel is None:
            raise ImportError(
                "VoxProfileAccentModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileAccentModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @property
    def classes(self) -> List[str]:
        return NARROW_ENG_ACCENT_LIST

    @staticmethod
    def add_class_args(
        parser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):
        """Register VoxProfileNarrowAccentEvaluator CLI arguments."""
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
                default="tiantiaf/whisper-large-v3-narrow-accent",
                help="Path to the pretrained VoxProfile narrow accent model.",
            )

        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_narrow_accent",
                help="Prefix for the output fields.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))


class VoxProfileBroadAccentEvaluator(VoxProfileEvaluator):

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-broad-accent",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_broad_accent",
        return_logits: bool = False,
    ):

        if VoxProfileAccentModel is None:
            raise ImportError(
                "VoxProfileAccentModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileAccentModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @property
    def classes(self) -> List[str]:
        return BROAD_ENG_ACCENT_LIST

    @staticmethod
    def add_class_args(
        parser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):
        """Register VoxProfileNarrowAccentEvaluator CLI arguments."""
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
                default="tiantiaf/whisper-large-v3-broad-accent",
                help="Path to the pretrained VoxProfile broad accent model.",
            )

        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_broad_accent",
                help="Prefix for the output fields.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
