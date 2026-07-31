"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import List, Optional, Set, Union

import torch
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:
    from vox_profile.model.accent.whisper_accent import (
        WhisperWrapper as VoxProfileAccentModel,
    )

except ImportError:
    VoxProfileAccentModel = None


# Label List
NARROW_ENG_ACCENT_CLASSES = [
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

BROAD_ENG_ACCENT_CLASSES = ["eng-gbr", "eng-usa", "eng-oth"]


class VoxProfileNarrowAccentEvaluator(VoxProfileEvaluator):
    """Evaluate narrow English accent categories using a Whisper model.

    Attributes:
        return_logits: Whether logits are returned alongside probabilities.
        output_prefix: Prefix prepended to output keys.
        model: Loaded Whisper-based accent classifier.
        device: Torch device used for inference.
        max_batch_length: Maximum duration (seconds) processed per batch.
    """

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-narrow-accent",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_narrow_accent",
        return_logits: bool = False,
    ):
        """Instantiate a narrow-accent evaluator.

        Args:
            model_path: Hugging Face identifier or local path to the accent model.
            device: Torch device where inference runs.
            max_batch_length: Maximum audio length (seconds) processed at once.
            output_prefix: Prefix for the output keys written by the evaluator.
            return_logits: Whether to include raw logits in the output dictionary.
        """

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

    @staticmethod
    def classes() -> List[str]:
        """Return the ordered list of narrow accent labels."""
        return NARROW_ENG_ACCENT_CLASSES

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments specific to ``VoxProfileNarrowAccentEvaluator``."""
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        VoxProfileEvaluator.add_class_args(parser, prefix=None, skip=skip)

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
    """Evaluate broad English accent categories using a Whisper model.

    Attributes:
        return_logits: Whether logits are returned alongside probabilities.
        output_prefix: Prefix prepended to output keys.
        model: Loaded Whisper-based accent classifier.
        device: Torch device used for inference.
        max_batch_length: Maximum duration (seconds) processed per batch.
    """

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-broad-accent",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_broad_accent",
        return_logits: bool = False,
    ):
        """Instantiate a broad-accent evaluator.

        Args:
            model_path: Hugging Face identifier or local path to the accent model.
            device: Torch device where inference runs.
            max_batch_length: Maximum audio length (seconds) processed at once.
            output_prefix: Prefix for the output keys written by the evaluator.
            return_logits: Whether to include raw logits in the output dictionary.
        """

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

    @staticmethod
    def classes() -> List[str]:
        """Return the ordered list of broad accent labels."""
        return BROAD_ENG_ACCENT_CLASSES

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments specific to ``VoxProfileBroadAccentEvaluator``."""
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        VoxProfileEvaluator.add_class_args(parser, prefix=None, skip=skip)

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
