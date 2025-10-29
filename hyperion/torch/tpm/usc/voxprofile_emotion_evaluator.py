"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:
    from src.model.emotion.whisper_emotion import (
        WhisperWrapper as VoxProfileCategoricalEmotionModel,
    )
    from src.model.emotion.whisper_emotion_dim import (
        WhisperWrapper as VoxProfileDimensionalEmotionModel,
    )

except ImportError:
    VoxProfileCategoricalEmotionModel = None
    VoxProfileDimensionalEmotionModel = None


# Label List
EMOTION_CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
    "other",
]

EMOTION_DIMENSIONS = ["arousal", "valence", "dominance"]


class VoxProfileCategoricalEmotionEvaluator(VoxProfileEvaluator):
    """Evaluate categorical emotions using a Whisper-based classifier.

    Attributes:
        model: Loaded categorical-emotion model used for inference.
        device: Torch device on which the model runs.
        max_batch_length: Maximum duration (seconds) processed per batch.
        output_prefix: Prefix applied to output keys.
        return_logits: Whether logits are returned alongside probabilities.
    """

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-msp-podcast-emotion",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_emotion_categorical",
        return_logits: bool = False,
    ):
        """Instantiate the categorical-emotion evaluator.

        Args:
            model_path: Hugging Face identifier or local path to the model weights.
            device: Torch device used for evaluation.
            max_batch_length: Maximum audio length (seconds) processed per batch.
            output_prefix: Prefix for emitted result keys.
            return_logits: Whether to include raw logits in outputs.
        """

        if VoxProfileCategoricalEmotionModel is None:
            raise ImportError(
                "VoxProfileEmotionModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileCategoricalEmotionModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @staticmethod
    def classes() -> List[str]:
        """Return categorical emotion labels."""
        return EMOTION_CLASSES

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: Iterable[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        """Score one clip and return its categorical emotion predictions.

        Args:
            audio_batches: Iterable of tensors containing chunk batches.
            audio_id: Unique identifier for the clip.

        Returns:
            Dictionary with the predicted label, probability, and optional logits.
        """
        prefix = self.output_prefix
        logits = []
        for audio_batch in audio_batches:
            logits_i = self.model(audio_batch, return_features=False)[0]
            logits.append(logits_i)

        logits = torch.cat(logits, dim=0).mean(dim=0)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax().item()
        pred_label = self.classes()[pred]
        pred_prod = probs[pred].item()
        result = {"id": audio_id, prefix: pred_label, f"{prefix}_prob": pred_prod}
        if self.return_logits:
            for label, logit in zip(self.classes(), logits):
                result[f"{prefix}_logit_{label}"] = logit.item()

        return result

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments specific to ``VoxProfileCategoricalEmotionEvaluator``."""
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
                default="tiantiaf/whisper-large-v3-msp-podcast-emotion",
                help="Path to the pretrained VoxProfile categorical emotion model.",
            )

        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_emotion_categorical",
                help="Prefix for the output fields.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))


class VoxProfileDimensionalEmotionEvaluator(VoxProfileEvaluator):
    """Evaluate arousal/valence/dominance using a Whisper-based regressor.

    Attributes:
        model: Loaded dimensional-emotion model used for inference.
        device: Torch device on which the model runs.
        max_batch_length: Maximum duration (seconds) processed per batch.
        output_prefix: Prefix applied to output keys.
        return_logits: Whether logits are returned (not used for dimensional outputs).
    """

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-msp-podcast-emotion-dim",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_emotion_dimensional",
        return_logits: bool = False,
    ):
        """Instantiate the dimensional-emotion evaluator.

        Args:
            model_path: Hugging Face identifier or local path to the model weights.
            device: Torch device used for evaluation.
            max_batch_length: Maximum audio length (seconds) processed per batch.
            output_prefix: Prefix for emitted result keys.
            return_logits: Whether to include logits (unused but accepted for parity).
        """

        if VoxProfileDimensionalEmotionModel is None:
            raise ImportError(
                "VoxProfileDimensionalEmotionModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileDimensionalEmotionModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @staticmethod
    def dimensions() -> List[str]:
        """Return the list of dimensional labels."""
        return EMOTION_DIMENSIONS

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: Iterable[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        """Score one clip and return arousal/valence/dominance predictions."""
        prefix = self.output_prefix
        dim_preds = []
        for audio_batch in audio_batches:
            arousal, valence, dominance = self.model(audio_batch, return_features=False)
            dim_preds_i = torch.stack([arousal, valence, dominance], dim=-1)
            dim_preds.append(dim_preds_i)

        dim_preds = torch.cat(dim_preds, dim=0).mean(dim=0)

        result = {"id": audio_id}
        for dim, pred in zip(self.dimensions(), dim_preds):
            result[f"{prefix}_{dim}"] = pred.item()

        return result

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments specific to ``VoxProfileDimensionalEmotionEvaluator``."""
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
                default="tiantiaf/whisper-large-v3-msp-podcast-emotion-dim",
                help="Path to the pretrained VoxProfile dimensional emotion model.",
            )
        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_emotion_dimensional",
                help="Prefix for the output fields.",
            )
        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
