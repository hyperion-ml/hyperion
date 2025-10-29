"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:
    from src.model.fluency.whisper_fluency import (
        WhisperWrapper as VoxProfileFluencyModel,
    )
except ImportError:
    VoxProfileFluencyModel = None

VOXPROFILE_FLUENCY_MAX_AUDIO_LEN = 3.0  # seconds

# Label List
FLUENCY_CLASSES = ["fluent", "disfluent"]

DISFLUENCY_TYPES = [
    "block",
    "prolongation",
    "sound-repetition",
    "word-repetition",
    "interjection",
]


class VoxProfileFluencyEvaluator(VoxProfileEvaluator):
    """Evaluate fluency and disfluency types using a Whisper-based classifier.

    Attributes:
        model: Loaded fluency model used for inference.
        device: Torch device on which the model runs.
        max_batch_length: Maximum duration (seconds) processed per batch.
        output_prefix: Prefix applied to output keys in the results.
        return_logits: Whether logits are included alongside probabilities.
        return_per_window_values: Whether per-window predictions are returned.
        disfluency_type_threshold: Sigmoid threshold applied to disfluency types.
    """

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/whisper-large-v3-speech-flow",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_fluency",
        return_logits: bool = False,
        return_per_window_values: bool = False,
        disfluency_type_threshold: float = 0.7,
    ):
        """Instantiate the fluency evaluator.

        Args:
            model_path: Hugging Face identifier or local path to the model weights.
            device: Torch device used for evaluation.
            max_batch_length: Maximum audio length (seconds) processed per batch.
            output_prefix: Prefix for emitted result keys.
            return_logits: Whether to include raw logits in outputs.
            return_per_window_values: Whether to return per-window details.
            disfluency_type_threshold: Threshold applied to disfluency probabilities.
        """

        if VoxProfileFluencyModel is None:
            raise ImportError(
                "VoxProfileFluencyModel could not be imported. Please install the required dependencies."
            )

        model = VoxProfileFluencyModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_chunk_length=VOXPROFILE_FLUENCY_MAX_AUDIO_LEN,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )
        self.return_per_window_values = return_per_window_values
        self.disfluency_type_threshold = disfluency_type_threshold

    @staticmethod
    def classes() -> List[str]:
        """Return the fluency labels."""
        return FLUENCY_CLASSES

    @staticmethod
    def disfluency_types() -> List[str]:
        """Return the disfluency type labels."""
        return DISFLUENCY_TYPES

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: Iterable[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        """Score a clip, returning fluency, disfluency types, and optional extras."""
        prefix = self.output_prefix
        fluency_preds = []
        disfluency_preds = []
        for audio_batch in audio_batches:
            fluency_preds_i, disfluency_preds_i = self.model(
                audio_batch, return_features=False
            )
            fluency_preds.append(fluency_preds_i)
            disfluency_preds.append(disfluency_preds_i)

        fluency_tensor = torch.cat(fluency_preds, dim=0)
        disfluency_tensor = torch.cat(disfluency_preds, dim=0)
        num_windows = fluency_tensor.shape[0]
        if num_windows == 0:
            raise ValueError(
                "No fluency predictions were generated for the input audio."
            )

        window_winners = fluency_tensor.argmax(dim=1)
        vote_counts = torch.bincount(window_winners, minlength=len(self.classes()))
        winner_idx = vote_counts.argmax().item()
        pred_label = self.classes()[winner_idx]
        pred_votes = vote_counts[winner_idx].item()
        pred_prob = pred_votes / float(num_windows)

        disfluency_probs_per_window = torch.sigmoid(disfluency_tensor)
        disfluency_positive = (
            disfluency_probs_per_window > self.disfluency_type_threshold
        ).int()
        try:
            fluent_idx = self.classes().index("fluent")
            disfluent_idx = self.classes().index("disfluent")
        except ValueError as exc:
            raise ValueError(
                "Fluency classes must include 'fluent' and 'disfluent'."
            ) from exc

        fluent_mask = window_winners == fluent_idx
        if fluent_mask.any():
            disfluency_positive[fluent_mask] = 0

        disfluent_mask = window_winners == disfluent_idx
        if disfluent_mask.any():
            disfluency_counts = disfluency_positive[disfluent_mask].sum(dim=0)
            disfluency_ratios = disfluency_counts.float() / disfluent_mask.sum().float()
        else:
            disfluency_counts = torch.zeros(
                disfluency_positive.shape[1],
                device=disfluency_positive.device,
                dtype=disfluency_positive.dtype,
            )
            disfluency_ratios = disfluency_counts.float()

        result = {"id": audio_id, prefix: pred_label, f"{prefix}_prob": pred_prob}
        for label, ratio in zip(self.disfluency_types(), disfluency_ratios):
            result[f"{prefix}_disfluency_type_{label}_prob"] = ratio.item()

        if self.return_logits:
            fluency_logit_diff = (
                fluency_tensor[:, fluent_idx] - fluency_tensor[:, disfluent_idx]
            )
            result[f"{prefix}_fluency_logit_diff"] = (
                fluency_logit_diff.detach().cpu().tolist()
            )
            result[f"{prefix}_disfluency_logits"] = (
                disfluency_tensor.detach().cpu().tolist()
            )

        if self.return_per_window_values:
            window_labels = [
                self.classes()[idx] for idx in window_winners.detach().cpu().tolist()
            ]
            result[f"{prefix}_window_labels"] = window_labels
            window_disfluencies: List[List[str]] = []
            disfluency_positive_cpu = disfluency_positive.detach().cpu()
            for win_idx, label in enumerate(window_labels):
                if label == "fluent":
                    window_disfluencies.append([])
                    continue
                win_disfluencies: List[str] = []
                for d_idx, is_positive in enumerate(disfluency_positive_cpu[win_idx]):
                    if is_positive.item():
                        win_disfluencies.append(self.disfluency_types()[d_idx])
                window_disfluencies.append(win_disfluencies)
            result[f"{prefix}_window_disfluency_labels"] = window_disfluencies

        return result

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI arguments specific to ``VoxProfileFluencyEvaluator``."""
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
                default="tiantiaf/whisper-large-v3-speech-flow",
                help="Path to the pretrained VoxProfile fluency model.",
            )
        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_fluency",
                help="Prefix for the output fields.",
            )

        if "return_per_window_values" not in skip:
            parser.add_argument(
                "--return-per-window-values",
                type=ActionYesNo,
                default=False,
                help="Whether to return per-window fluency and disfluency probabilities.",
            )

        if "disfluency_type_threshold" not in skip:
            parser.add_argument(
                "--disfluency-type-threshold",
                type=float,
                default=0.7,
                help="Threshold for disfluency type detection.",
            )
        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
