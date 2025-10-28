"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import PathLike
from .voxprofile_evaluator import VOXPROFILE_MAX_AUDIO_LEN, VoxProfileEvaluator

try:
    from src.model.age_sex.wavlm_demographics import (
        WavLMWrapper as VoxProfileAgeSexModel,
    )

except ImportError:
    VoxProfileAccentModel = None
    VoxProfileAgeSexModel = None
    VoxProfileVoiceQualityModel = None
    VoxProfileFluencyModel = None


SEX_LIST = ["f", "m"]


class VoxProfileAgeSexEvaluator(VoxProfileEvaluator):

    def __init__(
        self,
        model_path: PathLike = "tiantiaf/wavlm-large-age-sex",
        device: Union[int, torch.device, str] = 0,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile_demographics",
        return_logits: bool = False,
    ):

        if VoxProfileAgeSexModel is None:
            raise ImportError(
                "VoxProfileAgeSexModel could not be imported. Please install the required dependencies."
            )
        model = VoxProfileAgeSexModel.from_pretrained(model_path)
        super().__init__(
            model=model,
            device=device,
            max_batch_length=max_batch_length,
            output_prefix=output_prefix,
            return_logits=return_logits,
        )

    @property
    def sex_classes(self) -> List[str]:
        return SEX_LIST

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: List[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        prefix = self.output_prefix
        age_preds = []
        sex_logits = []
        for audio_batch in audio_batches:
            age_preds_i, sex_logits_i = self.model(audio_batch)
            age_preds.append(age_preds_i)
            sex_logits.append(sex_logits_i)

        sex_logits = torch.cat(sex_logits, dim=0).mean(dim=0)
        sex_probs = F.softmax(sex_logits, dim=-1)
        pred = sex_probs.argmax().item()
        pred_label = self.sex_classes[pred]
        pred_prod = sex_probs[pred].item()
        age_preds = torch.cat(age_preds, dim=0).mean().item() * 100
        result = {
            "id": audio_id,
            f"{prefix}_sex": pred_label,
            f"{prefix}_sex_prob": pred_prod,
            f"{prefix}_age": age_preds,
        }
        if self.return_logits:
            for label, logit in zip(self.sex_classes, sex_logits):
                result[f"{prefix}_sex_logit_{label}"] = logit.item()

        return result

    @staticmethod
    def add_class_args(
        parser, prefix: Optional[str] = None, skip: Optional[set] = None
    ):
        """Register VoxProfileAgeSexEvaluator CLI arguments."""
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
                default="tiantiaf/wavlm-large-age-sex",
                help="Path to the pretrained VoxProfile age and sex model.",
            )

        if "output_prefix" not in skip:
            parser.add_argument(
                "--output-prefix",
                type=str,
                default="voxprofile_demographics",
                help="Prefix for the output fields.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
