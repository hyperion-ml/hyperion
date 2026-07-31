"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....np.preprocessing import ResamplerToTargetFreq
from ....utils.misc import PathLike

VOXPROFILE_MAX_AUDIO_LEN = 15.0  # seconds
VOXPROFILE_SAMPLE_FREQ = 16000  # Hz


class VoxProfileEvaluator:
    """Base helper to score VoxProfile-style models on variable-length audio.

    Attributes:
        model: Neural network used to produce logits from audio tensors.
        device: Torch device where inference runs.
        max_batch_length: Maximum length in seconds processed per batch.
        max_chunk_samples: Maximum number of samples per chunk fed to the model.
        max_batch_samples: Maximum number of samples across a batch of chunks.
        resampler: Utility that resamples inputs to ``VOXPROFILE_SAMPLE_FREQ``.
        output_prefix: Prefix applied to output keys for this evaluator.
        return_logits: When ``True``, raw logits are added to the result payloads.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[int, torch.device, str] = 0,
        max_chunk_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        max_batch_length: float = VOXPROFILE_MAX_AUDIO_LEN,
        output_prefix: str = "voxprofile",
        return_logits: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            model: Torch module that produces logits when called on audio tensors.
            device: Target device for running inference.
            max_chunk_length: Maximum duration (seconds) for each inference chunk.
            max_batch_length: Maximum duration (seconds) allowed per batch of chunks.
            output_prefix: Prefix added to keys in the returned result dictionary.
            return_logits: If true, include raw logits in the output.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_batch_length = max_batch_length
        self.max_chunk_samples = int(max_chunk_length * VOXPROFILE_SAMPLE_FREQ)
        self.max_batch_samples = int(self.max_batch_length * VOXPROFILE_SAMPLE_FREQ)
        self.resampler = ResamplerToTargetFreq(VOXPROFILE_SAMPLE_FREQ)
        self.output_prefix = output_prefix
        self.return_logits = return_logits

    def _prepare_audio(
        self, audio: Union[np.ndarray, torch.Tensor], fs: float
    ) -> Tuple[List[torch.Tensor], float]:
        """Resample and split an utterance into batches of uniform-length chunks.

        Chunks are padded with zeros so every chunk in a batch shares the same
        number of samples and both the chunk length and the total number of
        samples per batch respect the evaluator limits.

        Args:
            audio: Input waveform, either as a NumPy array or torch tensor.
            fs: Original sampling rate in Hz.

        Returns:
            A tuple with the batches of chunk tensors (List[List[Tensor]]) and the
            resampled sampling rate.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio, fs = self.resampler(audio, fs)
        audio = torch.from_numpy(audio).float().to(self.device)
        if audio.ndim != 1:
            raise ValueError(
                f"VoxProfile expects mono audio. Received shape {audio.shape}"
            )

        total_len = audio.shape[0]
        if total_len == 0:
            return [], fs

        max_chunk = min(self.max_chunk_samples, self.max_batch_samples)
        if max_chunk <= 0:
            raise ValueError(
                "max_chunk_samples and max_batch_samples must be positive integers."
            )

        num_chunks = max(1, int(math.ceil(total_len / max_chunk)))
        chunk_size = min(max_chunk, int(math.ceil(total_len / num_chunks)))

        chunks: List[torch.Tensor] = []
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = audio[start:end]
            if chunk.numel() == 0:
                break
            if chunk.shape[0] < chunk_size:
                pad_len = chunk_size - chunk.shape[0]
                pad = torch.zeros(pad_len, device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            start += chunk_size

        batches: List[torch.Tensor] = []
        current_batch: List[torch.Tensor] = []
        current_batch_len = 0
        for chunk in chunks:
            chunk_len = chunk.shape[0]
            if current_batch_len + chunk_len > self.max_batch_samples and current_batch:
                batches.append(torch.stack(current_batch, dim=0))
                current_batch = []
                current_batch_len = 0

            current_batch.append(chunk)
            current_batch_len += chunk_len

        if current_batch:
            batches.append(torch.stack(current_batch, dim=0))

        return batches, fs

    @staticmethod
    def classes() -> List[str]:
        """Return the list of class labels for this evaluator."""
        raise NotImplementedError

    @torch.no_grad()
    def _score_single(
        self,
        audio_batches: Iterable[torch.Tensor],
        audio_id: str,
    ) -> Dict[str, float]:
        """Score a single utterance and return the formatted prediction payload.

        Args:
            audio_batches: Iterable of tensors to feed into the model. Each tensor
                represents a batch of audio chunks stacked in the model's expected
                input shape.
            audio_id: Identifier for the audio clip, used as the DataFrame index.

        Returns:
            Dictionary containing the predicted label and probability (and logits if
            requested).
        """
        prefix = self.output_prefix
        logits = []
        for audio_batch in audio_batches:
            logits_i = self.model(audio_batch, return_feature=False)
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

    @torch.no_grad()
    def __call__(
        self,
        audios: Sequence[Union[torch.Tensor, np.ndarray]],
        audio_fs: Sequence[float],
        audio_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Compute VoxProfile predictions for a batch of utterances.

        Args:
            audios: Sequence of audio waveforms as NumPy arrays or torch tensors.
            audio_fs: Sequence of sampling rates corresponding to ``audios``.
            audio_ids: Optional sequence of identifiers; defaults to ``range(len(audios))``.

        Returns:
            ``pandas.DataFrame`` indexed by ``audio_ids`` with prediction outputs for
            each clip.
        """
        audio_fs = list(audio_fs)
        if audio_ids is None:
            audio_ids = [str(i) for i in range(len(audios))]
        else:
            audio_ids = list(audio_ids)

        if not (len(audios) == len(audio_fs) == len(audio_ids)):
            raise ValueError(
                "Mismatch between number of audios, sample rates, and audio ids"
            )

        results: List[Dict[str, float]] = []
        for audio, fs, audio_id in zip(audios, audio_fs, audio_ids):
            audio_batches, fs_out = self._prepare_audio(audio, fs)
            clip_result = self._score_single(audio_batches, audio_id)
            results.append(clip_result)

        df = pd.DataFrame(results)
        df.set_index("id", inplace=True)
        return df

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register VoxProfile evaluator arguments with ``jsonargparse`` parsers."""
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "device" not in skip:
            parser.add_argument(
                "--device",
                type=str,
                default="cuda",
                choices=["cpu", "cuda"],
                help="Execution device.",
            )

        if "max_batch_length" not in skip:
            parser.add_argument(
                "--max-batch-length",
                type=float,
                default=VOXPROFILE_MAX_AUDIO_LEN,
                help="Maximum length (in seconds) of audio batches processed at once.",
            )

        if "return_logits" not in skip:
            parser.add_argument(
                "--return-logits",
                action=ActionYesNo,
                default=False,
                help="Whether to return raw logits instead of probabilities.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
