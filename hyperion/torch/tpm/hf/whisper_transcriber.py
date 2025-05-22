"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Union

import numpy as np
import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import pipeline

from ....np.preprocessing import ResamplerToTargetFreq
from ....utils.misc import PathLike


class WhisperTranscriber:
    """Wrapper around the Hugging Face Whisper model to perform audio transcription.

    Arguments:
        pretrained_model_path (`str` or `os.PathLike`):
            File path or Hugging Face Hub identifier to the pre-trained Whisper model.
        task (`str`, optional, defaults to `"transcribe"`):
            Task perform to either `"transcribe"` (same language) or `"translate"` (to English).
        language (`str`, optional):
            Language spoken in the audio. If `None`, the model will attempt to detect it.
        beam_size (`int`, optional, defaults to `5`):
            Beam size used for beam search decoding. Use `1` for greedy decoding.
        chunk_length (`float`, optional, defaults to `30`):
            Length (in seconds) of each audio chunk. Long audio will be split into these chunks.
        chunk_shift (`float`, optional, defaults to `5`):
            Overlap (in seconds) between chunks for better continuity and accuracy.
        return_timestamps (`bool`, optional, defaults to `False`):
            If `True`, includes timestamps with each segment or word in the transcription output.
        temperature (`float`, optional):
            Sampling temperature. Set to `0.0` for deterministic decoding (recommended for ASR).
        length_penalty (`float`, optional):
            Penalty applied to longer sequences during decoding. Values <1 favor shorter outputs.
        device (`int` or `torch.device`, optional, defaults to `0`):
            Device identifier for GPU (`0`, `1`, etc.) or `"cpu"`.
        fp16 (`bool`, optional, defaults to `False`):
            If `True`, performs inference in float16 precision for better performance on GPUs.
    """

    def __init__(
        self,
        pretrained_model_path: PathLike,
        task: str = "transcribe",
        language: Optional[str] = None,
        beam_size: int = 5,
        chunk_length: float = 30,
        chunk_shift: float = 5,
        return_timestamps: bool = False,
        temperature: Optional[float] = None,
        length_penalty: Optional[float] = None,
        device: Union[int, torch.device] = 0,
        fp16: bool = False,
    ):
        self.task = task
        self.language = language
        self.beam_size = beam_size
        self.chunk_length = chunk_length
        self.chunk_shift = chunk_shift
        self.return_timestamps = return_timestamps
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.fp16 = fp16
        self.device = device
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=pretrained_model_path,
            chunk_length_s=chunk_length,
            stride_length_s=chunk_shift,
            device=device,
        )

        if fp16:
            self.pipe.model = self.pipe.model.half().to(self.device)

        self.resampler = ResamplerToTargetFreq(16000.0)

    def __call__(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        fs: Union[int, float],
        language: Optional[str] = None,
    ):
        """
        Transcribe a single audio sample.

        This method processes an in-memory audio signal and returns a transcription using
        the Hugging Face Whisper model.

        Args:
            audio (torch.Tensor or np.ndarray):
                A 1D audio waveform tensor or array representing mono audio. The waveform
                should be normalized to [-1.0, 1.0] and have shape (samples,).
            fs (int or float):
                Sampling rate (in Hz) of the input audio. The audio will be resampled
                internally to 16 kHz if needed.
            language (str, optional):
                ISO 639-1 language code (e.g., 'en', 'es'). Overrides the default or
                detected language if specified.

        Returns:
            dict:
                A dictionary containing transcription output, with at minimum the `"text"` field.
                May include `"segments"` or `"chunks"` if `return_timestamps=True` was set.

        Raises:
            ValueError: If the input audio format is invalid or unsupported.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        audio, fs = self.resampler(audio, fs)
        generate_kwargs = {"task": self.task, "num_beams": self.beam_size}
        if language is not None:
            generate_kwargs["language"] = language
        elif self.language is not None:
            generate_kwargs["language"] = self.language

        if self.length_penalty is not None:
            generate_kwargs["length_penalty"] = self.length_penalty

        if self.temperature is not None:
            generate_kwargs["temperature"] = self.temperature

        result = self.pipe(
            audio,
            return_timestamps=self.return_timestamps,
            generate_kwargs=generate_kwargs,
        )
        return result

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        """
        Add command-line arguments corresponding to the WhisperTranscriber class constructor.

        This method adds all configurable arguments required to instantiate a WhisperTranscriber
        via the command line. It supports optional namespacing using `prefix`, and selective exclusion
        of parameters via the `skip` set.

        Args:
            parser (jsonargparse):
                The argparse parser to which the arguments should be added.
            prefix (str, optional):
                If provided, wraps the arguments under a nested namespace using `ActionParser`.
                Useful for argument grouping in multi-component pipelines.
            skip (set[str], optional):
                A set of constructor argument names (e.g., {"beam_size", "fp16"}) to exclude
                from the parser. Useful when embedding inside higher-level wrappers or presets.
        """

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "pretrained_model_path" not in skip:
            parser.add_argument(
                "--pretrained-model-path",
                type=str,
                default="openai/whisper-large-v3-turbo",
                help="Path to pre-trained Whisper model (local or Hugging Face Hub).",
            )
        if "task" not in skip:
            parser.add_argument(
                "--task",
                type=str,
                default="transcribe",
                choices=["transcribe", "translate"],
                help="Task to perform: 'transcribe' or 'translate'.",
            )

        if "language" not in skip:
            parser.add_argument(
                "--language",
                type=str,
                default=None,
                help="Language code (e.g., 'en', 'es'). If not set, Whisper will detect it.",
            )

        if "beam_size" not in skip:
            parser.add_argument(
                "--beam-size",
                type=int,
                default=5,
                help="Beam size for decoding (1 = greedy).",
            )

        if "chunk_length" not in skip:
            parser.add_argument(
                "--chunk-length",
                type=float,
                default=30.0,
                help="Chunk length in seconds for splitting long audio.",
            )

        if "chunk_shift" not in skip:
            parser.add_argument(
                "--chunk-shift",
                type=float,
                default=5.0,
                help="Chunk overlap in seconds (stride).",
            )

        if "return_timestamps" not in skip:
            parser.add_argument(
                "--return-timestamps",
                default=False,
                action=ActionYesNo,
                help="Include timestamps in the transcription output.",
            )

        if "temperature" not in skip:
            parser.add_argument(
                "--temperature",
                type=float,
                default=None,
                help="Sampling temperature (0.0 = deterministic).",
            )

        if "length_penalty" not in skip:
            parser.add_argument(
                "--length-penalty",
                type=float,
                default=None,
                help="Optional penalty applied to longer outputs.",
            )

        if "fp16" not in skip:
            parser.add_argument(
                "--fp16",
                default=False,
                action=ActionYesNo,
                help="Use float16 precision (faster inference on GPU).",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
