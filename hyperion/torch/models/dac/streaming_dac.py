"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HypDataClass
from ....utils.misc import filter_func_args
from ...layers import LoudnessNorm, ResidualVectorQuantizer, VectorQuantizerOutput
from ...narchs import StreamingDACDecoder, StreamingDACEncoder
from ...torch_model import TorchModel
from ...utils.masking import scale_seq_lengths, seq_lengths_to_mask
from .dac import DACOutput


class StreamingDAC(TorchModel):
    """Streaming version of Descript Audio Codec (DAC) top-level model.

    This composes:
      1) an encoder (channels-last I/O: (B, T, C)) -> latents (B, T', D),
      2) a residual vector quantizer (RVQ) over latents,
      3) a decoder mapping quantized latents back to waveform (B, T_out, 1).

    Optionally, an input **loudness normalization** (LUFS) is applied before encoding.

    Attributes:
        encoder: Either a `StreamingDACEncoder` instance or a dict of kwargs for construction.
        quantizer: Either a `ResidualVectorQuantizer` or a dict of kwargs.
        decoder: Either a `StreamingDACDecoder` instance or a dict of kwargs.
        latent_feats: Latent feature dimension D. If None and `encoder` is a dict,
            inferred as `init_inner_channels * 2**len(strides)`. If `encoder`
            is an instance, defaults to `encoder.out_feats`.
        input_sample_freq: Input sampling frequency in Hz (e.g., 44100).
        norm_input_loudness: If True, normalize input loudness to `target_input_lufs`.
        target_input_lufs: Target loudness for normalization in **LUFS** (e.g., -16.0).
    """

    def __init__(
        self,
        encoder: Union[Dict[str, Any], StreamingDACEncoder],
        quantizer: Union[Dict[str, Any], ResidualVectorQuantizer],
        decoder: Union[Dict[str, Any], StreamingDACDecoder],
        latent_feats: Optional[int] = None,
        input_sample_freq: int = 44100,
        norm_input_loudness: bool = False,
        target_input_lufs: float = -16.0,
    ):
        super().__init__()
        if isinstance(encoder, dict):
            if latent_feats is None:
                latent_feats = encoder["init_inner_channels"] * (
                    2 ** len(encoder["strides"])
                )
            encoder["out_feats"] = latent_feats
            encoder = StreamingDACEncoder(**encoder)
        else:
            if latent_feats is None:
                latent_feats = encoder.out_feats

        if isinstance(quantizer, dict):
            quantizer["in_feats"] = latent_feats
            quantizer["channels_last"] = True
            quantizer = ResidualVectorQuantizer(**quantizer)
        if isinstance(decoder, dict):
            decoder["in_feats"] = latent_feats
            decoder = StreamingDACDecoder(**decoder)

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.latent_feats = latent_feats
        self.input_sample_freq = input_sample_freq

        self.hop_length = self.encoder.stride
        self.delay = self.get_delay()
        self.norm_input_loudness = norm_input_loudness
        self.target_input_lufs = target_input_lufs
        if norm_input_loudness:
            self.loudness_norm = LoudnessNorm(
                sample_freq=input_sample_freq, target_lufs=target_input_lufs
            )

    @property
    def input_sample_frequency(self) -> int:
        """Input sample frequency in Hz."""
        return self.input_sample_freq

    @property
    def output_sample_frequency(self) -> int:
        """Output sample frequency in Hz.

        This accounts for total up/downsampling across encoder/decoder.
        """
        return self.input_sample_freq * self.decoder.stride // self.encoder.stride

    @property
    def frame_shift(self):
        """Frame shift (hop length) at the **input** in samples."""
        return self.hop_length

    @property
    def frame_length(self):
        """Total input context length in samples (left + right + center sample)."""
        left_context, right_context = self.in_context()
        return left_context + right_context + 1

    @property
    def encoder_frame_length(self):
        """Encoder-only frame length in samples."""
        return self.encoder.frame_length

    def encoder_in_context(self) -> Tuple[int, int]:
        """(left, right) input context consumed by the encoder, in samples."""
        return self.encoder.in_context()

    def in_context(self) -> Tuple[int, int]:
        """Total (left, right) input context consumed by encoder+decoder, in samples."""
        left_context, right_context = self.encoder.in_context()
        stride = self.encoder.stride
        left_context_dec, right_context_dec = self.decoder.in_context()
        left_context += left_context_dec * stride
        right_context += right_context_dec * stride
        return left_context, right_context

    def max_out_length(self, max_in_length: int) -> int:
        """
        Returns the maximum output length given an input length.

        Args:
            max_in_length (int): Maximum input length in samples.

        Returns:
            int: Maximum output length in samples.
        """
        z_length = self.encoder.max_out_length(max_in_length)
        return self.decoder.max_out_length(z_length)

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns the output lengths given input lengths.
        Args:
            in_lengths (torch.Tensor): Input lengths in samples.

        Returns:
            torch.Tensor: Output lengths in frames.
        """
        z_lengths = self.encoder.out_lengths(in_lengths)
        return self.decoder.out_lengths(z_lengths)

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Predict output tensor shape given an input shape.

        Args:
            in_shape: (B, T_in) or (B, T_in, 1) — only T_in is used here.

        Returns:
            (B, 1, T_out) where T_out is derived from strides/padding.
        """
        B = in_shape[0]
        T = in_shape[1]
        if T is None:
            return (B, 1, None)
        else:
            out_length = self.max_out_length(T)
            return (B, 1, out_length)

    def get_delay(self):
        """Effective algorithmic delay (samples) introduced by encoder+decoder."""
        l_in = self.frame_length * 2
        l_out = self.max_out_length(l_in)
        return (l_in - l_out) // 2

    def get_target_matching_output(
        self,
        audios: torch.Tensor,
        audio_lengths: torch.Tensor,
        pad_left: bool = True,
    ):
        """Prepare **target** audio to match the discriminator/generator path.

        Steps:
          1) (Optional) loudness-normalize inputs (LUFS).
          2) Apply encoder's `preprocess` (e.g., right padding to stride).
          3) Center-crop to match the generator's effective output length.

        Args:
            audios: Input audio, shape (B, 1, T_in), float in [-1, 1].
            audio_lengths: Valid lengths per example (B,) in samples.
            pad_left: if True, in the forward function, we will pad to the left to not miss the beginning of the signal.

        Returns:
            (audios_matched, audio_lengths) with shape (B, 1, T_match).
        """
        with torch.no_grad():
            if self.norm_input_loudness:
                audios = self.loudness_norm(audios)
            audios = self.encoder.preprocess(audios)

        if pad_left:
            # we recover the same length at the output as in the input
            return audios, audio_lengths

        l_in = audios.size(-1)
        l_out = self.max_out_length(l_in)
        delta = l_in - l_out
        if delta > 0:
            audios = audios[..., delta:]
            audio_lengths = audio_lengths - delta

        return audios, audio_lengths

    def encode(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        pad_left: bool = True,
    ) -> VectorQuantizerOutput:
        """Encode waveform and quantize latents.

        Args:
            x: Waveform, shape (B, 1, T_in), float in [-1, 1].
            x_lengths: Valid input lengths (B,) in samples.
            num_quantizers: If set, use only the first N residual VQ stages.

        Returns:
            VectorQuantizerOutput: quantized latents, losses, codes, etc.
        """
        if self.norm_input_loudness:
            x, input_lufs = self.loudness_norm(x, return_input_lufs=True)

        if pad_left:
            x = torch.nn.functional.pad(x, (self.delay, 0), mode="constant", value=0.0)
            if x_lengths is not None:
                x_lengths = x_lengths + self.delay

        z = self.encoder(x, x_lengths)
        z_lengths = scale_seq_lengths(x_lengths, z.shape[1], x.shape[1])
        vq_output = self.quantizer(z, z_lengths, num_quantizers=num_quantizers)

        if self.norm_input_loudness:
            if vq_output.extras is None:
                vq_output.extras = {}
            vq_output.extras["input_lufs"] = input_lufs

        return vq_output

    def decode(self, z: torch.Tensor, z_lengths: Optional[torch.Tensor] = None):
        """Decode quantized latents to waveform.

        Args:
            z: Quantized latents, shape (B, T_z, D).
            z_lengths: Valid latent lengths (B,) in frames.

        Returns:
            Waveform (B, 1, T_out).
        """
        x = self.decoder(z, z_lengths)
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        pad_left: bool = True,
    ):
        """End-to-end forward: encode → RVQ → decode.

        Args:
            x: Waveform, shape (B, 1, T_in), float in [-1, 1].
            x_lengths: Valid input lengths (B,) in samples.
            num_quantizers: If set, use only the first N residual VQ stages.

        Returns:
            DACOutput with reconstructed waveform and VQ info.
        """
        vq_output = self.encode(
            x, x_lengths, num_quantizers=num_quantizers, pad_left=pad_left
        )
        x_recons = self.decode(vq_output.z_q, vq_output.z_lengths)
        x_recons_lengths = scale_seq_lengths(x_lengths, x_recons.shape[1], x.shape[1])
        output = DACOutput(
            x_recons=x_recons, x_recons_lengths=x_recons_lengths, vq=vq_output
        )
        return output

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable config describing the model."""
        return {
            "encoder": self.encoder.get_config(),
            "quantizer": self.quantizer.get_config(),
            "decoder": self.decoder.get_config(),
            "latent_feats": self.latent_feats,
            "input_sample_freq": self.input_sample_freq,
            "norm_input_loudness": self.norm_input_loudness,
            "target_input_lufs": self.target_input_lufs,
        }

    @staticmethod
    def filter_args(**kwargs):
        """
        Filter keyword arguments relevant to `StreamingDACDecoder.__init__`.

        Returns:
            dict: Filtered kwargs usable to instantiate `StreamingDACDecoder`.
        """
        return filter_func_args(StreamingDACDecoder.__init__, kwargs)

    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None):
        """Register DAC model arguments on an `ArgumentParser`.

        If `prefix` is provided, a nested sub-parser is created and attached under
        `--{prefix}` via `ActionParser`.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        StreamingDACDecoder.add_class_args(parser, prefix="encoder", skip={"out_feats"})
        ResidualVectorQuantizer.add_class_args(
            parser, prefix="quantizer", skip={"in_feats"}
        )
        StreamingDACDecoder.add_class_args(parser, prefix="decoder", skip={"in_feats"})
        parser.add_argument(
            "--latent-feats",
            type=int,
            default=None,
            help="Number of latent features.",
        )
        parser.add_argument(
            "--input-sample-freq",
            type=int,
            default=44100,
            help="Input sample frequency.",
        )
        parser.add_argument(
            "--norm-input-loudness",
            action=ActionYesNo,
            default=False,
            help="If true, adds a loudness normalization layer at the input of the encoder.",
        )
        parser.add_argument(
            "--target-input-lufs",
            type=float,
            default=-16.0,
            help="Target loudness level in LUFS for input loudness normalization.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
