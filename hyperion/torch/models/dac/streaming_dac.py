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

from ....utils import HyperDataClass
from ....utils.misc import filter_func_args
from ...hyper_torch_model import HyperTorchModel
from ...layers import LoudnessNorm, ResidualVectorQuantizer, VectorQuantizerOutput
from ...narchs import (
    StreamingDACDecoder,
    StreamingDACDecoderState,
    StreamingDACEncoder,
    StreamingDACEncoderState,
)
from .dac import DACOutput, DACTrainMode


@dataclass
class StreamingDACState(HyperDataClass):
    """Aggregated cache state for `StreamingDAC`.

    Attributes:
        encoder_state: Cache state for the `StreamingDACEncoder`.
        decoder_state: Cache state for the `StreamingDACDecoder`.
    """

    encoder_state: StreamingDACEncoderState
    decoder_state: StreamingDACDecoderState


class StreamingDAC(HyperTorchModel):
    """Streaming version of Descript Audio Codec (DAC) top-level model.

    This composes:
      1) an encoder (audio input (B, T) or (B, C, T)) -> latents (B, T', D),
      2) a residual vector quantizer (RVQ) over latents,
      3) a decoder mapping quantized latents back to waveform (B, 1, T_out).

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
        alignment_look_ahead: int = 0,
    ):
        """Construct a streaming DAC model.

        Args:
            encoder: Encoder instance or configuration dictionary.
            quantizer: Residual vector quantizer instance or configuration dictionary.
            decoder: Decoder instance or configuration dictionary.
            latent_feats: Latent feature dimension. If ``None``, infer it from the encoder.
            input_sample_freq: Input sampling frequency in Hz.
            norm_input_loudness: If ``True``, normalize inputs to ``target_input_lufs``.
            target_input_lufs: Target loudness in LUFS for input normalization.
            alignment_look_ahead: Number of samples of look-ahead used for alignment.
        """
        super().__init__()
        if isinstance(encoder, dict):
            init_inner_channels = encoder.get("init_inner_channels") or 64
            strides = encoder.get("strides") or [2, 4, 8, 8]
            if latent_feats is None:
                latent_feats = init_inner_channels * (2 ** len(strides))
            encoder["out_feats"] = latent_feats
            encoder.setdefault("init_inner_channels", init_inner_channels)
            encoder.setdefault("strides", strides)
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
        self.norm_input_loudness = norm_input_loudness
        self.target_input_lufs = target_input_lufs
        if norm_input_loudness:
            self.loudness_norm = LoudnessNorm(
                sample_freq=input_sample_freq, target_lufs=target_input_lufs
            )

        self.alignment_look_ahead = alignment_look_ahead
        self.register_buffer("vq_is_valid", torch.zeros(1, dtype=torch.bool))

        self.delay = self.get_delay()

    def change_config(
        self,
        rebuild_quantizer: bool = False,
        reconfig_quantizer: bool = False,
        quantizer: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Change internal model configuration during finetuning.
        Args:
            rebuild_quantizer: If true, rebuilds the quantizer with new config.
            reconfig_quantizer: If true, reconfigures the quantizer with new config.
            quantizer: New quantizer config dict.
        """
        if rebuild_quantizer and quantizer is not None:
            quantizer["in_feats"] = self.latent_feats
            quantizer["channels_last"] = True
            self.quantizer = ResidualVectorQuantizer(**quantizer)

        if reconfig_quantizer and quantizer is not None:
            if self.quantizer.base_vq_type == quantizer.get(
                "base_vq_type", self.quantizer.base_vq_type
            ):
                logging.info("reconfiguring quantizer with new config")
                self.quantizer.change_config(**quantizer)
            else:
                quantizer["in_feats"] = self.latent_feats
                quantizer["channels_last"] = True
                new_quantizer = ResidualVectorQuantizer(**quantizer)
                new_quantizer.init_from_rvq(self.quantizer)
                self.quantizer = new_quantizer

    @property
    def input_sample_frequency(self) -> int:
        """Input sample frequency in Hz.

        Returns:
            Input sampling frequency in Hz.
        """
        return self.input_sample_freq

    @property
    def output_sample_frequency(self) -> int:
        """Output sample frequency in Hz.

        This accounts for total up/downsampling across encoder/decoder.

        Returns:
            Output sampling frequency in Hz.
        """
        return self.input_sample_freq * self.decoder.stride // self.encoder.stride

    @property
    def frame_shift(self) -> int:
        """Frame shift at the input in samples.

        Returns:
            Input hop length in samples.
        """
        return self.hop_length

    @property
    def frame_length(self) -> int:
        """Total input context length in samples.

        Returns:
            Total context length in samples, including left and right context and the center sample.
        """
        left_context, right_context = self.in_context()
        return int(left_context + right_context + 1)

    @property
    def encoder_frame_length(self) -> int:
        """Encoder-only frame length in samples.

        Returns:
            Encoder frame length in samples.
        """
        return self.encoder.frame_length

    def encoder_in_context(self) -> Tuple[int, int]:
        """Return the encoder input context.

        Returns:
            A ``(left_context, right_context)`` tuple in samples.
        """
        return self.encoder.in_context()

    def in_context(self) -> Tuple[int, int]:
        """Return the total input context consumed by encoder and decoder.

        Returns:
            A ``(left_context, right_context)`` tuple in samples.
        """
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
            torch.Tensor: Output lengths in samples.
        """
        z_lengths = self.encoder.out_lengths(in_lengths)
        return self.decoder.out_lengths(z_lengths)

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Predict output tensor shape given an input shape.

        Args:
            in_shape: (B, T_in) or (B, C, T_in). The last dimension is time.

        Returns:
            (B, C_out, T_out) where T_out is derived from strides/padding.
        """
        B = in_shape[0]
        T = in_shape[-1]
        if T is None:
            return (B, self.decoder.out_feats, None)
        else:
            out_length = self.max_out_length(T)
            return (B, self.decoder.out_feats, out_length)

    def get_delay(self) -> int:
        """Return the effective algorithmic delay in samples.

        Returns:
            Algorithmic delay in samples.
        """
        return self.alignment_look_ahead

    @torch.no_grad()
    def update_quantizer_hyperparams(self, global_step: int) -> None:
        """Update internal quantizer hyperparameters.

        Args:
            global_step: Current global training step.
        """
        if self.vq_is_valid.item():
            self.quantizer.update_hyperparams(global_step)

    # @torch.no_grad()
    # def get_target_matching_output(
    #     self,
    #     audios: torch.Tensor,
    #     audio_lengths: torch.Tensor,
    #     alignment: str = "start",
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Prepare **target** audio to match the discriminator/generator path.

    #     Steps:
    #       1) (Optional) loudness-normalize inputs (LUFS).
    #       2) Apply encoder's `preprocess` (e.g., right padding to stride).
    #       3) Crop or pad to match the generator's effective output length.

    #     Args:
    #         audios: Input audio, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
    #         audio_lengths: Valid lengths per example (B,) in samples.
    #         alignment: Anchor used when crop/pad is needed. Use "start" to keep
    #             sample 0 aligned, or "center" to align waveform centers.

    #     Returns:
    #         (audios_matched, audio_lengths) with audio shape (B, C, T_match).
    #     """
    #     if alignment not in {"start", "center"}:
    #         raise ValueError(f"invalid alignment={alignment}")

    #     input_lengths = audio_lengths
    #     if self.norm_input_loudness:
    #         audios = self.loudness_norm(audios)

    #     if self.alignment_look_ahead > 0:
    #         audios = torch.nn.functional.pad(
    #             audios, (self.alignment_look_ahead, 0), mode="constant", value=0.0
    #         )
    #         audio_lengths = audio_lengths + self.alignment_look_ahead

    #     audios = self.encoder.preprocess(audios)
    #     if self.alignment_look_ahead > 0:
    #         audios = audios[..., : -self.alignment_look_ahead]
    #         audio_lengths = audio_lengths - self.alignment_look_ahead

    #     l_in = audios.size(-1)
    #     l_out = self.max_out_length(l_in)
    #     delta = l_in - l_out
    #     if delta > 0:
    #         if alignment == "start":
    #             audios = audios[..., :l_out]
    #         else:
    #             start = delta // 2
    #             stop = delta - start
    #             audios = audios[..., start : l_in - stop]
    #     elif delta < 0:
    #         pad = -delta
    #         if alignment == "start":
    #             left_pad = 0
    #             right_pad = pad
    #         else:
    #             left_pad = pad // 2
    #             right_pad = pad - left_pad
    #         audios = torch.nn.functional.pad(audios, (left_pad, right_pad))

    #     output_lengths = self.out_lengths(audio_lengths)
    #     audio_lengths = torch.minimum(output_lengths, audio_lengths)
    #     audio_lengths = audio_lengths.clamp(min=0, max=audios.size(-1))
    #     return audios, audio_lengths

    @torch.no_grad()
    def get_target_matching_output(
        self,
        audios: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare **target** audio to match the discriminator/generator path.

        Steps:
          1) (Optional) loudness-normalize inputs (LUFS).
          2) Pad the beginning of the audio with `alignment_look_ahead` samples, if needed.
          3) Apply encoder's `preprocess` (e.g., right padding to stride).

        Args:
            audios: Input audio, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
            audio_lengths: Valid lengths per example (B,) in samples.

        Returns:
            (audios_matched, audio_lengths) with audio shape (B, C, T_match).
        """
        if self.norm_input_loudness:
            audios = self.loudness_norm(audios)

        if self.alignment_look_ahead > 0:
            audios = torch.nn.functional.pad(
                audios, (self.alignment_look_ahead, 0), mode="constant", value=0.0
            )
            audio_lengths = audio_lengths + self.alignment_look_ahead

        audios = self.encoder.preprocess(audios)
        audio_lengths = audio_lengths.clamp(min=0, max=audios.size(-1))
        return audios, audio_lengths

    def encode(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        return_codes: bool = False,
    ) -> VectorQuantizerOutput:
        """Encode waveform and quantize latents.

        Args:
            x: Waveform, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
            x_lengths: Valid input lengths (B,) in samples.
            num_quantizers: If set, use only the first N residual VQ stages.
            return_codes: If True, include quantizer code indices in the output.

        Returns:
            VectorQuantizerOutput: quantized latents, losses, codes, etc.
        """
        if self.norm_input_loudness:
            x, input_lufs = self.loudness_norm(x, return_input_lufs=True)

        if self.delay > 0:
            x = torch.nn.functional.pad(
                x,
                (0, self.alignment_look_ahead),
                mode="constant",
                value=0.0,
            )
            if x_lengths is not None:
                x_lengths = x_lengths + self.delay

        z, z_lengths = self.encoder(x, x_lengths)

        if (
            self.training
            and self.train_mode != DACTrainMode.NO_VQ
            and not self.vq_is_valid.item()
        ):
            self.vq_is_valid.fill_(True)

        if self.vq_is_valid.item():
            vq_output = self.quantizer(
                z,
                z_lengths,
                num_quantizers=num_quantizers,
                return_codes=return_codes,
            )
        else:
            vq_output = VectorQuantizerOutput(
                z_q=z,
                z_lengths=z_lengths,
                codebook_loss=torch.as_tensor(0.0, device=z.device),
                perplexity=torch.zeros((1,), device=z.device),
                commitment_loss=torch.as_tensor(0.0, device=z.device),
                codes=None,
                extras=None,
            )

        if self.norm_input_loudness:
            if vq_output.extras is None:
                vq_output.extras = {}
            vq_output.extras["input_lufs"] = input_lufs

        return vq_output

    def decode(
        self, z: torch.Tensor, z_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode quantized latents to waveform.

        Args:
            z: Quantized latents, shape (B, T_z, D).
            z_lengths: Valid latent lengths (B,) in frames.

        Returns:
            Waveform (B, 1, T_out) and valid output lengths in samples.
        """
        x, x_lengths = self.decoder(z, z_lengths)
        return x, x_lengths

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        return_codes: bool = False,
    ) -> DACOutput:
        """End-to-end forward: encode → RVQ → decode.

        Args:
            x: Waveform, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
            x_lengths: Valid input lengths (B,) in samples.
            num_quantizers: If set, use only the first N residual VQ stages.
            return_codes: If True, include quantizer code indices in `DACOutput.vq`.

        Returns:
            DACOutput with reconstructed waveform and VQ info.
        """
        vq_output = self.encode(
            x,
            x_lengths,
            num_quantizers=num_quantizers,
            return_codes=return_codes,
        )
        x_recons, x_recons_lengths = self.decode(vq_output.z_q, vq_output.z_lengths)
        output = DACOutput(
            x_recons=x_recons, x_recons_lengths=x_recons_lengths, vq=vq_output
        )
        return output

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StreamingDACState:
        """Initialize cache state for streaming inference

        Args:
            batch_size: Batch size.
            device: Device of the state tensors. If None, uses model parameters device.
            dtype: Data type of the state tensors. If None, uses model parameters dtype.

        Returns:
            StreamingDACState: Initial cache state.
        """
        encoder_state = self.encoder.init_state(batch_size, device=device, dtype=dtype)
        decoder_state = self.decoder.init_state(batch_size, device=device, dtype=dtype)
        return StreamingDACState(
            encoder_state=encoder_state, decoder_state=decoder_state
        )

    @torch.no_grad()
    def stream(
        self,
        x: torch.Tensor,
        state: StreamingDACState,
        flush: bool = False,
    ) -> Tuple[DACOutput, StreamingDACState]:
        """Streaming inference step.

        Args:
            x: Input waveform chunk, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
            state: Current cache state.
            flush: If True, flush encoder/decoder buffered context (final chunk).

        Returns:
            A `DACOutput` with the reconstructed chunk and VQ info, plus the updated cache state.
        """
        if self.norm_input_loudness:
            x, _ = self.loudness_norm(x, return_input_lufs=True)

        # Encode
        z, new_encoder_state = self.encoder.stream(x, state.encoder_state, flush=flush)

        # Quantize
        if self.vq_is_valid.item():
            vq_output = self.quantizer(z)
        else:
            vq_output = VectorQuantizerOutput(
                z_q=z,
                z_lengths=None,
                codebook_loss=torch.as_tensor(0.0, device=z.device),
                perplexity=torch.zeros((1,), device=z.device),
                commitment_loss=torch.as_tensor(0.0, device=z.device),
                codes=None,
                extras=None,
            )

        z_q = vq_output.z_q

        # Decode
        x_out, new_decoder_state = self.decoder.stream(
            z_q, state.decoder_state, flush=flush
        )

        new_state = StreamingDACState(
            encoder_state=new_encoder_state, decoder_state=new_decoder_state
        )
        output = DACOutput(x_recons=x_out, vq=vq_output)
        return output, new_state

    @torch.no_grad()
    def stream_encode(
        self,
        x: torch.Tensor,
        state: StreamingDACEncoderState,
        flush: bool = False,
        return_codes: bool = True,
    ) -> Tuple[torch.Tensor, StreamingDACEncoderState]:
        """Encode one streaming input chunk into codes or quantized latents.

        In eval mode, decoding the returned codes with :meth:`stream_decode`
        reproduces the same quantized latents used by :meth:`stream`.

        Args:
            x: Input waveform chunk, shape (B, T_in) or (B, C, T_in), float in [-1, 1].
            state: Current encoder cache state.
            flush: If True, flush encoder buffered context for the final chunk.
            return_codes: If True, return RVQ code indices with shape (B, M, T_z).
                If False, return quantized latents `z_q` with shape (B, T_z, D).

        Returns:
            Encoded chunk representation and the updated encoder cache state.
        """
        if self.norm_input_loudness:
            x, _ = self.loudness_norm(x, return_input_lufs=True)

        z, new_state = self.encoder.stream(x, state, flush=flush)

        if not self.vq_is_valid.item():
            if return_codes:
                raise RuntimeError("Cannot return codes before the RVQ is valid")
            return z, new_state

        vq_output = self.quantizer(z, return_codes=return_codes)
        if return_codes:
            if vq_output.codes is None:
                raise RuntimeError("RVQ did not return codes")
            return vq_output.codes, new_state

        return vq_output.z_q, new_state

    @torch.no_grad()
    def stream_decode(
        self,
        z: torch.Tensor,
        state: StreamingDACDecoderState,
        flush: bool = False,
        input_is_codes: bool = True,
    ) -> Tuple[torch.Tensor, StreamingDACDecoderState]:
        """Decode one streaming code or latent chunk into waveform samples.

        In eval mode, decoding codes returned by :meth:`stream_encode` produces
        the same waveform chunk as :meth:`stream` for the same decoder state.

        Args:
            z: RVQ codes with shape (B, M, T_z) when `input_is_codes` is True, or
                quantized latents `z_q` with shape (B, T_z, D) otherwise.
            state: Current decoder cache state.
            flush: If True, flush decoder buffered context for the final chunk.
            input_is_codes: If True, decode `z` from RVQ codes before waveform
                synthesis. If False, treat `z` as quantized latents.

        Returns:
            Reconstructed waveform chunk and the updated decoder cache state.
        """
        if input_is_codes:
            if not self.vq_is_valid.item():
                raise RuntimeError("Cannot decode codes before the RVQ is valid")
            z = self.quantizer.decode_codes(z)

        return self.decoder.stream(z, state, flush=flush)

    def set_train_mode(self, mode: str) -> None:
        """Set the training mode and freeze/unfreeze submodules.

        Args:
            mode: Training mode string or `DACTrainMode` value.
        """
        if mode == self._train_mode:
            return
        logging.info("setting DAC train mode to %s", mode)
        if mode == DACTrainMode.FULL:
            self.unfreeze()
        elif mode == DACTrainMode.FROZEN:
            self.freeze()
        else:
            self.unfreeze()
            if mode == DACTrainMode.NO_VQ:
                for p in self.quantizer.parameters():
                    p.requires_grad = False
            elif mode == DACTrainMode.VQ_DECODER:
                self.encoder.freeze()
            elif mode == DACTrainMode.VQ_ONLY:
                self.encoder.freeze()
                self.decoder.freeze()
            else:
                raise ValueError(f"invalid train_mode={mode}")

        # if mode in [DACTrainMode.VQ_ONLY, DACTrainMode.VQ_DECODER]:
        #     logging.info("using torch.no_grad for encoder")
        #     self._encoder_context = torch.no_grad()

        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Switch the model into a supported training regime.

        Args:
            train_mode: Requested training mode.
        """
        if train_mode in [DACTrainMode.FULL, DACTrainMode.FROZEN]:
            super()._train(train_mode)
        elif train_mode in [
            DACTrainMode.NO_VQ,
            DACTrainMode.VQ_DECODER,
            DACTrainMode.VQ_ONLY,
        ]:
            super()._train(DACTrainMode.FULL)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable config describing the model.

        Returns:
            JSON-serializable configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "encoder": self.encoder.get_config(no_class_name=True),
                "quantizer": self.quantizer.get_config(),
                "decoder": self.decoder.get_config(no_class_name=True),
                "latent_feats": self.latent_feats,
                "input_sample_freq": self.input_sample_freq,
                "norm_input_loudness": self.norm_input_loudness,
                "target_input_lufs": self.target_input_lufs,
                "alignment_look_ahead": self.alignment_look_ahead,
            }
        )
        return config

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments relevant to `StreamingDAC.__init__`.

        Args:
            kwargs: Keyword arguments to filter.

        Returns:
            dict: Filtered kwargs usable to instantiate `StreamingDAC`.
        """
        return filter_func_args(StreamingDAC.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register `StreamingDAC` model arguments on an `ArgumentParser`.

        If `prefix` is provided, a nested sub-parser is created and attached under
        `--{prefix}` via `ActionParser`.

        Args:
            parser: Parser to extend.
            prefix: Optional nested parser prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        StreamingDACEncoder.add_class_args(parser, prefix="encoder", skip={"out_feats"})
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
        parser.add_argument(
            "--alignment-look-ahead",
            type=int,
            default=0,
            help="Number of look-ahead samples to add for alignment purposes.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter keyword arguments relevant to `StreamingDAC` finetuning.

        Args:
            kwargs: Keyword arguments to filter.

        Returns:
            dict: Filtered kwargs usable to finetune `StreamingDAC`.
        """
        return filter_func_args(StreamingDAC.change_config, kwargs)

    @staticmethod
    def add_finetune_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register `StreamingDAC` finetune arguments on an `ArgumentParser`.

        If `prefix` is provided, a nested sub-parser is created and attached under
        `--{prefix}` via `ActionParser`.

        Args:
            parser: Parser to extend.
            prefix: Optional nested parser prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--rebuild-quantizer",
            action=ActionYesNo,
            default=False,
            help="If true, rebuilds the quantizer during finetuning.",
        )
        parser.add_argument(
            "--reconfig-quantizer",
            action=ActionYesNo,
            default=False,
            help="If true, reconfigures the quantizer during finetuning.",
        )
        ResidualVectorQuantizer.add_class_args(
            parser, prefix="quantizer", skip={"in_feats"}
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


@torch.no_grad()
def stream_dac_demo(
    B: int = 1,
    T: int = 2048,
    chunk: int = 512,
    in_feats: int = 1,
    init_inner_channels: int = 4,
    encoder_kernel: int = 5,
    encoder_strides: Optional[List[int]] = None,
    encoder_dilations: Optional[List[int]] = None,
    decoder_kernel: int = 5,
    decoder_strides: Optional[List[int]] = None,
    decoder_dilations: Optional[List[int]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compare full forward vs streaming for the top-level StreamingDAC.

    Args:
        B: Batch size.
        T: Total input length in samples.
        chunk: Streaming chunk size.
        in_feats: Number of input channels.
        init_inner_channels: Initial internal channel count.
        encoder_kernel: Encoder kernel size.
        encoder_strides: Encoder stride configuration.
        encoder_dilations: Encoder dilation configuration.
        decoder_kernel: Decoder kernel size.
        decoder_strides: Decoder stride configuration.
        decoder_dilations: Decoder dilation configuration.
        device: Torch device string.
        dtype: Torch dtype for tensors.

    Returns:
        (y_stream, y_ref)
    """
    torch.manual_seed(0)
    enc_cfg = dict(
        in_feats=in_feats,
        out_feats=init_inner_channels,
        init_inner_channels=init_inner_channels,
        kernel_size=encoder_kernel,
        strides=encoder_strides,
        dilations=encoder_dilations,
    )
    # decoder input channels = encoder output channels
    dec_cfg = dict(
        in_feats=init_inner_channels,
        out_feats=in_feats,
        init_inner_channels=init_inner_channels
        * (2 ** (len(encoder_strides or [2, 4, 8, 8]))),
        kernel_size=decoder_kernel,
        strides=decoder_strides,
        dilations=decoder_dilations,
    )
    model = StreamingDAC(
        encoder=enc_cfg,
        quantizer={
            "in_feats": init_inner_channels,
            "num_groups": 1,
            "num_quantizers": 1,
            "codebook_sizes": 2,
            "channels_last": True,
        },
        decoder=dec_cfg,
        input_sample_freq=16000,
    ).to(device=device, dtype=dtype)
    model.set_train_mode(DACTrainMode.NO_VQ)

    from .dac import DAC

    model_dac = DAC(
        encoder=enc_cfg,
        quantizer={
            "in_feats": init_inner_channels,
            "num_groups": 1,
            "num_quantizers": 1,
            "codebook_sizes": 2,
            "channels_last": True,
        },
        decoder=dec_cfg,
        input_sample_freq=16000,
    ).to(device=device, dtype=dtype)
    model_dac.set_train_mode(DACTrainMode.NO_VQ)

    x_full = torch.randn(B, T, device=device, dtype=dtype)
    out_ref = model(x_full)
    y_ref = out_ref.x_recons
    zq_ref = out_ref.vq.z_q

    out_ref_dac = model_dac(x_full)
    y_ref_dac = out_ref_dac.x_recons
    zq_ref_dac = out_ref_dac.vq.z_q
    print(f"x_length={x_full.size(-1)}")
    print(f"dac frame_shift={model_dac.frame_shift}")
    print(f"streaming dac frame_shift={model.frame_shift}")
    print(
        f"{model_dac.max_out_length(x_full.size(-1))}, {model.max_out_length(x_full.size(-1))}"
    )
    print(
        f"{model_dac.encoder.max_out_length(x_full.size(-1))}, {model.encoder.max_out_length(x_full.size(-1))}"
    )
    print(f"dac y_length={y_ref_dac.size(-1)} stream_dac y_length={y_ref.size(-1)}")
    print(f"dac z_length={zq_ref_dac.size(1)} stream_dac z_length={zq_ref.size(1)}")
    print(f"dac delay={model_dac.delay} stream_dac delay={model.delay}", flush=True)

    state = model.init_state(B, device=device, dtype=dtype)
    outs = []
    z_q = []
    t = 0
    while t < T:
        x_chunk = x_full[:, t : t + chunk]
        flush = (t + x_chunk.size(1)) >= T
        out_emit, state = model.stream(x_chunk, state, flush=flush)
        outs.append(out_emit.x_recons)
        z_q.append(out_emit.vq.z_q)
        t += x_chunk.size(1)

    y_stream = torch.cat(outs, dim=-1)
    zq_stream = torch.cat(z_q, dim=1)
    T_stream = y_stream.size(-1)
    y_ref = y_ref[..., -T_stream:]
    zq_ref = zq_ref[:, -zq_stream.size(1) :]
    print(f"y_ref={y_ref[..., :10]}...{y_ref[..., -10:]}")
    print(f"y_stream={y_stream[..., :10]}...{y_stream[..., -10:]}")
    print(f"y_ref.shape={y_ref.shape}, y_stream.shape={y_stream.shape}")
    print(f"zq_ref.shape={zq_ref.shape}, zq_stream.shape={zq_stream.shape}")
    assert zq_ref.shape == zq_stream.shape, f"{zq_ref.shape=} {zq_stream.shape=}"
    assert y_ref.shape == y_stream.shape, f"{y_ref.shape=} {y_stream.shape=}"
    atol = 1e-5 if dtype == torch.float32 else 5e-4
    rtol = 1e-4 if dtype == torch.float32 else 1e-3
    max_abs = (y_ref - y_stream).abs().max().item()
    assert torch.allclose(y_ref, y_stream, atol=atol, rtol=rtol), f"max_abs={max_abs}"
    return y_stream, y_ref
