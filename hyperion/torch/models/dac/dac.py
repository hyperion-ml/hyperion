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

import numpy as np  # xxx
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F  # xxx
from einops import rearrange  # xxx
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils import weight_norm  # xxx

from ....utils import HypDataClass
from ....utils.misc import filter_func_args
from ...layers import LoudnessNorm, ResidualVectorQuantizer, VectorQuantizerOutput
from ...narchs import DACDecoder, DACEncoder
from ...torch_model import TorchModel
from ...utils.masking import scale_seq_lengths, seq_lengths_to_mask
from ...utils.timers import CUDATimer


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        with torch.no_grad():
            flat_indices = indices.reshape(-1)
            usage = torch.bincount(flat_indices, minlength=self.codebook_size).float()
            total = usage.sum()
            if total.item() > 0:
                probs = usage / total
                perp = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            else:
                perp = torch.tensor(0.0, device=usage.device)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e, perp

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z = z.transpose(1, 2).contiguous()  # (B x D x T)
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []
        perp = torch.zeros((self.n_codebooks,), device=z.device)

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i, perp_i = (
                quantizer(residual)
            )
            perp[i] = perp_i
            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        # latents = torch.cat(latents, dim=1)

        z_q = z_q.transpose(1, 2).contiguous()
        output = VectorQuantizerOutput(
            z_q=z_q,
            codes=codes,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perp,
        )
        return output

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class DACTrainMode(str, Enum):
    """Training modes for the DAC model."""

    FULL = "full"
    FROZEN = "frozen"
    NO_VQ = "no-vq"
    VQ_DECODER = "vq-decoder"
    VQ_ONLY = "vq-only"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of valid training-mode strings."""
        return [o.value for o in DACTrainMode]


@dataclass
class DACOutput(HypDataClass):
    """Container for DAC forward outputs.

    Attributes:
        x_recons: Reconstructed waveform, shape (B, T_out, 1).
        x_recons_lengths: Output lengths in samples (B,), if provided/derived.
        vq: Vector-quantizer output (codes, losses, etc.).
    """

    x_recons: torch.Tensor
    x_recons_lengths: Optional[torch.Tensor] = None
    vq: Optional[VectorQuantizerOutput] = None


class DAC(TorchModel):
    """Descript Audio Codec (DAC) top-level model.

    This composes:
      1) an encoder (channels-last I/O: (B, T, C)) -> latents (B, T', D),
      2) a residual vector quantizer (RVQ) over latents,
      3) a decoder mapping quantized latents back to waveform (B, T_out, 1).

    Optionally, an input **loudness normalization** (LUFS) is applied before encoding.

    Attributes:
        encoder: Either a `DACEncoder` instance or a dict of kwargs for construction.
        quantizer: Either a `ResidualVectorQuantizer` or a dict of kwargs.
        decoder: Either a `DACDecoder` instance or a dict of kwargs.
        latent_feats: Latent feature dimension D. If None and `encoder` is a dict,
            inferred as `init_inner_channels * 2**len(strides)`. If `encoder`
            is an instance, defaults to `encoder.out_feats`.
        input_sample_freq: Input sampling frequency in Hz (e.g., 44100).
        norm_input_loudness: If True, normalize input loudness to `target_input_lufs`.
        target_input_lufs: Target loudness for normalization in **LUFS** (e.g., -16.0).
    """

    def __init__(
        self,
        encoder: Union[Dict[str, Any], DACEncoder],
        quantizer: Union[Dict[str, Any], ResidualVectorQuantizer],
        decoder: Union[Dict[str, Any], DACDecoder],
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
            encoder = DACEncoder.filter_args(**encoder)
            encoder = DACEncoder(**encoder)
        else:
            if latent_feats is None:
                latent_feats = encoder.out_feats

        if isinstance(quantizer, dict):
            quantizer["in_feats"] = latent_feats
            quantizer["channels_last"] = True
            quantizer = ResidualVectorQuantizer(**quantizer)
        if isinstance(decoder, dict):
            decoder["in_feats"] = latent_feats
            decoder = DACDecoder.filter_args(**decoder)
            decoder = DACDecoder(**decoder)

        # self.quantizer2 = ResidualVectorQuantize(1024, 9, 1024, 8, 0.5)

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

        self.register_buffer("vq_is_valid", torch.zeros(1, dtype=torch.bool))

    def change_config(
        self,
        rebuild_quantizer: bool = False,
        quantizer: Optional[Dict[str, Any]] = None,
    ) -> None:
        if rebuild_quantizer and quantizer is not None:
            quantizer["in_feats"] = self.latent_feats
            quantizer["channels_last"] = True
            self.quantizer = ResidualVectorQuantizer(**quantizer)

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
        """Effective algorithmic delay (samples) around a single hop."""
        l_in = self.hop_length
        l_out = self.max_out_length(l_in)
        return (l_in - l_out) // 2

    @torch.no_grad()
    def update_quantizer_hyperparams(self, global_step: int):
        """Update any internal quantizer parameters, e.g., for annealing."""
        self.quantizer.update_hyperparams(global_step)

    def get_target_matching_output(
        self,
        audios: torch.Tensor,
        audio_lengths: torch.Tensor,
    ):
        """Prepare **target** audio to match the discriminator/generator path.

        Steps:
          1) (Optional) loudness-normalize inputs (LUFS).
          2) Apply encoder's `preprocess` (e.g., right padding to stride).
          3) Center-crop to match the generator's effective output length.

        Args:
            audios: Input audio, shape (B, 1, T_in), float in [-1, 1].
            audio_lengths: Valid lengths per example (B,) in samples.

        Returns:
            (audios_matched, audio_lengths) with shape (B, 1, T_match).
        """
        with torch.no_grad():
            if self.norm_input_loudness:
                audios = self.loudness_norm(audios)
            audios = self.encoder.preprocess(audios)

        l_in = audios.size(-1)
        l_out = self.max_out_length(l_in)
        delta = l_in - l_out
        if delta > 0:
            start = delta // 2
            stop = delta - start
            audios = audios[..., start:-stop]
            audio_lengths = (audio_lengths - start).clamp(max=audios.size(-1))

        return audios, audio_lengths

    def encode(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
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
            # self.timer.start("loudness-norm")
            x, input_lufs = self.loudness_norm(x, return_input_lufs=True)
            # self.timer.stop("loudness-norm")

        # self.timer.start("encoder")
        z = self.encoder(x, x_lengths)
        # self.timer.stop("encoder")
        z_lengths = scale_seq_lengths(x_lengths, z.shape[1], x.shape[1])
        # self.timer.start("quantizer")
        # assert not self.vq_is_valid.item()
        if (
            self.training
            and self.train_mode != DACTrainMode.NO_VQ
            and not self.vq_is_valid.item()
        ):
            self.vq_is_valid.fill_(True)

        # assert not self.vq_is_valid.item()
        if self.vq_is_valid.item():
            vq_output = self.quantizer(z, z_lengths, num_quantizers=num_quantizers)
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
            print("Skipping VQ for first forward pass", vq_output, flush=True)

        # vq_output = self.quantizer2(z)
        # self.timer.stop("quantizer")

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
    ):
        """End-to-end forward: encode → RVQ → decode.

        Args:
            x: Waveform, shape (B, 1, T_in), float in [-1, 1].
            x_lengths: Valid input lengths (B,) in samples.
            num_quantizers: If set, use only the first N residual VQ stages.

        Returns:
            DACOutput with reconstructed waveform and VQ info.
        """
        # self.timer = CUDATimer()  # for profiling
        # self.timer.start("total")
        # self.timer.start("encode")
        vq_output = self.encode(x, x_lengths, num_quantizers=num_quantizers)
        # self.timer.stop("encode")
        # self.timer.start("decode")
        x_recons = self.decode(vq_output.z_q, vq_output.z_lengths)
        # self.timer.stop("decode")
        x_recons_lengths = scale_seq_lengths(x_lengths, x_recons.shape[1], x.shape[1])
        output = DACOutput(
            x_recons=x_recons, x_recons_lengths=x_recons_lengths, vq=vq_output
        )
        # self.timer.stop("total")
        # print("Timer report:", self.timer.synchronize_and_report(), flush=True)
        return output

    def set_train_mode(self, mode: str):
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

    def _train(self, train_mode: str):
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
        """Return a JSON-serializable config describing the model."""
        config = super().get_config()
        encoder = self.encoder.get_config()
        quantizer = self.quantizer.get_config()
        decoder = self.decoder.get_config()
        del encoder["class_name"]
        del decoder["class_name"]
        config.update(
            {
                "encoder": encoder,
                "quantizer": quantizer,
                "decoder": decoder,
                "latent_feats": self.latent_feats,
                "input_sample_freq": self.input_sample_freq,
                "norm_input_loudness": self.norm_input_loudness,
                "target_input_lufs": self.target_input_lufs,
            }
        )
        return config

    @staticmethod
    def filter_args(**kwargs):
        """
        Filter keyword arguments relevant to `DACEncoder.__init__`.

        Returns:
            dict: Filtered kwargs usable to instantiate `DACEncoder`.
        """
        return filter_func_args(DAC.__init__, kwargs)

    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None):
        """Register DAC model arguments on an `ArgumentParser`.

        If `prefix` is provided, a nested sub-parser is created and attached under
        `--{prefix}` via `ActionParser`.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        DACEncoder.add_class_args(parser, prefix="encoder", skip={"out_feats"})
        ResidualVectorQuantizer.add_class_args(
            parser, prefix="quantizer", skip={"in_feats"}
        )
        DACDecoder.add_class_args(parser, prefix="decoder", skip={"in_feats"})
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

    @staticmethod
    def filter_finetune_args(**kwargs):
        """
        Filter keyword arguments relevant to `DAC` finetuning.

        Returns:
            dict: Filtered kwargs usable to finetune `DAC`.
        """
        return filter_func_args(DAC.change_config, kwargs)

    @staticmethod
    def add_finetune_args(parser: ArgumentParser, prefix: Optional[str] = None):
        """Register DAC finetune arguments on an `ArgumentParser`.

        If `prefix` is provided, a nested sub-parser is created and attached under
        `--{prefix}` via `ActionParser`.
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
        ResidualVectorQuantizer.add_class_args(
            parser, prefix="quantizer", skip={"in_feats"}
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


# class CodecMixin:
#     @property
#     def padding(self):
#         if not hasattr(self, "_padding"):
#             self._padding = True
#         return self._padding

#     @padding.setter
#     def padding(self, value):
#         assert isinstance(value, bool)

#         layers = [
#             l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
#         ]

#         for layer in layers:
#             if value:
#                 if hasattr(layer, "original_padding"):
#                     layer.padding = layer.original_padding
#             else:
#                 layer.original_padding = layer.padding
#                 layer.padding = tuple(0 for _ in range(len(layer.padding)))

#         self._padding = value

#     def get_delay(self):
#         # Any number works here, delay is invariant to input length
#         l_out = self.get_output_length(0)
#         L = l_out

#         layers = []
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
#                 layers.append(layer)

#         for layer in reversed(layers):
#             d = layer.dilation[0]
#             k = layer.kernel_size[0]
#             s = layer.stride[0]

#             if isinstance(layer, nn.ConvTranspose1d):
#                 L = ((L - d * (k - 1) - 1) / s) + 1
#             elif isinstance(layer, nn.Conv1d):
#                 L = (L - 1) * s + d * (k - 1) + 1

#             L = math.ceil(L)

#         l_in = L

#         return (l_in - l_out) // 2

#     def get_output_length(self, input_length):
#         L = input_length
#         # Calculate output length
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
#                 d = layer.dilation[0]
#                 k = layer.kernel_size[0]
#                 s = layer.stride[0]

#                 if isinstance(layer, nn.Conv1d):
#                     L = ((L - d * (k - 1) - 1) / s) + 1
#                 elif isinstance(layer, nn.ConvTranspose1d):
#                     L = (L - 1) * s + d * (k - 1) + 1

#                 L = math.floor(L)
#         return L

#     @torch.no_grad()
#     def compress(
#         self,
#         audio_path_or_signal: Union[str, Path, AudioSignal],
#         win_duration: float = 1.0,
#         verbose: bool = False,
#         normalize_db: float = -16,
#         n_quantizers: int = None,
#     ) -> DACFile:
#         """Processes an audio signal from a file or AudioSignal object into
#         discrete codes. This function processes the signal in short windows,
#         using constant GPU memory.

#         Parameters
#         ----------
#         audio_path_or_signal : Union[str, Path, AudioSignal]
#             audio signal to reconstruct
#         win_duration : float, optional
#             window duration in seconds, by default 5.0
#         verbose : bool, optional
#             by default False
#         normalize_db : float, optional
#             normalize db, by default -16

#         Returns
#         -------
#         DACFile
#             Object containing compressed codes and metadata
#             required for decompression
#         """
#         audio_signal = audio_path_or_signal
#         if isinstance(audio_signal, (str, Path)):
#             audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

#         self.eval()
#         original_padding = self.padding
#         original_device = audio_signal.device

#         audio_signal = audio_signal.clone()
#         original_sr = audio_signal.sample_rate

#         resample_fn = audio_signal.resample
#         loudness_fn = audio_signal.loudness

#         # If audio is > 10 minutes long, use the ffmpeg versions
#         if audio_signal.signal_duration >= 10 * 60 * 60:
#             resample_fn = audio_signal.ffmpeg_resample
#             loudness_fn = audio_signal.ffmpeg_loudness

#         original_length = audio_signal.signal_length
#         resample_fn(self.sample_rate)
#         input_db = loudness_fn()

#         if normalize_db is not None:
#             audio_signal.normalize(normalize_db)
#         audio_signal.ensure_max_of_audio()

#         nb, nac, nt = audio_signal.audio_data.shape
#         audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
#         win_duration = (
#             audio_signal.signal_duration if win_duration is None else win_duration
#         )

#         if audio_signal.signal_duration <= win_duration:
#             # Unchunked compression (used if signal length < win duration)
#             self.padding = True
#             n_samples = nt
#             hop = nt
#         else:
#             # Chunked inference
#             self.padding = False
#             # Zero-pad signal on either side by the delay
#             audio_signal.zero_pad(self.delay, self.delay)
#             n_samples = int(win_duration * self.sample_rate)
#             # Round n_samples to nearest hop length multiple
#             n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
#             hop = self.get_output_length(n_samples)

#         codes = []
#         range_fn = range if not verbose else tqdm.trange

#         for i in range_fn(0, nt, hop):
#             x = audio_signal[..., i : i + n_samples]
#             x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))

#             audio_data = x.audio_data.to(self.device)
#             audio_data = self.preprocess(audio_data, self.sample_rate)
#             _, c, _, _, _ = self.encode(audio_data, n_quantizers)
#             codes.append(c.to(original_device))
#             chunk_length = c.shape[-1]

#         codes = torch.cat(codes, dim=-1)

#         dac_file = DACFile(
#             codes=codes,
#             chunk_length=chunk_length,
#             original_length=original_length,
#             input_db=input_db,
#             channels=nac,
#             sample_rate=original_sr,
#             padding=self.padding,
#             dac_version=SUPPORTED_VERSIONS[-1],
#         )

#         if n_quantizers is not None:
#             codes = codes[:, :n_quantizers, :]

#         self.padding = original_padding
#         return dac_file

#     @torch.no_grad()
#     def decompress(
#         self,
#         obj: Union[str, Path, DACFile],
#         verbose: bool = False,
#     ) -> AudioSignal:
#         """Reconstruct audio from a given .dac file

#         Parameters
#         ----------
#         obj : Union[str, Path, DACFile]
#             .dac file location or corresponding DACFile object.
#         verbose : bool, optional
#             Prints progress if True, by default False

#         Returns
#         -------
#         AudioSignal
#             Object with the reconstructed audio
#         """
#         self.eval()
#         if isinstance(obj, (str, Path)):
#             obj = DACFile.load(obj)

#         original_padding = self.padding
#         self.padding = obj.padding

#         range_fn = range if not verbose else tqdm.trange
#         codes = obj.codes
#         original_device = codes.device
#         chunk_length = obj.chunk_length
#         recons = []

#         for i in range_fn(0, codes.shape[-1], chunk_length):
#             c = codes[..., i : i + chunk_length].to(self.device)
#             z = self.quantizer.from_codes(c)[0]
#             r = self.decode(z)
#             recons.append(r.to(original_device))

#         recons = torch.cat(recons, dim=-1)
#         recons = AudioSignal(recons, self.sample_rate)

#         resample_fn = recons.resample
#         loudness_fn = recons.loudness

#         # If audio is > 10 minutes long, use the ffmpeg versions
#         if recons.signal_duration >= 10 * 60 * 60:
#             resample_fn = recons.ffmpeg_resample
#             loudness_fn = recons.ffmpeg_loudness

#         recons.normalize(obj.input_db)
#         resample_fn(obj.sample_rate)
#         recons = recons[..., : obj.original_length]
#         loudness_fn()
#         recons.audio_data = recons.audio_data.reshape(
#             -1, obj.channels, obj.original_length
#         )

#         self.padding = original_padding
#         return recons

# class DAC(BaseModel, CodecMixin):
#     def __init__(
#         self,
#         encoder_dim: int = 64,
#         encoder_rates: List[int] = [2, 4, 8, 8],
#         latent_dim: int = None,
#         decoder_dim: int = 1536,
#         decoder_rates: List[int] = [8, 8, 4, 2],
#         n_codebooks: int = 9,
#         codebook_size: int = 1024,
#         codebook_dim: Union[int, list] = 8,
#         quantizer_dropout: bool = False,
#         sample_rate: int = 44100,
#     ):
#         super().__init__()

#         self.encoder_dim = encoder_dim
#         self.encoder_rates = encoder_rates
#         self.decoder_dim = decoder_dim
#         self.decoder_rates = decoder_rates
#         self.sample_rate = sample_rate

#         if latent_dim is None:
#             latent_dim = encoder_dim * (2 ** len(encoder_rates))

#         self.latent_dim = latent_dim

#         self.hop_length = np.prod(encoder_rates)
#         self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

#         self.n_codebooks = n_codebooks
#         self.codebook_size = codebook_size
#         self.codebook_dim = codebook_dim
#         self.quantizer = ResidualVectorQuantize(
#             input_dim=latent_dim,
#             n_codebooks=n_codebooks,
#             codebook_size=codebook_size,
#             codebook_dim=codebook_dim,
#             quantizer_dropout=quantizer_dropout,
#         )

#         self.decoder = Decoder(
#             latent_dim,
#             decoder_dim,
#             decoder_rates,
#         )
#         self.sample_rate = sample_rate
#         self.apply(init_weights)

#         self.delay = self.get_delay()

#     def preprocess(self, audio_data, sample_rate):
#         if sample_rate is None:
#             sample_rate = self.sample_rate
#         assert sample_rate == self.sample_rate

#         length = audio_data.shape[-1]
#         right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
#         audio_data = nn.functional.pad(audio_data, (0, right_pad))

#         return audio_data

#     def encode(
#         self,
#         audio_data: torch.Tensor,
#         n_quantizers: int = None,
#     ):
#         """Encode given audio data and return quantized latent codes

#         Parameters
#         ----------
#         audio_data : Tensor[B x 1 x T]
#             Audio data to encode
#         n_quantizers : int, optional
#             Number of quantizers to use, by default None
#             If None, all quantizers are used.

#         Returns
#         -------
#         dict
#             A dictionary with the following keys:
#             "z" : Tensor[B x D x T]
#                 Quantized continuous representation of input
#             "codes" : Tensor[B x N x T]
#                 Codebook indices for each codebook
#                 (quantized discrete representation of input)
#             "latents" : Tensor[B x N*D x T]
#                 Projected latents (continuous representation of input before quantization)
#             "vq/commitment_loss" : Tensor[1]
#                 Commitment loss to train encoder to predict vectors closer to codebook
#                 entries
#             "vq/codebook_loss" : Tensor[1]
#                 Codebook loss to update the codebook
#             "length" : int
#                 Number of samples in input audio
#         """
#         z = self.encoder(audio_data)
#         z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
#             z, n_quantizers
#         )
#         return z, codes, latents, commitment_loss, codebook_loss

#     def decode(self, z: torch.Tensor):
#         """Decode given latent codes and return audio data

#         Parameters
#         ----------
#         z : Tensor[B x D x T]
#             Quantized continuous representation of input
#         length : int, optional
#             Number of samples in output audio, by default None

#         Returns
#         -------
#         dict
#             A dictionary with the following keys:
#             "audio" : Tensor[B x 1 x length]
#                 Decoded audio data.
#         """
#         return self.decoder(z)

#     def forward(
#         self,
#         audio_data: torch.Tensor,
#         sample_rate: int = None,
#         n_quantizers: int = None,
#     ):
#         """Model forward pass

#         Parameters
#         ----------
#         audio_data : Tensor[B x 1 x T]
#             Audio data to encode
#         sample_rate : int, optional
#             Sample rate of audio data in Hz, by default None
#             If None, defaults to `self.sample_rate`
#         n_quantizers : int, optional
#             Number of quantizers to use, by default None.
#             If None, all quantizers are used.

#         Returns
#         -------
#         dict
#             A dictionary with the following keys:
#             "z" : Tensor[B x D x T]
#                 Quantized continuous representation of input
#             "codes" : Tensor[B x N x T]
#                 Codebook indices for each codebook
#                 (quantized discrete representation of input)
#             "latents" : Tensor[B x N*D x T]
#                 Projected latents (continuous representation of input before quantization)
#             "vq/commitment_loss" : Tensor[1]
#                 Commitment loss to train encoder to predict vectors closer to codebook
#                 entries
#             "vq/codebook_loss" : Tensor[1]
#                 Codebook loss to update the codebook
#             "length" : int
#                 Number of samples in input audio
#             "audio" : Tensor[B x 1 x length]
#                 Decoded audio data.
#         """
#         length = audio_data.shape[-1]
#         audio_data = self.preprocess(audio_data, sample_rate)
#         z, codes, latents, commitment_loss, codebook_loss = self.encode(
#             audio_data, n_quantizers
#         )

#         x = self.decode(z)
#         return {
#             "audio": x[..., :length],
#             "z": z,
#             "codes": codes,
#             "latents": latents,
#             "vq/commitment_loss": commitment_loss,
#             "vq/codebook_loss": codebook_loss,
#         }
