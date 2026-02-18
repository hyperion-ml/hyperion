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
from ...narchs.audio_feats_mvn import AudioFeatsMVN
from ...narchs.hifi_generator import HiFiGenerator
from ...narchs.nvp_flow import WaveNetNVPFlow as NVPFlow
from ...narchs.wavenet_posterior_encoder import WaveNetPosteriorEncoder
from ...hyper_torch_model import HyperTorchModel
from ...utils.masking import seq_lengths_to_mask
from ...utils.misc import slice_segments


class FreeVCFwdMode(str, Enum):
    RECONS = "recons"
    VC = "vc"
    FEATS_ONLY = "feats-only"

    @staticmethod
    def choices():
        return [o.value for o in FreeVCFwdMode]


class FreeVCTrainMode(str, Enum):
    FULL = "full"
    FROZEN = "frozen"
    HF_FEATS_FROZEN = "hf-feats-frozen"
    HF_FEATS_FROZEN_NOGRAD = "hf-feats-frozen-nograd"

    @staticmethod
    def choices():
        return [o.value for o in FreeVCTrainMode]


@dataclass
class FreeVCOutput(HyperDataClass):
    """
    Output data class for FreeVC model.
    Contains the output tensor and optional metadata.
    """

    gen_audio: torch.Tensor
    z: Optional[torch.Tensor] = None
    z_flow: Optional[torch.Tensor] = None
    prior_z_mean: Optional[torch.Tensor] = None
    prior_z_logs: Optional[torch.Tensor] = None
    post_z_mean: Optional[torch.Tensor] = None
    post_z_logs: Optional[torch.Tensor] = None
    kldiv_loss: Optional[torch.Tensor] = None


class FreeVC(HyperTorchModel):
    """
    FreeVC model for voice conversion.
    """

    def __init__(
        self,
        hf_feats: HyperTorchModel,
        audio_feats: Union[Dict[str, Any], AudioFeatsMVN],
        prior_encoder: Union[Dict[str, Any], WaveNetPosteriorEncoder],
        prior_flow: Union[Dict[str, Any], NVPFlow],
        posterior_encoder: Union[Dict[str, Any], WaveNetPosteriorEncoder],
        decoder: Union[Dict[str, Any], HiFiGenerator],
        internal_feats: int = 192,
        speaker_feats: int = 192,
        l2_norm_speaker: bool = False,
    ):
        super().__init__()
        self.hf_feats = hf_feats
        self._hf_context = contextlib.nullcontext()
        self.internal_feats = internal_feats
        self.speaker_feats = speaker_feats

        if isinstance(audio_feats, dict):
            if "class_name" in audio_feats:
                del audio_feats["class_name"]
            audio_feats = AudioFeatsMVN(**audio_feats)
        else:
            assert isinstance(audio_feats, AudioFeatsMVN)
        self.audio_feats = audio_feats
        self._audio_feats_context = contextlib.nullcontext()

        if isinstance(prior_encoder, dict):
            if "class_name" in prior_encoder:
                del prior_encoder["class_name"]

            if "num_coupling_layers" in prior_encoder:
                prior_encoder = {}

            prior_encoder["in_feats"] = hf_feats.out_feats
            prior_encoder["out_feats"] = internal_feats
            prior_encoder = WaveNetPosteriorEncoder(**prior_encoder)
        elif not isinstance(prior_encoder, WaveNetPosteriorEncoder):
            raise TypeError(
                f"prior_encoder must be a dict or WaveNetPosteriorEncoder, got {type(prior_encoder)}"
            )
        self.prior_encoder = prior_encoder
        if isinstance(prior_flow, dict):
            if "class_name" in prior_flow:
                del prior_flow["class_name"]

            prior_flow["cond_channels"] = speaker_feats
            prior_flow["channels"] = internal_feats
            prior_flow = NVPFlow(**prior_flow)
        elif not isinstance(prior_flow, NVPFlow):
            raise TypeError(
                f"prior_flow must be a dict or NVPFlow, got {type(prior_flow)}"
            )
        self.prior_flow = prior_flow
        if isinstance(posterior_encoder, dict):
            if "class_name" in posterior_encoder:
                del posterior_encoder["class_name"]

            posterior_encoder["in_feats"] = audio_feats.out_feats
            posterior_encoder["cond_channels"] = speaker_feats
            posterior_encoder["out_feats"] = internal_feats
            posterior_encoder = WaveNetPosteriorEncoder(**posterior_encoder)
        elif not isinstance(posterior_encoder, WaveNetPosteriorEncoder):
            raise TypeError(
                f"posterior_encoder must be a dict or WaveNetPosteriorEncoder, got {type(posterior_encoder)}"
            )
        self.posterior_encoder = posterior_encoder
        if isinstance(decoder, dict):
            if "class_name" in decoder:
                del decoder["class_name"]

            decoder["cond_channels"] = speaker_feats
            decoder["in_feats"] = internal_feats
            decoder = HiFiGenerator(**decoder)
        elif not isinstance(decoder, HiFiGenerator):
            raise TypeError(
                f"decoder must be a dict or HiFiGenerator, got {type(decoder)}"
            )
        self.decoder = decoder
        self.l2_norm_speaker = l2_norm_speaker
        assert (
            hf_feats.frame_shift
            == audio_feats.frame_shift * audio_feats.sample_frequency / 1000
        ), "Mismatch in frame shift between hf_feats and audio_feats"

    @property
    def input_sample_frequency(self) -> int:
        """
        Returns the input sample frequency of the model.
        This is typically the sample frequency of the audio features used in the model.
        """
        return self.hf_feats.sample_frequency

    @property
    def output_sample_frequency(self) -> int:
        """
        Returns the output sample frequency of the model.
        This is typically the sample frequency of the audio features used in the model.
        """
        return (
            self.hf_feats.sample_frequency
            * self.decoder.stride
            // self.hf_feats.frame_shift
        )

    @property
    def frame_shift(self) -> int:
        """
        Returns the frame shift of the model.
        This is the number of samples between consecutive frames in the audio features.
        """
        return self.hf_feats.frame_shift

    def max_out_length(self, max_in_length: int) -> int:
        """
        Returns the maximum output length given an input length.
        This is calculated based on the feature extraction process of the model.

        Args:
            max_in_length (int): Maximum input length in samples.

        Returns:
            int: Maximum output length in frames.
        """
        return self.hf_feats.max_out_length(max_in_length) * self.decoder.stride

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns the output lengths given input lengths.
        Args:
            in_lengths (torch.Tensor): Input lengths in samples.

        Returns:
            torch.Tensor: Output lengths in frames.
        """
        return self.hf_feats.out_lengths(in_lengths) * self.decoder.stride

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        B = in_shape[0]
        T = in_shape[1]
        if T is None:
            return (B, 1, None)
        else:
            out_length = self.max_out_length(T)
            return (B, 1, out_length)

    def get_input_context_given_input_length(
        self, input_length: int
    ) -> Tuple[int, int]:
        """Returns the left and right context given an input length.
        Args:
            input_length (int): Length of the input audio in samples.
        Returns:
            Tuple[int, int]: Left and right context in samples.
        """
        max_valid_in_length = (
            self.hf_feats.max_out_length(input_length) * self.hf_feats.frame_shift
        )
        left_context = (input_length - max_valid_in_length) // 2
        right_context = input_length - left_context - max_valid_in_length
        return left_context, right_context

    def get_input_idxs_matching_output(self, input_length: int) -> List[int]:
        """Returns the input indices that match the output length."""
        # max_out_length = self.max_out_length(input_length)
        # max_valid_in_length = max_out_length * self.hf_feats.frame_shift // self.hf_feats.sample_frequency
        max_valid_in_length = (
            self.hf_feats.max_out_length(input_length) * self.hf_feats.frame_shift
        )
        left_context = (input_length - max_valid_in_length) // 2
        return left_context, left_context + max_valid_in_length

    def get_input_matching_output(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ):
        """
        Returns the input audio tensor and lengths that match the output length.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (B, C, T).
            audio_lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the matched audio tensor and lengths.
        """
        start_idx, end_idx = self.get_input_idxs_matching_output(audios.shape[-1])
        audios = audios[:, start_idx:end_idx]
        audio_lengths = audio_lengths - start_idx
        audio_lengths = audio_lengths.clamp(min=0, max=audios.shape[-1])
        return audios[:, start_idx:end_idx], audio_lengths

    def get_target_idxs_matching_output(self, input_length: int) -> List[int]:
        """Returns the target indices that match the output length."""
        # max_out_length = self.max_out_length(input_length)
        # max_valid_in_length = max_out_length * self.hf_feats.frame_shift // self.hf_feats.sample_frequency
        max_valid_target_length = (
            self.hf_feats.max_out_length(input_length) * self.decoder.stride
        )
        left_context = (input_length - max_valid_target_length) // 2
        return left_context, left_context + max_valid_target_length

    def get_target_matching_output(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor, max_input_length: int
    ):
        """
        Returns the input audio tensor and lengths that match the output length.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (B, T).
            audio_lengths (torch.Tensor): Lengths of each sequence in the batch.
            max_input_length (int): Maximum input length in samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the matched audio tensor and lengths.
        """
        start_idx, end_idx = self.get_target_idxs_matching_output(max_input_length)
        audios = audios[:, start_idx:end_idx]
        audio_lengths = audio_lengths - start_idx
        audio_lengths = audio_lengths.clamp(min=0, max=audios.shape[-1])
        return audios, audio_lengths

    def freeze_hf_feats(self):
        self.hf_feats.freeze()

    @staticmethod
    def _kldiv_loss(
        z_flow: torch.Tensor,
        post_z_logs: torch.Tensor,
        prior_z_mean: torch.Tensor,
        prior_z_logs: torch.Tensor,
        z_lengths: Optional[torch.Tensor] = None,
        flow_logdetJ: Optional[torch.Tensor] = None,
    ):
        z_mask = seq_lengths_to_mask(
            z_lengths, z_flow.size(1), time_dim=1, ndim=3, dtype=z_flow.dtype
        )

        with torch.amp.autocast(device_type=z_flow.device.type, enabled=False):
            z_flow = z_flow.float()
            post_z_logs = post_z_logs.float()
            prior_z_mean = prior_z_mean.float()
            prior_z_logs = prior_z_logs.float()
            z_mask = z_mask.float()
            # E[q(z|x)] = - d/2 * (\log(2 pi) +1 )  - \sum_i \log(\sigma_q_i)
            # Ez~q(z) [p(f(z)) + logdet|J|] = - d/2 * log(2 pi) - \sum_i \log(\sigma_p_i) - \sum_i (f(z)_i-mu_p_i)/(2 \sigma_p_i^2) + logdet|J|
            kl_div = prior_z_logs - post_z_logs - 0.5
            kl_div += (
                0.5 * ((z_flow - prior_z_mean) ** 2) * torch.exp(-2.0 * prior_z_logs)
            )

            if z_mask is not None:
                kl_div = torch.sum(kl_div * z_mask) / torch.sum(z_mask)
            else:
                kl_div = torch.mean(kl_div)

            if flow_logdetJ is not None:
                # assert (
                #     flow_logdetJ[0] == 0.0
                # ), f"flow_logdetJ must start with 0.0 {flow_logdetJ}"
                if z_mask is not None:
                    kl_div -= flow_logdetJ.sum() / torch.sum(z_mask)
                else:
                    kl_div -= flow_logdetJ.mean() / z_flow.shape[1]

            return kl_div

    @staticmethod
    def _match_feat_lengths(
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_feat_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Matches the lengths of features and audio features.

        Args:
            feats (torch.Tensor): Input features of shape (B, C, T).
            feat_lengths (torch.Tensor): Lengths of each sequence in the batch.
            audio_feats (torch.Tensor): Audio features of shape (B, C, T).
            audio_feat_lengths (torch.Tensor): Lengths of each audio feature sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Matched features and their lengths.
        """
        delta_length = feats.shape[1] - audio_feats.shape[1]
        assert (
            abs(delta_length) <= 1
        ), f"Audio features and feats must have similar lengths, got delta_length={delta_length}"
        if delta_length > 0:
            logging.warning("Audio features padded to match feats length")
            audio_feats = torch.nn.functional.pad(
                audio_feats,
                (0, 0, 0, -delta_length),
                mode="replicate",
            )
            audio_feat_lengths = feat_lengths
        elif delta_length < 0:
            audio_feats = audio_feats[:, : feats.shape[1]]
            audio_feat_lengths = feat_lengths

        return feats, feat_lengths, audio_feats, audio_feat_lengths

    def forward_hf_feats(self, x, x_lengths, chunk_length=0, detach_chunks=False):

        with self._hf_context:
            assert not torch.is_grad_enabled(), "Not in no_grad context!"
            hf_output = self.hf_feats(
                x,
                x_lengths,
                chunk_length=chunk_length,
                detach_chunks=detach_chunks,
            )

        feats = hf_output["last_hidden_state"]
        feat_lengths = hf_output["hidden_states_lengths"]
        return feats, feat_lengths

    def forward_audio_feats(self, x, x_lengths):
        with self._audio_feats_context:
            assert not torch.is_grad_enabled(), "Not in no_grad context!"
            feats, feat_lengths = self.audio_feats(x, x_lengths)
        return feats, feat_lengths

    def forward(
        self,
        source_audios: torch.Tensor,
        source_audio_lengths: torch.Tensor,
        speaker_feats: torch.Tensor,
        mode: FreeVCFwdMode = FreeVCFwdMode.RECONS,
        slice_start_idxs: Optional[torch.Tensor] = None,
        slice_segment_length: Optional[int] = None,
        feats: Optional[torch.Tensor] = None,
        feat_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the FreeVC model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            x_lengths (torch.Tensor): Lengths of each sequence in the batch.
            mode (FreeVCMode): Mode of operation (recons or vc).

        Returns:
            FreeVCOutput: Output containing the processed tensor and metadata.
        """
        if feats is None:
            feats, feat_lengths = self.forward_hf_feats(
                source_audios, source_audio_lengths
            )
            if mode == FreeVCFwdMode.FEATS_ONLY:
                # Return only the features without further processing
                return feats, feat_lengths

        if speaker_feats.dim() == 2:
            speaker_feats = speaker_feats.unsqueeze(1)

        if self.l2_norm_speaker:
            speaker_feats = nn.functional.normalize(
                speaker_feats, p=2, dim=-1, eps=1e-12
            )

        if mode == FreeVCFwdMode.RECONS:
            output = self._forward_recons(
                source_audios,
                source_audio_lengths,
                feats,
                feat_lengths,
                speaker_feats=speaker_feats,
                slice_start_idxs=slice_start_idxs,
                slice_segment_length=slice_segment_length,
            )
        elif mode == FreeVCFwdMode.VC:
            output = self._forward_vc(
                feats,
                feat_lengths,
                speaker_feats=speaker_feats,
                slice_start_idxs=slice_start_idxs,
                slice_segment_length=slice_segment_length,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return output

    def _forward_recons(
        self,
        source_audios: torch.Tensor,
        source_audio_lengths: torch.Tensor,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        speaker_feats: torch.Tensor,
        slice_start_idxs: Optional[torch.Tensor] = None,
        slice_segment_length: Optional[int] = None,
    ):
        """
        Forward pass for reconstruction mode.

        Args:
            feats (torch.Tensor): Input features of shape (B, C, T).
            feat_lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Reconstructed output tensor.
        """
        audio_feats, audio_feat_lengths = self.forward_audio_feats(
            source_audios, source_audio_lengths
        )
        feats, feat_lengths, audio_feats, audio_feat_lengths = self._match_feat_lengths(
            feats, feat_lengths, audio_feats, audio_feat_lengths
        )
        _, prior_z_mean, prior_z_logs = self.prior_encoder(feats, feat_lengths)
        post_z, post_z_mean, post_z_logs = self.posterior_encoder(
            audio_feats, audio_feat_lengths, condition=speaker_feats
        )
        post_z_flow, flow_logdetJ = self.prior_flow(
            post_z, feat_lengths, condition=speaker_feats
        )

        # print(
        #     "shapes",
        #     source_audios.shape,
        #     source_audio_lengths.max(),
        #     audio_feats.shape,
        #     audio_feat_lengths.max(),
        #     feats.shape,
        #     feat_lengths.max(),
        #     post_z.shape,
        #     post_z_mean.shape,
        #     post_z_logs.shape,
        #     post_z_flow.shape,
        #     flow_logdetJ.shape,
        #     prior_z_mean.shape,
        #     prior_z_logs.shape,
        # )
        kldiv_loss = self._kldiv_loss(
            post_z_flow,
            post_z_logs,
            prior_z_mean,
            prior_z_logs,
            z_lengths=feat_lengths,
            flow_logdetJ=flow_logdetJ,
        )
        if slice_start_idxs is not None:
            time_scale = 1 / self.hf_feats.frame_shift
            post_z_slice = slice_segments(
                post_z,
                (time_scale * slice_start_idxs).to(torch.long),
                int(time_scale * slice_segment_length),
                dim=1,
                permissive=1,
            )
        else:
            post_z_slice = post_z

        gen_audio = self.decoder(post_z_slice, condition=speaker_feats)
        output = FreeVCOutput(
            gen_audio=gen_audio,
            z=post_z,
            z_flow=post_z_flow,
            prior_z_mean=prior_z_mean,
            prior_z_logs=prior_z_logs,
            post_z_mean=post_z_mean,
            post_z_logs=post_z_logs,
            kldiv_loss=kldiv_loss,
        )
        return output

    def _forward_vc(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        speaker_feats: torch.Tensor,
        slice_start_idxs: Optional[torch.Tensor] = None,
        slice_segment_length: Optional[int] = None,
    ):
        """
        Forward pass for voice conversion mode.

        Args:
            feats (torch.Tensor): Input features of shape (B, C, T).
            feat_lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Converted output tensor.
        """
        prior_z, prior_z_mean, prior_z_logs = self.prior_encoder(feats, feat_lengths)
        z, _ = self.prior_flow(
            prior_z, feat_lengths, condition=speaker_feats, reverse=True
        )
        if slice_start_idxs is not None:
            time_scale = 1 / self.hf_feats.frame_shift
            z_slice = slice_segments(
                z,
                (time_scale * slice_start_idxs).to(torch.long),
                int(time_scale * slice_segment_length),
                dim=1,
                permissive=1,
            )
            z_slice_lengths = None
        else:
            z_slice = z
            z_slice_lengths = feat_lengths
        gen_audio = self.decoder(
            z_slice, x_lengths=z_slice_lengths, condition=speaker_feats
        )
        output = FreeVCOutput(
            gen_audio=gen_audio,
            z=z,
            z_flow=prior_z,
            prior_z_mean=prior_z_mean,
            prior_z_logs=prior_z_logs,
        )
        return output

    def set_train_mode(self, mode: str):
        if mode == self._train_mode:
            return
        logging.info("setting FreeVC train mode to %s", mode)
        if mode == FreeVCTrainMode.FULL:
            self.unfreeze()
        elif mode == FreeVCTrainMode.FROZEN:
            self.freeze()
        elif mode in [
            FreeVCTrainMode.HF_FEATS_FROZEN,
            FreeVCTrainMode.HF_FEATS_FROZEN_NOGRAD,
        ]:
            self.unfreeze()
            self.freeze_hf_feats()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        if mode in [FreeVCTrainMode.HF_FEATS_FROZEN_NOGRAD]:
            logging.info("using torch.no_grad for hf_feats")
            self._hf_context = torch.no_grad()

        logging.info("using torch.no_grad for audio feats")
        self._audio_feats_context = torch.no_grad()
        self.audio_feats.freeze()
        self._train_mode = mode

    def _train(self, train_mode: str):
        if train_mode in [FreeVCTrainMode.FULL, FreeVCTrainMode.FROZEN]:
            super()._train(train_mode)
        elif train_mode in [
            FreeVCTrainMode.HF_FEATS_FROZEN,
            FreeVCTrainMode.HF_FEATS_FROZEN_NOGRAD,
        ]:
            super()._train(FreeVCTrainMode.FULL)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the FreeVC model.
        This includes the configurations of all submodules.
        """
        config = super().get_config()
        hf_feats = self.hf_feats.get_config()
        audio_feats = self.audio_feats.get_config()
        prior_encoder = self.prior_encoder.get_config()
        prior_flow = self.prior_flow.get_config()
        posterior_encoder = self.posterior_encoder.get_config()
        decoder = self.decoder.get_config()
        del hf_feats["class_name"]
        del audio_feats["class_name"]
        del prior_encoder["class_name"]
        del prior_flow["class_name"]
        del decoder["class_name"]
        config.update(
            {
                "hf_feats": hf_feats,
                "audio_feats": audio_feats,
                "prior_encoder": prior_encoder,
                "prior_flow": prior_flow,
                "posterior_encoder": posterior_encoder,
                "decoder": decoder,
                "internal_feats": self.internal_feats,
                "speaker_feats": self.speaker_feats,
                "l2_norm_speaker": self.l2_norm_speaker,
            }
        )
        return config

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ):
        """
        Adds FreeVC class arguments to an ArgumentParser.

        Args:
            parser (ArgumentParser): The parser to which the arguments will be added.
            prefix (Optional[str]): Optional prefix for argument names.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AudioFeatsMVN.add_class_args(parser, prefix="audio_feats")
        WaveNetPosteriorEncoder.add_class_args(
            parser, prefix="prior_encoder", skip={"in_feats", "cond_channels"}
        )
        NVPFlow.add_class_args(
            parser, prefix="prior_flow", skip={"cond_channels", "channels"}
        )
        WaveNetPosteriorEncoder.add_class_args(
            parser, prefix="posterior_encoder", skip={"in_feats", "cond_channels"}
        )
        HiFiGenerator.add_class_args(
            parser, prefix="decoder", skip={"cond_channels", "in_feats"}
        )
        parser.add_argument(
            "--internal-feats",
            type=int,
            default=192,
            help="Number of internal features in the model.",
        )
        parser.add_argument(
            "--speaker-feats",
            type=int,
            default=192,
            help="Number of speaker features in the model.",
        )
        parser.add_argument(
            "--l2-norm-speaker",
            action=ActionYesNo,
            default=False,
            help="Whether to apply L2 normalization to speaker features.",
        )
        # parser.add_argument(
        #     "--output-sample-frequency",
        #     type=int,
        #     default=16000,
        #     help="Output sample frequency of the model.",
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


# class SynthesizerTrn(nn.Module):
#     """
#     Synthesizer for Training
#     """

#     def __init__(
#         self,
#         spec_channels,
#         segment_size,
#         inter_channels,
#         hidden_channels,
#         filter_channels,
#         n_heads,
#         n_layers,
#         kernel_size,
#         p_dropout,
#         resblock,
#         resblock_kernel_sizes,
#         resblock_dilation_sizes,
#         upsample_rates,
#         upsample_initial_channel,
#         upsample_kernel_sizes,
#         gin_channels,
#         ssl_dim,
#         use_spk,
#         **kwargs,
#     ):

#         super().__init__()
#         self.spec_channels = spec_channels
#         self.inter_channels = inter_channels
#         self.hidden_channels = hidden_channels
#         self.filter_channels = filter_channels
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.kernel_size = kernel_size
#         self.p_dropout = p_dropout
#         self.resblock = resblock
#         self.resblock_kernel_sizes = resblock_kernel_sizes
#         self.resblock_dilation_sizes = resblock_dilation_sizes
#         self.upsample_rates = upsample_rates
#         self.upsample_initial_channel = upsample_initial_channel
#         self.upsample_kernel_sizes = upsample_kernel_sizes
#         self.segment_size = segment_size
#         self.gin_channels = gin_channels
#         self.ssl_dim = ssl_dim
#         self.use_spk = use_spk

#         self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
#         self.dec = Generator(
#             inter_channels,
#             resblock,
#             resblock_kernel_sizes,
#             resblock_dilation_sizes,
#             upsample_rates,
#             upsample_initial_channel,
#             upsample_kernel_sizes,
#             gin_channels=gin_channels,
#         )
#         self.enc_q = Encoder(
#             spec_channels,
#             inter_channels,
#             hidden_channels,
#             5,
#             1,
#             16,
#             gin_channels=gin_channels,
#         )
#         self.flow = ResidualCouplingBlock(
#             inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
#         )

#         if not self.use_spk:
#             self.enc_spk = SpeakerEncoder(
#                 model_hidden_size=gin_channels, model_embedding_size=gin_channels
#             )

#     def feat_lengthsself, c, spec, g=None, mel=None, c_lengths=None, spec_lengths=None):
#         if c_lengths == None:
#             c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
#         if spec_lengths == None:
#             spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

#         if not self.use_spk:
#             g = self.enc_spk(mel.transpose(1, 2))
#         g = g.unsqueeze(-1)

#         _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
#         z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
#         z_p = self.flow(z, spec_mask, g=g)

#         z_slice, ids_slice = commons.rand_slice_segments(
#             z, spec_lengths, self.segment_size
#         )
#         o = self.dec(z_slice, g=g)

#         return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

#     def infer(self, c, g=None, mel=None, c_lengths=None):
#         if c_lengths == None:
#             c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
#         if not self.use_spk:
#             g = self.enc_spk.embed_utterance(mel.transpose(1, 2))
#         g = g.unsqueeze(-1)

#         z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
#         z = self.flow(z_p, c_mask, g=g, reverse=True)
#         o = self.dec(z * c_mask, g=g)

#         return o
