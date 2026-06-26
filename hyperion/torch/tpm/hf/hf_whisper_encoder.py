"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from ....utils.misc import filter_func_args
from ...utils.ddp import ddp_get_rank, ddp_wait_for_all_procs
from .hf_wav2vec_base import HFWav2VecBase


class WhisperEncoderFrontend(nn.Module):
    """Torch implementation of the Hugging Face Whisper audio frontend.

    This module reproduces the Whisper mel feature extraction path in torch.

    Attributes:
        sampling_rate (`int`): expected waveform sampling rate.
        num_mel_bins (`int`): number of mel filters.
        n_fft (`int`): FFT size used by Whisper.
        hop_length (`int`): STFT hop length in samples.
        chunk_length (`int`): padded/truncated input length in seconds.
        dither (`float`): optional waveform dither standard deviation.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
        dither: float = 0.0,
    ) -> None:
        """Initializes the Whisper frontend.

        Args:
            sampling_rate: Expected waveform sampling rate.
            num_mel_bins: Number of mel filters.
            n_fft: FFT size.
            hop_length: STFT hop size in samples.
            chunk_length: Padded/truncated input duration in seconds.
            dither: Optional waveform dither standard deviation.

        Returns:
            None.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.dither = dither

        self.register_buffer("window", torch.hann_window(n_fft))
        mel_filters = self._make_slaney_mel_filters(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=num_mel_bins,
            sampling_rate=sampling_rate,
        )
        self.register_buffer(
            "mel_filters",
            torch.tensor(mel_filters, dtype=torch.get_default_dtype()),
        )

    @staticmethod
    def _hertz_to_mel(freq: np.ndarray) -> np.ndarray:
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        mels = 3.0 * freq / 200.0
        log_region = freq >= min_log_hertz
        mels[log_region] = (
            min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
        )
        return mels

    @staticmethod
    def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        freq = 200.0 * mels / 3.0
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(
            logstep * (mels[log_region] - min_log_mel)
        )
        return freq

    @classmethod
    def _make_slaney_mel_filters(
        cls,
        num_frequency_bins: int,
        num_mel_filters: int,
        sampling_rate: int,
    ) -> np.ndarray:
        mel_min = cls._hertz_to_mel(np.array([0.0]))[0]
        mel_max = cls._hertz_to_mel(np.array([8000.0]))[0]
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        filter_freqs = cls._mel_to_hertz(mel_freqs)
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

        filter_diff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
        enorm = 2.0 / (
            filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters]
        )
        mel_filters *= np.expand_dims(enorm, 0)
        return mel_filters

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Computes encoder output lengths from waveform lengths.

        Args:
            in_lengths: Waveform lengths in samples.

        Returns:
            Valid hidden-state lengths after Whisper's stride-2 encoder frontend.
        """
        feature_lengths = torch.div(
            in_lengths.clamp(max=self.n_samples), self.hop_length, rounding_mode="floor"
        )
        return torch.div(feature_lengths + 1, 2, rounding_mode="floor")

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts Whisper log-mel features.

        Args:
            x: Waveform tensor with shape `(batch, num_samples)`.
            x_lengths: Optional waveform lengths in samples.

        Returns:
            Tuple with features shaped `(batch, num_mel_bins, num_frames)` and valid output lengths.
        """
        if x.dim() != 2:
            raise ValueError(f"expected 2-D waveform tensor, got shape {x.shape}")

        batch_size, num_samples = x.shape
        device = x.device
        if x_lengths is None:
            x_lengths = torch.full(
                (batch_size,), num_samples, dtype=torch.long, device=device
            )

        if num_samples > self.n_samples:
            x = x[:, : self.n_samples]
        elif num_samples < self.n_samples:
            x = torch.nn.functional.pad(x, (0, self.n_samples - num_samples))

        if self.dither != 0.0:
            x = x + self.dither * torch.randn_like(x)

        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=self.window.to(dtype=x.dtype),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().pow(2)
        mel_filters = self.mel_filters.to(dtype=magnitudes.dtype)
        mel_spec = torch.matmul(mel_filters.t(), magnitudes)

        log_spec = mel_spec.clamp(min=1e-10).log10()
        max_val = log_spec.flatten(1).max(dim=1).values[:, None, None]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec, self.out_lengths(x_lengths)


class HFWhisperEncoder(HFWav2VecBase):
    r"""This is wrapper over HuggingFace Whisper encoder model.
        See documentation: https://huggingface.co/docs/transformers/model_doc/whisper

        This wrapper makes the HuggingFace encoder to have the same interface
        as other hyperion models. It also adds extra functionalities.

        The config. parameters are the same as in the HuggingFace WhisperConfig class.

    Attributes:
        pretrained_model_path (`str`, defaults to None): file path or HuggingFace Hub path to
            pre-trained model.
        normalize_input (`bool`, defaults to True): whether or not to zero-mean unit-variance
            normalize the input.
        use_input_attention_mask (`bool`, defaults to False): kept for interface compatibility.
            Whisper encoder ignores attention masks.
        vocab_size (`int`, defaults to 51865): vocabulary size of the
            model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method.
        num_mel_bins (`int`, defaults to 80): number of log-mel bins extracted by the frontend.
        encoder_layers (`int`, defaults to 4): number of encoder layers.
        encoder_attention_heads (`int`, defaults to 6): number of attention heads in the encoder.
        decoder_layers (`int`, defaults to 4): number of decoder layers in the Whisper config.
        decoder_attention_heads (`int`, defaults to 6): number of decoder attention heads.
        decoder_ffn_dim (`int`, defaults to 1536): dimensionality of the decoder feed-forward layer.
        encoder_ffn_dim (`int`, defaults to 1536): dimensionality of the encoder feed-forward layer.
        encoder_layerdrop (`float`, defaults to 0.0): probability of dropping an encoder layer.
        decoder_layerdrop (`float`, defaults to 0.0): probability of dropping a decoder layer.
        decoder_start_token_id (`int`, defaults to 50257): decoder start token id.
        use_cache (`bool`, defaults to True): whether to use the decoder cache in the HF config.
        activation_function (`str` or `function`, defaults to `"gelu"`): encoder activation function.
        d_model (`int`, defaults to 384): dimensionality of the encoder layers.
        dropout (`float`, defaults to 0.0): dropout probability used by the encoder.
        attention_dropout (`float`, defaults to 0.0): attention dropout probability.
        activation_dropout (`float`, defaults to 0.0): activation dropout probability.
        init_std (`float`, defaults to 0.02): standard deviation used for weight initialization.
        scale_embedding (`bool`, defaults to False): whether to scale token embeddings.
        max_source_positions (`int`, defaults to 1500): maximum number of encoder positions.
        max_target_positions (`int`, defaults to 448): maximum number of decoder positions.
        pad_token_id (`int`, defaults to 50256): pad token id.
        bos_token_id (`int`, defaults to 50256): beginning-of-sentence token id.
        eos_token_id (`int`, defaults to 50256): end-of-sentence token id.
        apply_spec_augment (`bool`, defaults to False): whether to apply SpecAugment.
        mask_time_prob (`float`, defaults to 0.05): time masking probability.
        mask_time_length (`int`, defaults to 10): time masking span length.
        mask_time_min_masks (`int`, defaults to 2): minimum number of time masks.
        mask_feature_prob (`float`, defaults to 0.0): feature masking probability.
        mask_feature_length (`int`, defaults to 10): feature masking span length.
        mask_feature_min_masks (`int`, defaults to 0): minimum number of feature masks.
        median_filter_width (`int`, defaults to 7): median filter width used by Whisper config.
        hop_length (`int`, defaults to 160): frontend hop length in samples.
        chunk_length (`int`, defaults to 30): frontend chunk length in seconds.
        n_fft (`int`, defaults to 400): frontend FFT size.
        dither (`float`, defaults to 0.0): frontend waveform dither.
        cache_dir (`Union[str, os.PathLike]`, defaults to `"./.cache/hyperion_hf"`): path to a directory in which a downloaded pretrained
            model configuration should be cached if the standard cache should not be used.
        force_download (`bool`, defaults to `False`): whether or not to force the (re-)download
            the model weights and configuration files and override the
            cached versions if they exist.
        resume_download (`bool`, defaults to `False`): whether or not to delete incompletely
            received files. Will attempt to resume the download if such a file exists.
        revision (`str`, defaults to `"main"`): the specific model version to use.
            It can be a branch name, a tag name, or a commit id.
        ignore_pretrained (`bool`, defaults to `False`): if True, it ignores the pretrained_model_path
            and initializes the model from the configuration. This is set to True for models that have
            already been fine-tuned.
        override_dropouts (`bool`, defaults to `False`): if True, it ignores the dropout probs. in the pretrained model
            and uses the ones passed as arguments.
        override_spec_augment (`bool`, defaults to `False`): if True, it ignores the spec. augment.
            configuration in the pretrained model and uses the ones passed in the arguments.
        left_encoder_context (`int`, defaults to `0`): past context frames used by the transformer encoder when the signal is evaluated
            chunk by chunk, if it is too long to fit in GPU.
        right_encoder_context (`int`, defaults to `0`): future context frames used by the transformer encoder.
        sample_frequency (`int`, defaults to `16000`): waveform sample frequency used to train the model.
        feat_extract_lr (`Optional[float]`, defaults to `None`): learning rate for the Whisper feature encoder, serves to set a lr different than the global one.
        encoder_lr (`Optional[float]`, defaults to `None`): learning rate for the Whisper encoder, serves to set a lr different than the global one.
        use_lora (`bool`, defaults to `False`): use low-rank adapters
        lora_components (`List[str]`, defaults to `["q_proj", "v_proj"]`): list of components where we apply LoRA, e.g., [Wq, Wv]
        lora_rank (`int`, defaults to `4`): rank of LoRA
        lora_alpha (`int`, defaults to `8`): scale for LoRA
        lora_dropout (`float`, defaults to `0.0`): dropout rate for LoRA
        lora_merge_weights (`bool`, defaults to `True`): lora weights are merged with the pretrained weights at inference.
        bias_weight_decay (`Optional[float]`, defaults to `None`): weight decay for bias parameters, if not None overrides global weight decay
    """

    def __init__(
        self,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None,
        normalize_input: bool = True,
        use_input_attention_mask: bool = False,
        vocab_size: int = 51865,
        num_mel_bins: int = 80,
        encoder_layers: int = 4,
        encoder_attention_heads: int = 6,
        decoder_layers: int = 4,
        decoder_attention_heads: int = 6,
        decoder_ffn_dim: int = 1536,
        encoder_ffn_dim: int = 1536,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        decoder_start_token_id: int = 50257,
        use_cache: bool = True,
        activation_function: Union[str, Callable] = "gelu",
        d_model: int = 384,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        scale_embedding: bool = False,
        max_source_positions: int = 1500,
        max_target_positions: int = 448,
        pad_token_id: Optional[int] = 50256,
        bos_token_id: Optional[int] = 50256,
        eos_token_id: Optional[int] = 50256,
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        median_filter_width: int = 7,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        dither: float = 0.0,
        cache_dir: Union[str, os.PathLike] = "./.cache/hyperion_hf",
        force_download: bool = False,
        resume_download: bool = False,
        revision: str = "main",
        drop_layers_gt: Optional[int] = None,
        ignore_pretrained: bool = False,
        override_dropouts: bool = False,
        override_spec_augment: bool = False,
        left_encoder_context: int = 0,
        right_encoder_context: int = 0,
        sample_frequency: int = 16000,
        feat_extract_lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        use_lora: bool = False,
        lora_components: List[str] = ["q_proj", "v_proj"],
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initializes the Hugging Face Whisper encoder wrapper.

        Args:
            pretrained_model_path: Local path or HF Hub model id.
            normalize_input: If True, kept for interface compatibility.
            use_input_attention_mask: If True, kept for interface compatibility.
            vocab_size: Vocabulary size.
            num_mel_bins: Number of mel bins.
            encoder_layers: Number of encoder layers.
            encoder_attention_heads: Number of encoder attention heads.
            decoder_layers: Number of decoder layers.
            decoder_attention_heads: Number of decoder attention heads.
            decoder_ffn_dim: Decoder feed-forward size.
            encoder_ffn_dim: Encoder feed-forward size.
            encoder_layerdrop: Encoder layerdrop probability.
            decoder_layerdrop: Decoder layerdrop probability.
            decoder_start_token_id: Decoder start token id.
            use_cache: Whether to use the decoder cache in the HF config.
            activation_function: Activation function used by the encoder.
            d_model: Encoder hidden size.
            dropout: Encoder dropout probability.
            attention_dropout: Attention dropout probability.
            activation_dropout: Activation dropout probability.
            init_std: Weight init standard deviation.
            scale_embedding: Whether to scale embeddings.
            max_source_positions: Maximum encoder positions.
            max_target_positions: Maximum decoder positions.
            pad_token_id: Pad token id.
            bos_token_id: BOS token id.
            eos_token_id: EOS token id.
            apply_spec_augment: Whether to apply SpecAugment.
            mask_time_prob: Time masking probability.
            mask_time_length: Time mask length.
            mask_time_min_masks: Minimum number of time masks.
            mask_feature_prob: Feature masking probability.
            mask_feature_length: Feature mask length.
            mask_feature_min_masks: Minimum number of feature masks.
            median_filter_width: Median filter width.
            hop_length: Frontend hop length in samples.
            chunk_length: Frontend chunk length in seconds.
            n_fft: Frontend FFT size.
            dither: Frontend dither.
            cache_dir: HF cache directory.
            force_download: If True, forces HF re-download.
            resume_download: Deprecated argument, kept for compatibility.
            revision: HF model revision.
            drop_layers_gt: Drops encoder layers above this index when set.
            ignore_pretrained: If True, builds model from config only.
            override_dropouts: Whether to override dropout settings.
            override_spec_augment: Whether to override spec-augment settings.
            left_encoder_context: Left chunked-inference context in frames.
            right_encoder_context: Right chunked-inference context in frames.
            sample_frequency: Expected waveform sample frequency.
            feat_extract_lr: Optional LR override for feature extractor parameters.
            encoder_lr: Optional LR override for encoder parameters.
            use_lora: Whether to enable LoRA adapters.
            lora_components: Target module names for LoRA.
            lora_rank: LoRA rank.
            lora_alpha: LoRA scale.
            lora_dropout: LoRA dropout.
            lora_merge_weights: Whether to merge LoRA weights at inference.
            bias_weight_decay: Optional separate weight decay for bias params.

        Returns:
            None.
        """
        super().__init__(
            pretrained_model_path=pretrained_model_path,
            normalize_input=normalize_input,
            use_input_attention_mask=use_input_attention_mask,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            revision=revision,
            drop_layers_gt=drop_layers_gt,
            ignore_pretrained=True,
            override_dropouts=override_dropouts,
            override_spec_augment=override_spec_augment,
            left_encoder_context=left_encoder_context,
            right_encoder_context=right_encoder_context,
            sample_frequency=sample_frequency,
            feat_extract_lr=feat_extract_lr,
            encoder_lr=encoder_lr,
            use_lora=use_lora,
            lora_components=lora_components,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_merge_weights=lora_merge_weights,
            bias_weight_decay=bias_weight_decay,
        )
        self.ignore_pretrained = ignore_pretrained
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.dither = dither

        if pretrained_model_path is not None and not ignore_pretrained:
            rank = ddp_get_rank()
            if rank == 0:
                feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                )
                logging.info(
                    "Downloading HF Whisper model from %s", pretrained_model_path
                )
                model = WhisperModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                )
            ddp_wait_for_all_procs()
            if rank > 0:
                feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    revision=revision,
                )
                model = WhisperModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    revision=revision,
                )
            ddp_wait_for_all_procs()
            self.hf_model = model.encoder
            self.sample_frequency = feature_extractor.sampling_rate
            self.hop_length = feature_extractor.hop_length
            self.chunk_length = feature_extractor.chunk_length
            self.n_fft = feature_extractor.n_fft
            self.dither = feature_extractor.dither
            self.change_config(
                override_dropouts=self.override_dropouts,
                override_spec_augment=self.override_spec_augment,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                apply_spec_augment=apply_spec_augment,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
            )
        else:
            config = WhisperConfig(
                vocab_size=vocab_size,
                num_mel_bins=num_mel_bins,
                encoder_layers=encoder_layers,
                encoder_attention_heads=encoder_attention_heads,
                decoder_layers=decoder_layers,
                decoder_attention_heads=decoder_attention_heads,
                decoder_ffn_dim=decoder_ffn_dim,
                encoder_ffn_dim=encoder_ffn_dim,
                encoder_layerdrop=encoder_layerdrop,
                decoder_layerdrop=decoder_layerdrop,
                decoder_start_token_id=decoder_start_token_id,
                use_cache=use_cache,
                activation_function=activation_function,
                d_model=d_model,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                init_std=init_std,
                scale_embedding=scale_embedding,
                max_source_positions=max_source_positions,
                max_target_positions=max_target_positions,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                apply_spec_augment=apply_spec_augment,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
                median_filter_width=median_filter_width,
            )
            self.hf_model = WhisperEncoder(config)

        self.frontend = WhisperEncoderFrontend(
            sampling_rate=self.sample_frequency,
            num_mel_bins=self.hf_config.num_mel_bins,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            chunk_length=self.chunk_length,
            dither=self.dither,
        )

        if drop_layers_gt is not None:
            self.drop_upper_layers(drop_layers_gt)

        if use_lora:
            self._make_lora_layers(lora_components, lora_rank, lora_alpha, lora_dropout)

        self.ignore_pretrained = True

    @property
    def num_encoder_layers(self) -> int:
        """Returns the number of transformer encoder layers.

        Args:
            None.

        Returns:
            Number of encoder layers.
        """
        return self.hf_config.encoder_layers

    @property
    def hidden_size(self) -> int:
        """Returns the encoder hidden dimension.

        Args:
            None.

        Returns:
            Hidden size.
        """
        return self.hf_config.d_model

    @property
    def feature_encoder_context(self) -> Tuple[int, int]:
        """Returns Whisper frontend context in waveform samples.

        Args:
            None.

        Returns:
            A tuple `(left_context, right_context)` in input samples.
        """
        return self.n_fft // 2, self.n_fft // 2

    @property
    def frame_shift(self) -> int:
        """Returns encoder hidden-state shift in waveform samples.

        Args:
            None.

        Returns:
            Effective frame shift in samples.
        """
        return self.hop_length * 2

    def max_out_length(self, max_in_length: int) -> int:
        """Computes maximum output length from waveform length.

        Args:
            max_in_length: Input length in waveform samples.

        Returns:
            Maximum encoded sequence length in frames.
        """
        return self.out_lengths(torch.tensor(max_in_length, dtype=torch.long)).item()

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Computes output lengths from waveform lengths.

        Args:
            in_lengths: Input lengths in waveform samples.

        Returns:
            Output lengths in encoder frames.
        """
        return self.frontend.out_lengths(in_lengths)

    def out_shape(self, in_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        """Computes output tensor shape from input batch/length shape.

        Args:
            in_shape: Input shape tuple `(batch_size, num_samples)`.

        Returns:
            Output shape tuple `(batch_size, num_frames, hidden_size)`.
        """
        return (
            in_shape[0],
            self.hf_config.max_source_positions,
            self.hf_config.d_model,
        )

    @property
    def out_feats(self) -> int:
        """Number of output features of the model."""
        return self.hf_config.d_model

    def change_dropouts(
        self,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Updates dropout values in config and instantiated HF modules.

        Args:
            dropout: Encoder dropout probability.
            attention_dropout: Attention dropout probability.
            activation_dropout: Activation dropout probability.
            **kwargs: Extra unused keyword arguments.

        Returns:
            None.
        """
        self.hf_model.config.dropout = dropout
        self.hf_model.config.attention_dropout = attention_dropout
        self.hf_model.config.activation_dropout = activation_dropout

        self.hf_model.dropout = dropout
        for layer in self.hf_model.layers:
            layer.dropout = dropout
            layer.activation_dropout = activation_dropout
            layer.self_attn.dropout = attention_dropout

    def change_spec_augment(
        self,
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        **kwargs: Any,
    ) -> None:
        """Sets SpecAugment-related fields in the HF config.

        Args:
            apply_spec_augment: Enables or disables SpecAugment.
            mask_time_prob: Time-mask sampling probability.
            mask_time_length: Time-mask length in frames.
            mask_time_min_masks: Minimum number of time masks.
            mask_feature_prob: Feature-mask sampling probability.
            mask_feature_length: Feature-mask length in channels.
            mask_feature_min_masks: Minimum number of feature masks.
            **kwargs: Unused extra keyword arguments.

        Returns:
            None.
        """
        self.hf_model.config.apply_spec_augment = apply_spec_augment
        self.hf_model.config.mask_time_prob = mask_time_prob
        self.hf_model.config.mask_time_length = mask_time_length
        self.hf_model.config.mask_time_min_masks = mask_time_min_masks
        self.hf_model.config.mask_feature_prob = mask_feature_prob
        self.hf_model.config.mask_feature_length = mask_feature_length
        self.hf_model.config.mask_feature_min_masks = mask_feature_min_masks

    def drop_upper_layers(self, max_layers: int) -> None:
        """Drops encoder layers above `max_layers`.

        Args:
            max_layers: Number of lower encoder layers to keep.

        Returns:
            None.
        """
        if max_layers >= self.hf_config.encoder_layers:
            return
        self.hf_model.layers = nn.ModuleList(
            [layer for i, layer in enumerate(self.hf_model.layers) if i < max_layers]
        )
        self.hf_config.encoder_layers = max_layers

    def _preprocess(
        self, x: torch.Tensor, x_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts waveforms to fixed-length Whisper log-mel features.

        Args:
            x: Waveform tensor of shape `(batch, sequence_length)`.
            x_lengths: Optional waveform lengths in samples.

        Returns:
            Tuple with features and valid encoder lengths.
        """
        return self.frontend(x, x_lengths=x_lengths)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
        chunk_length: float = 0,
        detach_chunks: bool = True,
    ) -> Any:
        """Forward function for Whisper encoder from waveform input.

        Args:
            x: Input waveforms.
            x_lengths: Optional waveform lengths in samples.
            return_attentions: Whether or not to return attentions from all layers.
            return_hid_states: Whether or not to return hidden states from all layers.
            chunk_length: Chunk size in seconds. Whisper does not support chunked inference here.
            detach_chunks: Unused, kept for interface compatibility.

        Returns:
            HF model output.
        """
        if chunk_length:
            logging.warning(
                "chunked forwarding is not implemented for HFWhisperEncoder; "
                "using Whisper's fixed-length frontend"
            )
        return self.forward_impl(x, x_lengths, return_attentions, return_hid_states)

    def forward_impl(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
    ) -> Any:
        """Forward function for Whisper encoder from waveform input.

        Args:
            x: Input waveforms.
            x_lengths: Optional waveform lengths in samples.
            return_attentions: Whether or not to return attentions from all layers.
            return_hid_states: Whether or not to return hidden states from all layers.

        Returns:
            HF model output.
        """
        x, feat_lengths = self._preprocess(x, x_lengths)
        output = self.hf_model(
            x,
            output_attentions=return_attentions,
            output_hidden_states=return_hid_states,
        )
        output["hidden_states_lengths"] = None if x_lengths is None else feat_lengths
        return output

    def freeze_feature_encoder(self) -> None:
        """Freezes Whisper convolutional frontend parameters.

        Args:
            None.

        Returns:
            None.
        """
        for param in self.hf_model.conv1.parameters():
            param.requires_grad = False
        for param in self.hf_model.conv2.parameters():
            param.requires_grad = False

    def trainable_feat_extract_params(self, bias: bool = True):
        """Returns trainable convolutional frontend parameters.

        Args:
            bias: If False, excludes bias parameters.

        Returns:
            Generator of trainable frontend parameters.
        """
        for module in (
            self._hf_backbone_model().conv1,
            self._hf_backbone_model().conv2,
        ):
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if not bias and name.endswith("bias"):
                    continue
                yield param

    def trainable_encoder_params(self, bias: bool = True):
        """Returns trainable transformer encoder parameters.

        Args:
            bias: If False, excludes bias parameters.

        Returns:
            Generator of trainable encoder parameters.
        """
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.layers.named_parameters():
            if not param.requires_grad:
                continue
            if not bias and name.endswith("bias"):
                continue
            yield param
        for name, param in hf_model.layer_norm.named_parameters():
            if param.requires_grad and (bias or not name.endswith("bias")):
                yield param

    def trainable_feat_extract_bias(self):
        """Returns trainable frontend bias parameters.

        Args:
            None.

        Returns:
            Generator of trainable frontend bias parameters.
        """
        for module in (
            self._hf_backbone_model().conv1,
            self._hf_backbone_model().conv2,
        ):
            for name, param in module.named_parameters():
                if param.requires_grad and name.endswith("bias"):
                    yield param

    def trainable_encoder_bias(self):
        """Returns trainable encoder bias parameters.

        Args:
            None.

        Returns:
            Generator of trainable encoder bias parameters.
        """
        for name, param in self._hf_backbone_model().layers.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param
        for name, param in self._hf_backbone_model().layer_norm.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Returns the configuration arguments for the object in a dictionary.

        Args:
            no_class_name: If True, omits the class name from the returned config.

        Returns:
            Configuration dictionary.
        """
        config = self.hf_model.config.to_dict()
        config = self.filter_args(**config)
        config["hop_length"] = self.hop_length
        config["chunk_length"] = self.chunk_length
        config["n_fft"] = self.n_fft
        config["dither"] = self.dither
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters kwargs to those accepted by constructor arguments.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Filtered constructor argument dictionary.
        """
        args_base = HFWav2VecBase.filter_args(**kwargs)
        args = filter_func_args(HFWhisperEncoder.__init__, kwargs)
        args.update(args_base)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Adds model-construction CLI arguments to parser.

        Args:
            parser: Parser to update.
            prefix: Optional nested prefix for parser composition.
            skip: Optional set of argument names to omit.

        Returns:
            None.
        """
        skip = set() if skip is None else set(skip)

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_class_args(parser, skip=skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip

        config_args = (
            ("vocab_size", "--vocab-size", 51865, int),
            ("num_mel_bins", "--num-mel-bins", 80, int),
            ("encoder_layers", "--encoder-layers", 4, int),
            ("encoder_attention_heads", "--encoder-attention-heads", 6, int),
            ("decoder_layers", "--decoder-layers", 4, int),
            ("decoder_attention_heads", "--decoder-attention-heads", 6, int),
            ("decoder_ffn_dim", "--decoder-ffn-dim", 1536, int),
            ("encoder_ffn_dim", "--encoder-ffn-dim", 1536, int),
            ("encoder_layerdrop", "--encoder-layerdrop", 0.0, float),
            ("decoder_layerdrop", "--decoder-layerdrop", 0.0, float),
            ("d_model", "--d-model", 384, int),
            ("dropout", "--dropout", 0.0, float),
            ("attention_dropout", "--attention-dropout", 0.0, float),
            ("activation_dropout", "--activation-dropout", 0.0, float),
            ("init_std", "--init-std", 0.02, float),
            ("max_source_positions", "--max-source-positions", 1500, int),
            ("max_target_positions", "--max-target-positions", 448, int),
            ("hop_length", "--hop-length", 160, int),
            ("chunk_length", "--chunk-length", 30, int),
            ("n_fft", "--n-fft", 400, int),
            ("dither", "--dither", 0.0, float),
        )
        for var_name, arg_name, default, arg_type in config_args:
            if _use_arg(var_name):
                parser.add_argument(arg_name, default=default, type=arg_type)

        if _use_arg("activation_function"):
            parser.add_argument("--activation-function", default="gelu")
        if _use_arg("scale_embedding"):
            parser.add_argument("--scale-embedding", default=False, action=ActionYesNo)
        if _use_arg("apply_spec_augment"):
            parser.add_argument(
                "--apply-spec-augment", default=False, action=ActionYesNo
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters kwargs to those accepted by finetuning reconfiguration.

        Args:
            **kwargs: Candidate finetuning keyword arguments.

        Returns:
            Filtered finetuning argument dictionary.
        """
        args_base = HFWav2VecBase.filter_finetune_args(**kwargs)
        valid_args = (
            "dropout",
            "attention_dropout",
            "activation_dropout",
            "apply_spec_augment",
            "mask_time_prob",
            "mask_time_length",
            "mask_time_min_masks",
            "mask_feature_prob",
            "mask_feature_length",
            "mask_feature_min_masks",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_base)
        return args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Adds finetuning CLI arguments to parser.

        Args:
            parser: Parser to update.
            prefix: Optional nested prefix for parser composition.
            skip: Optional set of argument names to omit.

        Returns:
            None.
        """
        skip = set() if skip is None else set(skip)

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_finetune_args(parser, skip=skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip

        for var_name, arg_name, default in (
            ("dropout", "--dropout", 0.0),
            ("attention_dropout", "--attention-dropout", 0.0),
            ("activation_dropout", "--activation-dropout", 0.0),
            ("mask_time_prob", "--mask-time-prob", 0.05),
            ("mask_feature_prob", "--mask-feature-prob", 0.0),
        ):
            if _use_arg(var_name):
                parser.add_argument(arg_name, default=default, type=float)

        for var_name, arg_name, default in (
            ("mask_time_length", "--mask-time-length", 10),
            ("mask_time_min_masks", "--mask-time-min-masks", 2),
            ("mask_feature_length", "--mask-feature-length", 10),
            ("mask_feature_min_masks", "--mask-feature-min-masks", 0),
        ):
            if _use_arg(var_name):
                parser.add_argument(arg_name, default=default, type=int)

        if _use_arg("apply_spec_augment"):
            parser.add_argument(
                "--apply-spec-augment", default=False, action=ActionYesNo
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
