"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from typing import Any, Dict, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertConfig,
    Wav2Vec2BertModel,
)

from ...utils.ddp import ddp_get_rank, ddp_wait_for_all_procs
from ...utils import seq_lengths_to_mask
from .hf_wav2vec_base import HFWav2VecBase


class Wav2Vec2BertFrontend(nn.Module):
    """Torch implementation of the SeamlessM4T/Wav2Vec2-BERT audio frontend.

    Attributes:
        sampling_rate: Expected waveform sampling rate.
        num_mel_bins: Number of Kaldi mel filters before frame stacking.
        stride: Number of consecutive mel frames stacked into each model frame.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        stride: int = 2,
    ) -> None:
        """Initializes the Wav2Vec2-BERT frontend.

        Args:
            sampling_rate: Expected waveform sampling rate.
            num_mel_bins: Number of log-mel filters.
            stride: Number of log-mel frames stacked into one model frame.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.frame_length = 400
        self.hop_length = 160
        self.fft_length = 512
        self.preemphasis = 0.97
        self.mel_floor = 1.192092955078125e-07

        window = torch.hann_window(self.frame_length, periodic=False).pow(0.85)
        self.register_buffer("window", window)

        mel_filters = self._make_kaldi_mel_filters(
            num_frequency_bins=self.fft_length // 2 + 1,
            num_mel_filters=num_mel_bins,
            sampling_rate=sampling_rate,
        )
        self.register_buffer(
            "mel_filters",
            torch.tensor(mel_filters, dtype=torch.get_default_dtype()),
        )

    @staticmethod
    def _hertz_to_mel(freq: np.ndarray) -> np.ndarray:
        return 1127.0 * np.log(1.0 + freq / 700.0)

    @classmethod
    def _make_kaldi_mel_filters(
        cls,
        num_frequency_bins: int,
        num_mel_filters: int,
        sampling_rate: int,
    ) -> np.ndarray:
        mel_min = cls._hertz_to_mel(np.array(20.0))
        mel_max = cls._hertz_to_mel(np.array(sampling_rate // 2))
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)

        fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
        fft_freqs = cls._hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins))

        filter_diff = np.diff(mel_freqs)
        slopes = np.expand_dims(mel_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Computes stacked frontend lengths from waveform lengths.

        Args:
            in_lengths: Waveform lengths in samples.

        Returns:
            Lengths after log-mel extraction and stride-frame stacking.
        """
        lengths = torch.div(
            in_lengths - self.frame_length,
            self.hop_length,
            rounding_mode="floor",
        ) + 1
        lengths = torch.clamp(lengths, min=0)
        return torch.div(lengths, self.stride, rounding_mode="floor")

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts stacked normalized log-mel features.

        Args:
            x: Waveform tensor with shape `(batch, num_samples)`.
            x_lengths: Optional waveform lengths in samples.
            normalize: If True, normalizes each mel bin over valid frames.

        Returns:
            Tuple with input features, model attention mask, and feature lengths.
        """
        if x.dim() != 2:
            raise ValueError(f"expected 2-D waveform tensor, got shape {x.shape}")

        batch_size, num_samples = x.shape
        device = x.device
        if x_lengths is None:
            x_lengths = torch.full(
                (batch_size,), num_samples, dtype=torch.long, device=device
            )

        mel_lengths = torch.div(
            x_lengths - self.frame_length,
            self.hop_length,
            rounding_mode="floor",
        ) + 1
        mel_lengths = torch.clamp(mel_lengths, min=0)

        x = x * (2**15)
        if num_samples < self.frame_length:
            num_frames = 0
            mel = x.new_zeros((batch_size, 0, self.num_mel_bins))
        else:
            frames = x.unfold(1, self.frame_length, self.hop_length)
            num_frames = frames.size(1)
            frames = frames - frames.mean(dim=-1, keepdim=True)
            if self.preemphasis != 0:
                prev = frames[..., :-1].clone()
                frames = frames.clone()
                frames[..., 1:] = frames[..., 1:] - self.preemphasis * prev
                frames[..., 0] = frames[..., 0] * (1 - self.preemphasis)

            frames = frames * self.window.to(dtype=frames.dtype)
            frames = torch.nn.functional.pad(
                frames, (0, self.fft_length - self.frame_length)
            )
            spec = torch.fft.rfft(frames, n=self.fft_length, dim=-1).abs().pow(2)
            mel_filters = self.mel_filters.to(dtype=spec.dtype)
            mel = torch.matmul(spec, mel_filters)
            mel = mel.clamp(min=self.mel_floor).log()

        mel_mask = seq_lengths_to_mask(mel_lengths, num_frames, dtype=torch.bool)
        if normalize and num_frames > 0:
            mask = mel_mask.unsqueeze(-1).to(dtype=mel.dtype)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (mel * mask).sum(dim=1, keepdim=True) / counts
            centered = (mel - mean) * mask
            denom = (counts - 1).clamp(min=1)
            var = centered.pow(2).sum(dim=1, keepdim=True) / denom
            mel = (mel - mean) / torch.sqrt(var + 1e-7)
            mel = mel.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        remainder = num_frames % self.stride
        if remainder != 0:
            pad_frames = self.stride - remainder
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_frames))
            mel_mask = torch.nn.functional.pad(mel_mask, (0, pad_frames))
            num_frames = num_frames + pad_frames

        features = mel.reshape(
            batch_size,
            num_frames // self.stride,
            self.num_mel_bins * self.stride,
        )
        attention_mask = mel_mask[:, self.stride - 1 :: self.stride]
        feature_lengths = attention_mask.long().sum(dim=1)
        return features, attention_mask.long(), feature_lengths


class HFWav2Vec2Bert(HFWav2VecBase):
    r"""This is wrapper over HuggingFace Wav2Vec2-BERT model.
        See documentation: https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert

        This wrapper makes the HuggingFace model to have the same interface
        as other hyperion models. It also adds extra functionalities.

        The config. parameters are the same as in the HuggingFace Wav2Vec2BertConfig class.

    Attributes:
        pretrained_model_path (`str`, defaults to None): file path or HuggingFace Hub path to
            pre-trained model.
        normalize_input (`bool`, defaults to True): whether or not to zero-mean unit-variance
            normalize the input.
        use_input_attention_mask (`bool`, defaults to False): whether we should input an
            attention mask to the wav2vec model.
        vocab_size (`int`, defaults to 32): vocabulary size of the
            model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method.
        hidden_size (`int`, defaults to 768): dimensionality of the encoder layers and
            the pooler layer.
        num_hidden_layers (`int`, defaults to 12): number of hidden layers in the
            Transformer encoder.
        num_attention_heads (`int`, defaults to 12): number of attention heads for
            each attention layer in the Transformer encoder.
        intermediate_size (`int`, defaults to 3072): dimensionality of the
            feed-forward layer in the Transformer encoder.
        hidden_act (`str` or `function`, defaults to `"gelu"`): the non-linear
            activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, defaults to 0.1): the dropout probability for all
            fully connected layers in the embeddings, encoder, and pooler.
        activation_dropout (`float`, defaults to 0.1): the dropout probability for all
            intermediate layer in feedforward transformer layers.
        attention_dropout (`float`, defaults to 0.1): the dropout ratio for the
            attention probabilities.
        layerdrop (`float`, defaults to 0.1): prob. of dropping a layer.
        initializer_range (`float`, defaults to 0.02): the standard deviation of the
            truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, defaults to 1e-12): the epsilon used by the layer
            normalization layers.
        feat_extract_norm (`str`, defaults to `"group"`):
            the norm to be applied to 1D convolutional layers in feature encoder.
            One of `"group"` for group normalization of only the first 1D convolutional
            layer or `"layer"` for layer normalization of all 1D convolutional layers.
        feat_proj_dropout (`float`, defaults to 0.0): the dropout probability for output
            of the feature encoder.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`): the non-linear
            activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`Tuple[int]`, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            a tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]`, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            a tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
            of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
        conv_kernel (`Tuple[int]`, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            a tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
            length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, defaults to `False`): whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, defaults to 128):
            number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, defaults to 16):
            number of groups of 1D convolutional positional embeddings layer.
        do_stable_layer_norm (`bool`, defaults to `False`):
            whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is
            True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is
            False` corresponds to applying layer norm after the attention layer.
        apply_spec_augment (`bool`, defaults to `True`):
            whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, defaults to 0.05):
            percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procedure generates ''`mask_time_prob*len(time_axis)/mask_time_length`'' independent masks over the axis. If
            reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, defaults to 10):
            length of vector span along the time axis.
        mask_time_min_masks (`int`, defaults to 2),:
            the minimum number of masks of length `mask_time_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''`mask_time_prob*len(time_axis)/mask_time_length` <
            mask_time_min_masks''
        mask_feature_prob (`float`, defaults to 0.0):
            percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procedure generates ''`mask_feature_prob*len(feature_axis)/mask_time_length`'' independent masks over
            the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, defaults to 10):
            length of vector span along the feature axis.
        mask_feature_min_masks (`int`, defaults to 0):
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''`mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`''
        add_adapter (`bool`, defaults to `False`):
            whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder. Can be very useful for
            warm-starting Wav2Vec2 for SpeechEncoderDecoder models.
        adapter_kernel_size (`int`, defaults to 3):
            kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adapter_stride (`int`, defaults to 2):
            stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        num_adapter_layers (`int`, defaults to 3):
            number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        output_hidden_size (`int`, defaults to None):
            dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
            if `add_adapter is True`.
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
            and initializes the model from the configuration. This is set to True for models that have already
            been fine-tuned.
        override_dropouts (`bool`, defaults to `False`): if True, it ignores the dropout probs. in the pretrained model
            and uses the ones passed as arguments.
        override_spec_augment (`bool`, defaults to `False`): if True, it ignores the spec. augment.
            configuration in the pretrained model and uses the ones passed in the arguments.
        left_encoder_context (`int`, defaults to `16`): past context frames used by the transformer encoder when the signal is evaluated
            chunk by chunk, if it is too long to fit in GPU.
        right_encoder_context (`int`, defaults to `16`): future context frames used by the transformer encoder.
        sample_frequency (`int`, defaults to `16000`): waveform sample frequency used to train the model.
        feat_extract_lr (`Optional[float]`, defaults to `None`): learning rate for conv feature extractor, serves to set a lr different than the global one.
        encoder_lr (`Optional[float]`, defaults to `None`): learning rate for the wav2vec encoder, serves to set a lr different than the global one.
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
        vocab_size: int = 32,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        feature_projection_input_dim: int = 160,
        hidden_act: Union[str, Callable] = "swish",
        hidden_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layerdrop: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        feat_proj_dropout: float = 0.0,
        apply_spec_augment: bool = True,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        add_adapter: bool = False,
        adapter_kernel_size: int = 3,
        adapter_stride: int = 2,
        num_adapter_layers: int = 1,
        adapter_act: Union[str, Callable] = "relu",
        use_intermediate_ffn_before_adapter: bool = False,
        output_hidden_size: Optional[int] = None,
        position_embeddings_type: Optional[str] = "relative_key",
        rotary_embedding_base: int = 10000,
        max_source_positions: int = 5000,
        left_max_position_embeddings: int = 64,
        right_max_position_embeddings: int = 8,
        conv_depthwise_kernel_size: int = 31,
        conformer_conv_dropout: float = 0.1,
        num_mel_bins: int = 80,
        feat_extract_stride: int = 2,
        cache_dir: Union[str, os.PathLike] = "./.cache/hyperion_hf",
        force_download: bool = False,
        resume_download: bool = False,  # deprecated
        revision: str = "main",
        drop_layers_gt: Optional[int] = None,
        ignore_pretrained: bool = False,
        override_dropouts: bool = False,
        override_spec_augment: bool = False,
        left_encoder_context: int = 16,
        right_encoder_context: int = 16,
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
        """Initializes the HuggingFace Wav2Vec2-BERT wrapper.

        Args:
            pretrained_model_path: Local path or HF Hub model id.
            normalize_input: If True, applies waveform normalization.
            use_input_attention_mask: If True, forwards attention mask to HF model.
            vocab_size: Vocabulary size.
            hidden_size: Transformer hidden size.
            num_hidden_layers: Number of transformer encoder layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Feed-forward intermediate size.
            hidden_act: Encoder activation function.
            hidden_dropout: Dropout used in hidden layers.
            activation_dropout: Dropout used in FFN/activation layers.
            attention_dropout: Dropout used in attention probabilities.
            layerdrop: Probability of dropping an encoder layer during training.
            initializer_range: Stddev for weight initialization.
            layer_norm_eps: Epsilon used by layer normalization.
            feat_extract_norm: Feature-extractor normalization type.
            feat_proj_dropout: Dropout after feature projection.
            feat_extract_activation: Feature-extractor activation function.
            conv_dim: Feature extractor channel dimensions.
            conv_stride: Feature extractor strides.
            conv_kernel: Feature extractor kernels.
            conv_bias: Whether conv layers include bias.
            num_conv_pos_embeddings: Convolutional positional embedding size.
            num_conv_pos_embedding_groups: Positional embedding groups.
            do_stable_layer_norm: Whether to use stable layer norm variant.
            apply_spec_augment: Whether to enable SpecAugment.
            mask_time_prob: SpecAugment time masking probability.
            mask_time_length: SpecAugment time mask length.
            mask_time_min_masks: SpecAugment min time masks.
            mask_feature_prob: SpecAugment feature masking probability.
            mask_feature_length: SpecAugment feature mask length.
            mask_feature_min_masks: SpecAugment min feature masks.
            add_adapter: Whether to add HF adapter stack.
            adapter_kernel_size: Adapter conv kernel size.
            adapter_stride: Adapter conv stride.
            num_adapter_layers: Number of adapter layers.
            output_hidden_size: Optional adapter output hidden size.
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
            feat_extract_lr: Optional LR override for feature extractor/projection.
            encoder_lr: Optional LR override for encoder/adapter.
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
        self.num_mel_bins = num_mel_bins
        self.feat_extract_stride = feat_extract_stride

        if pretrained_model_path is not None and not ignore_pretrained:
            rank = ddp_get_rank()
            if rank == 0:
                feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                )
                # rank 0 downloads the model from HF web
                logging.info(f"Downloading HF model from {pretrained_model_path}")
                self.hf_model = Wav2Vec2BertModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                )
            # all ranks wait until the model is downloaded
            ddp_wait_for_all_procs()
            if rank > 0:
                feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    revision=revision,
                )
                # the rest of ranks should read the configuration from the cache.
                self.hf_model = Wav2Vec2BertModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    revision=revision,
                )
            ddp_wait_for_all_procs()
            self.sample_frequency = feature_extractor.sampling_rate
            self.use_input_attention_mask = feature_extractor.return_attention_mask
            self.num_mel_bins = feature_extractor.num_mel_bins
            self.feat_extract_stride = feature_extractor.stride
            self.hf_model.config.layerdrop = 0.0
            self.change_config(
                override_dropouts=self.override_dropouts,
                override_spec_augment=self.override_spec_augment,
                hidden_dropout=hidden_dropout,
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                feat_proj_dropout=feat_proj_dropout,
                conformer_conv_dropout=conformer_conv_dropout,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
            )
        else:
            hf_config = Wav2Vec2BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                feature_projection_input_dim=feature_projection_input_dim,
                hidden_act=hidden_act,
                hidden_dropout=hidden_dropout,
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                feat_proj_dropout=feat_proj_dropout,
                layerdrop=0.0,  # layerdrop,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                apply_spec_augment=apply_spec_augment,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
                add_adapter=add_adapter,
                adapter_kernel_size=adapter_kernel_size,
                adapter_stride=adapter_stride,
                num_adapter_layers=num_adapter_layers,
                adapter_act=adapter_act,
                use_intermediate_ffn_before_adapter=use_intermediate_ffn_before_adapter,
                output_hidden_size=output_hidden_size,
                position_embeddings_type=position_embeddings_type,
                rotary_embedding_base=rotary_embedding_base,
                max_source_positions=max_source_positions,
                left_max_position_embeddings=left_max_position_embeddings,
                right_max_position_embeddings=right_max_position_embeddings,
                conv_depthwise_kernel_size=conv_depthwise_kernel_size,
                conformer_conv_dropout=conformer_conv_dropout,
            )
            self.hf_model = Wav2Vec2BertModel(hf_config)

        self.frontend = Wav2Vec2BertFrontend(
            sampling_rate=self.sample_frequency,
            num_mel_bins=self.num_mel_bins,
            stride=self.feat_extract_stride,
        )

        if drop_layers_gt is not None:
            self.drop_upper_layers(drop_layers_gt)

        if use_lora:
            self._make_lora_layers(
                lora_components,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )

        self.ignore_pretrained = True

    @property
    def num_encoder_layers(self) -> int:
        """Returns the number of transformer encoder layers.

        Args:
            None.

        Returns:
            Number of encoder layers.
        """
        return self.hf_config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Returns the encoder hidden dimension.

        Args:
            None.

        Returns:
            Hidden size.
        """
        return self.hf_config.hidden_size

    def change_dropouts(
        self,
        hidden_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        feat_proj_dropout: float = 0.0,
        conformer_conv_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Updates dropout values in config and instantiated HF modules.

        Args:
            hidden_dropout: Hidden-layer dropout probability.
            activation_dropout: Activation/FFN dropout probability.
            attention_dropout: Attention dropout probability.
            feat_proj_dropout: Feature-projection dropout probability.
            **kwargs: Extra unused keyword arguments.

        Returns:
            None.
        """
        import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as t

        self.hf_model.config.hidden_dropout = hidden_dropout
        self.hf_model.config.activation_dropout = activation_dropout
        self.hf_model.config.attention_dropout = attention_dropout
        self.hf_model.config.feat_proj_dropout = feat_proj_dropout
        self.hf_model.config.conformer_conv_dropout = conformer_conv_dropout

        self.hf_model.feature_projection.dropout.p = feat_proj_dropout
        for module in self.hf_model.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = hidden_dropout

        for module in self.hf_model.encoder.modules():
            if isinstance(module, t.Wav2Vec2BertSelfAttention):
                module.dropout.p = attention_dropout
            if isinstance(module, t.Wav2Vec2BertFeedForward):
                module.intermediate_dropout.p = activation_dropout
            if isinstance(module, t.Wav2Vec2BertConvolutionModule):
                module.dropout.p = conformer_conv_dropout

    def drop_upper_layers(self, max_layers: int) -> None:
        """Drops encoder layers above `max_layers`.

        Args:
            max_layers: Number of lower encoder layers to keep.

        Returns:
            None.
        """
        if max_layers >= self.hf_config.num_hidden_layers:
            return

        layers = self.hf_model.encoder.layers
        self.hf_model.encoder.layers = nn.ModuleList(
            [l for i, l in enumerate(layers) if i < max_layers]
        )
        self.hf_config.num_hidden_layers = max_layers

        if self.hf_model.adapter is not None:
            del self.hf_model.adapter
            self.hf_model.adapter = None
            self.hf_config.add_adapter = False

    @property
    def feature_encoder_context(self) -> Tuple[int, int]:
        """Returns frontend context in waveform samples."""
        left_context = self.frontend.frame_length - self.frontend.hop_length
        right_context = 0
        return left_context, right_context

    @property
    def frame_shift(self) -> int:
        """Returns model-frame shift in waveform samples."""
        return self.frontend.hop_length * self.frontend.stride

    def max_out_length(self, max_in_length: int) -> int:
        """Computes maximum output length from waveform length."""
        in_lengths = torch.tensor(max_in_length, dtype=torch.long)
        return self.out_lengths(in_lengths).item()

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Computes output lengths from waveform lengths."""
        lengths = self.frontend.out_lengths(in_lengths)
        return self.hf_model._get_feat_extract_output_lengths(lengths)

    def out_shape(self, in_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        """Computes output tensor shape from input batch/length shape."""
        out_length = self.max_out_length(in_shape[1])
        return (in_shape[0], out_length, self.hf_model.config.hidden_size)

    def _preprocess(
        self, x: torch.Tensor, x_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Converts waveforms to Wav2Vec2-BERT stacked log-mel features."""
        x, x_mask, feat_lengths = self.frontend(
            x,
            x_lengths=x_lengths,
            normalize=self.normalize_input,
        )
        if not self.use_input_attention_mask:
            x_mask = None
        return x, x_mask, feat_lengths

    def forward_impl(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
    ) -> Any:
        """Forward function for Wav2Vec2-BERT from waveform input."""
        x, x_mask, feat_lengths = self._preprocess(x, x_lengths)
        output = self.hf_model(
            x,
            x_mask,
            output_attentions=return_attentions,
            output_hidden_states=return_hid_states,
        )
        output["hidden_states_lengths"] = (
            None
            if x_lengths is None
            else self.hf_model._get_feat_extract_output_lengths(feat_lengths)
        )
        return output

    def forward_long_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Falls back to full-utterance forwarding for Wav2Vec2-BERT."""
        logging.warning(
            "chunked forwarding is not implemented for HFWav2Vec2Bert; "
            "falling back to full-utterance forwarding"
        )
        return self.forward_impl(*args[:4])

    def freeze_feature_encoder(self) -> None:
        """Freezes the Wav2Vec2-BERT input projection."""
        for param in self.hf_model.feature_projection.parameters():
            param.requires_grad = False

    def trainable_feat_extract_params(self, bias: bool = True):
        """Returns trainable input-projection parameters."""
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.feature_projection.named_parameters():
            if not param.requires_grad:
                continue
            if not bias and name.endswith("bias"):
                continue
            yield param

    def trainable_feat_extract_bias(self):
        """Returns trainable input-projection bias parameters."""
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.feature_projection.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Returns the configuration arguments for the object in a dictionary."""
        config = self.hf_model.config.to_dict()
        config = self.filter_args(**config)
        config["num_mel_bins"] = self.num_mel_bins
        config["feat_extract_stride"] = self.feat_extract_stride
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
        valid_args = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "feature_projection_input_dim",
            "hidden_act",
            "hidden_dropout",
            "activation_dropout",
            "attention_dropout",
            "feat_proj_dropout",
            "layerdrop",
            "initializer_range",
            "layer_norm_eps",
            "apply_spec_augment",
            "mask_time_prob",
            "mask_time_length",
            "mask_time_min_masks",
            "mask_feature_prob",
            "mask_feature_length",
            "mask_feature_min_masks",
            "add_adapter",
            "adapter_kernel_size",
            "adapter_stride",
            "num_adapter_layers",
            "adapter_act",
            "use_intermediate_ffn_before_adapter",
            "output_hidden_size",
            "position_embeddings_type",
            "rotary_embedding_base",
            "max_source_positions",
            "left_max_position_embeddings",
            "right_max_position_embeddings",
            "conv_depthwise_kernel_size",
            "conformer_conv_dropout",
            "num_mel_bins",
            "feat_extract_stride",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_base)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Optional[set] = None
    ) -> None:
        """Adds model-construction CLI arguments to parser.

        Args:
            parser: Parser to update.
            prefix: Optional nested prefix for parser composition.
            skip: Optional set of argument names to omit.

        Returns:
            None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_class_args(parser, skip=skip)

        skip = set() if skip is None else set(skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip
        if _use_arg("vocab_size"):
            parser.add_argument(
                "--vocab-size",
            default=32,
            type=int,
            help=(
                "vocabulary size of the "
                "model. Defines the different tokens that can be represented by the "
                "*inputs_ids* passed to the forward method."
            ),
        )
        if _use_arg("hidden_size"):
            parser.add_argument(
                "--hidden-size",
            default=768,
            type=int,
            help=("dimensionality of the encoder layers and the pooler layer."),
            )
        if _use_arg("num_hidden_layers"):
            parser.add_argument(
                "--num-hidden-layers",
            default=12,
            type=int,
            help=("number of hidden layers in the Transformer encoder"),
            )
        if _use_arg("num_attention_heads"):
            parser.add_argument(
                "--num-attention-heads",
            default=12,
            type=int,
            help=(
                "number of attention heads for "
                "each attention layer in the Transformer encoder"
            ),
        )
        if _use_arg("intermediate_size"):
            parser.add_argument(
                "--intermediate-size",
            default=3072,
            type=int,
            help=(
                "dimensionality of the " "feed-forward layer in the Transformer encoder"
            ),
        )
        if _use_arg("hidden_act"):
            parser.add_argument(
                "--hidden-act",
            default="gelu",
            choices=["gelu", "relu", "selu", "gelu_new"],
            help=(
                "the non-linear "
                "activation function (function or string) in the encoder and pooler"
            ),
        )
        if _use_arg("hidden_dropout"):
            parser.add_argument(
                "--hidden-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "fully connected layers in the embeddings, encoder, and pooler"
            ),
        )
        if _use_arg("activation_dropout"):
            parser.add_argument(
                "--activation-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "intermediate layer in feedforward transformer layers"
            ),
        )
        if _use_arg("attention_dropout"):
            parser.add_argument(
                "--attention-dropout",
            default=0.1,
            type=float,
            help=("the dropout ratio for the attention probabilities"),
            )
        if _use_arg("layerdrop"):
            parser.add_argument(
                "--layerdrop",
            default=0.1,
            type=float,
            help=("prob. of dropping a layer"),
            )
        if _use_arg("initializer_range"):
            parser.add_argument(
                "--initializer-range",
            default=0.02,
            type=float,
            help=(
                "the standard deviation of the "
                "truncated_normal_initializer for initializing all weight matrices"
            ),
        )
        if _use_arg("layer_norm_eps"):
            parser.add_argument(
                "--layer-norm-eps",
            default=1e-12,
            type=float,
            help=(
                "the standard deviation of the "
                "truncated_normal_initializer for initializing all weight matrices"
            ),
        )
        if _use_arg("feat_extract_norm"):
            parser.add_argument(
                "--feat-extract-norm",
            default="group",
            choices=["group", "layer"],
            help=(
                "the norm to be applied to 1D convolutional layers in feature encoder. "
                "One of `group` for group normalization of only the first 1D convolutional "
                "layer or `layer` for layer normalization of all 1D convolutional layers"
            ),
        )
        if _use_arg("feat_proj_dropout"):
            parser.add_argument(
                "--feat-proj-dropout",
            default=0.1,
            type=float,
            help=("the dropout probability for output of the feature encoder"),
            )
        if _use_arg("feat_extract_activation"):
            parser.add_argument(
                "--feat-extract-activation",
            default="gelu",
            choices=["gelu", "relu", "selu", "gelu_new"],
            help=(
                "the non-linear activation function (function or string) in the 1D "
                "convolutional layers of the feature extractor"
            ),
        )
        if _use_arg("conv_dim"):
            parser.add_argument(
                "--conv-dim",
            default=[512, 512, 512, 512, 512, 512, 512],
            nargs="+",
            type=int,
            help=(
                "a tuple of integers defining the number of input and output channels of each 1D convolutional layer in the "
                "feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers"
            ),
        )
        if _use_arg("conv_stride"):
            parser.add_argument(
                "--conv-stride",
            default=[5, 2, 2, 2, 2, 2, 2],
            nargs="+",
            type=int,
            help=(
                "a tuple of integers defining the stride of each 1D convolutional layer in the feature encoder"
            ),
        )
        if _use_arg("conv_kernel"):
            parser.add_argument(
                "--conv-kernel",
            default=[10, 3, 3, 3, 3, 3, 3],
            nargs="+",
            type=int,
            help=(
                "a tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder"
            ),
        )
        if _use_arg("conv_bias"):
            parser.add_argument(
                "--conv-bias",
            default=False,
            action=ActionYesNo,
            help=("whether the 1D convolutional layers have a bias"),
            )
        if _use_arg("num_conv_pos_embeddings"):
            parser.add_argument(
                "--num-conv-pos-embeddings",
            default=128,
            type=int,
            help=(
                "number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional "
                "embeddings layer"
            ),
        )
        if _use_arg("num_conv_pos_embedding_groups"):
            parser.add_argument(
                "--num-conv-pos-embedding-groups",
            default=16,
            type=int,
            help=("number of groups of 1D convolutional positional embeddings layer"),
            )
        if _use_arg("do_stable_layer_norm"):
            parser.add_argument(
                "--do-stable-layer-norm",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to apply *stable* layer norm architecture of the Transformer encoder"
            ),
        )
        if _use_arg("apply_spec_augment"):
            parser.add_argument(
                "--apply-spec-augment",
            default=True,
            action=ActionYesNo,
            help=(
                "whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder"
            ),
        )
        if _use_arg("mask_time_prob"):
            parser.add_argument(
                "--mask-time-prob",
            default=0.05,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the time axis which will be masked"
            ),
        )
        if _use_arg("mask_time_length"):
            parser.add_argument(
                "--mask-time-length",
            default=10,
            type=int,
            help=("length of vector span along the time axis"),
            )
        if _use_arg("mask_time_min_masks"):
            parser.add_argument(
                "--mask-time-min-masks",
            default=2,
            type=int,
            help=(
                "the minimum number of masks of length `mask_time_length` generated along the time axis"
            ),
        )
        if _use_arg("mask_feature_prob"):
            parser.add_argument(
                "--mask-feature-prob",
            default=0.0,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked"
            ),
        )
        if _use_arg("mask_feature_length"):
            parser.add_argument(
                "--mask-feature-length",
            default=10,
            type=int,
            help=(" length of vector span along the feature axis"),
            )
        if _use_arg("mask_feature_min_masks"):
            parser.add_argument(
                "--mask-feature-min-masks",
            default=0,
            type=int,
            help=(
                "The minimum number of masks of length `mask_feature_length` generated along the feature axis"
            ),
        )
        if _use_arg("add_adapter"):
            parser.add_argument(
                "--add-adapter",
            default=False,
            action=ActionYesNo,
            help=(
                "whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder"
            ),
        )
        if _use_arg("adapter_kernel_size"):
            parser.add_argument(
                "--adapter-kernel-size",
            default=3,
            type=int,
            help=("kernel size of the convolutional layers in the adapter network"),
            )
        if _use_arg("adapter_stride"):
            parser.add_argument(
                "--adapter-stride",
            default=2,
            type=int,
            help=("stride of the convolutional layers in the adapter network"),
            )
        if _use_arg("num_adapter_layers"):
            parser.add_argument(
                "--num-adapter-layers",
            default=3,
            type=int,
            help=(
                "number of convolutional layers that should be used in the adapter network"
            ),
        )
        if _use_arg("output_hidden_size"):
            parser.add_argument(
                "--output-hidden-size",
            default=None,
            type=int,
            help=(
                "dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*."
                " Only relevant if `add_adapter is True"
            ),
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
            "hidden_dropout",
            "activation_dropout",
            "attention_dropout",
            "feat_proj_dropout",
            "conformer_conv_dropout",
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
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Optional[set] = None
    ) -> None:
        """Adds finetuning CLI arguments to parser.

        Args:
            parser: Parser to update.
            prefix: Optional nested prefix for parser composition.
            skip: Optional set of argument names to omit.

        Returns:
            None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_finetune_args(parser, skip=skip)

        skip = set() if skip is None else set(skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip
        if _use_arg("hidden_dropout"):
            parser.add_argument(
                "--hidden-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "fully connected layers in the embeddings, encoder, and pooler"
            ),
        )
        if _use_arg("activation_dropout"):
            parser.add_argument(
                "--activation-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "intermediate layer in feedforward transformer layers"
            ),
        )
        if _use_arg("attention_dropout"):
            parser.add_argument(
                "--attention-dropout",
            default=0.1,
            type=float,
            help=("the dropout ratio for the attention probabilities"),
            )
        if _use_arg("apply_spec_augment"):
            parser.add_argument(
                "--apply-spec-augment",
            default=True,
            action=ActionYesNo,
            help=(
                "whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder"
            ),
        )
        if _use_arg("mask_time_prob"):
            parser.add_argument(
                "--mask-time-prob",
            default=0.05,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the time axis which will be masked"
            ),
        )
        if _use_arg("mask_time_length"):
            parser.add_argument(
                "--mask-time-length",
            default=10,
            type=int,
            help=("length of vector span along the time axis"),
            )
        if _use_arg("mask_time_min_masks"):
            parser.add_argument(
                "--mask-time-min-masks",
            default=2,
            type=int,
            help=(
                "the minimum number of masks of length `mask_time_length` generated along the time axis"
            ),
        )
        if _use_arg("mask_feature_prob"):
            parser.add_argument(
                "--mask-feature-prob",
            default=0.0,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked"
            ),
        )
        if _use_arg("mask_feature_length"):
            parser.add_argument(
                "--mask-feature-length",
            default=10,
            type=int,
            help=(" length of vector span along the feature axis"),
            )
        if _use_arg("mask_feature_min_masks"):
            parser.add_argument(
                "--mask-feature-min-masks",
            default=0,
            type=int,
            help=(
                "The minimum number of masks of length `mask_feature_length` generated along the feature axis"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
