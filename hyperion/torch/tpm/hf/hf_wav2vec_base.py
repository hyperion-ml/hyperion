"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from ....utils.misc import PathLike, filter_func_args
from ...hyper_torch_model import HyperTorchModel
from ...utils import scale_seq_lengths, seq_lengths_to_mask
from ...utils.ddp import ddp_get_rank, ddp_wait_for_all_procs


class HFWav2VecBase(HyperTorchModel):
    """Base class for Wav2Vec style models (Wav2Vec2, Hubert, WavLM, ...) in HuggingFace.

    This class includes the preprocessing steps common to all models.

    Attributes:
        pretrained_model_path (`PathLike`, defaults to None): file path or
            HuggingFace Hub path to pre-trained model.
        normalize_input (`bool`, defaults to True): whether or not to zero-mean unit-variance
            normalize the input.
        use_input_attention_mask (`bool`, defaults to False): whether we should input an
            attention mask to the wav2vec model.
        cache_dir (`PathLike`, defaults to `"./.cache/hyperion_hf"`): path to a directory in which a downloaded pretrained
            model configuration should be cached if the standard cache should not be used.
        force_download (`bool`, defaults to `False`): whether or not to force the (re-)download
            the model weights and configuration files and override the
            cached versions if they exist.
        revision (`str`, defaults to `"main"`): the specific model version to use.
            It can be a branch name, a tag name, or a commit id.
        drop_layers_gt (`Optional[int]`, defaults to `None`): drop encoder layers greater than this value (in [1, num_encoder_layers]).
            If None, the model is not changed.
        ignore_pretrained (`bool`, defaults to `False`): if True, it ignores the pretrained_model_path
            and initializes the model from the configuration. This is set to True for models that have
            already been fine-tuned.
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
        lora_components (`Optional[List[str]]`, defaults to `None`): list of components where we apply LoRA, e.g., [Wq, Wv]
        lora_rank (`int`, defaults to `4`): rank of LoRA
        lora_alpha (`int`, defaults to `8`): scale for LoRA
        lora_dropout (`float`, defaults to `0.0`): dropout rate for LoRA
        lora_merge_weights (`bool`, defaults to `True`): lora weights are merged with the pretrained weights at inference.
        bias_weight_decay (`Optional[float]`, defaults to `None`): weight decay for bias parameters, if not None overrides global weight decay
    """

    def __init__(
        self,
        pretrained_model_path: Optional[PathLike] = None,
        normalize_input: bool = True,
        use_input_attention_mask: bool = False,
        cache_dir: PathLike = "./.cache/hyperion_hf",
        force_download: bool = False,
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
        lora_components: Optional[List[str]] = None,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
        bias_weight_decay: Optional[float] = None,
        resume_download: bool = False,  # deprecated
    ):
        """Initializes the HuggingFace Wav2Vec-style base wrapper.

        Args:
            pretrained_model_path: Local path or HF Hub identifier for a pretrained model.
            normalize_input: If True, applies zero-mean, unit-variance normalization.
            use_input_attention_mask: If True, forwards an input attention mask.
            cache_dir: Cache directory used by HuggingFace download utilities.
            force_download: If True, forces re-download of pretrained artifacts.
            revision: Model revision name, tag, or commit ID.
            drop_layers_gt: Optional layer index threshold used by subclasses.
            ignore_pretrained: If True, avoids loading pretrained processor settings.
            override_dropouts: If True, enables dropout override in finetuning config.
            override_spec_augment: If True, enables spec-augment override.
            left_encoder_context: Left encoder context in frames for chunked inference.
            right_encoder_context: Right encoder context in frames for chunked inference.
            sample_frequency: Waveform sampling rate in Hz.
            feat_extract_lr: Optional LR override for feature extractor parameters.
            encoder_lr: Optional LR override for transformer encoder parameters.
            use_lora: If True, enables LoRA adapters.
            lora_components: Linear module names where LoRA is applied.
            lora_rank: Rank used by LoRA adapters.
            lora_alpha: Scaling coefficient used by LoRA adapters.
            lora_dropout: Dropout probability for LoRA adapters.
            lora_merge_weights: If True, merges LoRA weights at inference.
            bias_weight_decay: Optional weight decay for bias parameters.
            resume_download: Deprecated argument kept for backward compatibility.

        Returns:
            None.
        """
        if lora_components is None:
            lora_components = ["q_proj", "v_proj"]

        super().__init__(bias_weight_decay=bias_weight_decay)
        self.pretrained_model_path = pretrained_model_path
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.revision = revision
        self.drop_layers_gt = drop_layers_gt
        self.ignore_pretrained = ignore_pretrained
        self.override_dropouts = override_dropouts
        self.override_spec_augment = override_spec_augment
        self.right_encoder_context = right_encoder_context
        self.left_encoder_context = left_encoder_context
        self.feat_extract_lr = feat_extract_lr
        self.encoder_lr = encoder_lr
        self.use_lora = use_lora
        self.lora_components = lora_components
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_merge_weights = lora_merge_weights

        if pretrained_model_path is not None and not ignore_pretrained:
            rank = ddp_get_rank()
            if rank == 0:
                logging.info(
                    f"Downloading config for HF preprocessor from {pretrained_model_path}"
                )
                # rank 0 downloads the model from HF web
                try:
                    # some models donot have config for processor because do not have
                    # tokenizer, first we try to donwload feature_extractor config
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        revision=revision,
                    )
                except (OSError, ValueError):
                    # if fails, we try to download full processor config
                    processor = Wav2Vec2Processor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        revision=revision,
                    )
                    feature_extractor = processor.feature_extractor

            # all ranks wait until the model is downloaded
            ddp_wait_for_all_procs()
            if rank > 0:
                # the rest of ranks should read the configuration from the cache.
                try:
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=False,
                        revision=revision,
                    )
                except (OSError, ValueError):
                    # if fails, we try to download full processor config
                    processor = Wav2Vec2Processor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=False,
                        revision=revision,
                    )
                    feature_extractor = processor.feature_extractor

            ddp_wait_for_all_procs()
            normalize_input = feature_extractor.do_normalize
            use_input_attention_mask = feature_extractor.return_attention_mask
            sample_frequency = feature_extractor.sampling_rate

        self.normalize_input = normalize_input
        self.use_input_attention_mask = use_input_attention_mask
        self.sample_frequency = sample_frequency

        self._feature_encoder_context = None
        self._frame_shift = None
        self.hf_model = None
        self._lora_is_merged = False

    def __deepcopy__(self, memo: Dict[int, Any]) -> "HFWav2VecBase":
        """Reimplementation of deepcopy for Hugging Face models.

        The `weight_norm` in the Conv. Pos. Encoder of Wav2Vec models makes the
        default deepcopy fail.
        """
        cls = self.__class__  # Extract the class of the object
        cfg = self.get_config()
        del cfg["class_name"]
        # Create a new instance of the object based on extracted class
        new_obj = cls(**cfg)
        memo[id(self)] = new_obj
        new_obj.load_state_dict(self.state_dict())
        device = next(self.parameters()).device
        new_obj.to(device)
        return new_obj

    @property
    def feature_encoder_context(self) -> Tuple[int, int]:
        """Computes feature-extractor left/right temporal context in samples.

        Args:
            None.

        Returns:
            A tuple `(left_context, right_context)` in input samples.
        """
        if self._feature_encoder_context is not None:
            return self._feature_encoder_context

        total_context = 0
        total_stride = 1
        for kernel, stride in zip(
            self.hf_model.config.conv_kernel, self.hf_model.config.conv_stride
        ):
            total_context += total_stride * (kernel - 1) / 2
            total_stride *= stride

        self._feature_encoder_context = (int(total_context + 0.5), int(total_context))
        return self._feature_encoder_context

    @property
    def frame_shift(self) -> int:
        """Computes feature-extractor frame shift in input samples.

        Args:
            None.

        Returns:
            The effective frame shift (product of convolution strides).
        """
        if self._frame_shift is not None:
            return self._frame_shift

        total_stride = 1
        for stride in self.hf_model.config.conv_stride:
            total_stride *= stride

        self._frame_shift = total_stride
        return total_stride

    @property
    def context(self) -> Tuple[int, int]:
        """Returns total context including feature and encoder contexts.

        Args:
            None.

        Returns:
            A tuple `(left_context, right_context)` in input samples.
        """
        left, right = self.feature_encoder_context
        left += self.left_encoder_context
        right += self.right_encoder_context
        return left, right

    def max_out_length(self, max_in_length: int) -> int:
        """Computes maximum output sequence length for an input length.

        Args:
            max_in_length: Input length in waveform samples.

        Returns:
            Maximum encoded sequence length in frames.
        """
        return self.hf_model._get_feat_extract_output_lengths(max_in_length).item()
        # left_context, right_context = self.feature_encoder_context
        # max_in_length = max_in_length - left_context - right_context
        # return max_in_length // self.frame_shift

    def out_lengths(self, in_lengths: torch.Tensor) -> torch.Tensor:
        """Computes output frame lengths for a tensor of input lengths.

        Args:
            in_lengths: Input lengths in waveform samples.

        Returns:
            Output lengths in encoded frames.
        """
        return self.hf_model._get_feat_extract_output_lengths(in_lengths)
        # left_context, right_context = self.feature_encoder_context
        # in_lengths = in_lengths - left_context - right_context
        # return torch.div(in_lengths, self.frame_shift, rounding_mode="floor")

    def out_shape(self, in_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        """Computes output tensor shape from input batch/length shape.

        Args:
            in_shape: Input shape tuple `(batch_size, num_samples)`.

        Returns:
            Output shape tuple `(batch_size, num_frames, hidden_size)`.
        """
        out_length = self.max_out_length(in_shape[1])
        C = self.hf_model.config.hidden_size
        return (in_shape[0], out_length, C)

    @property
    def out_feats(self) -> int:
        """Number of output features of the model."""
        return self.hf_model.config.hidden_size

    def change_config(
        self,
        override_dropouts: bool,
        override_spec_augment: bool,
        override_lora: bool = False,
        feat_extract_lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        use_lora: bool = False,
        lora_components: Optional[List[str]] = None,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """Updates finetuning-time configuration options.

        Args:
            override_dropouts: If True, applies dropout overrides from `kwargs`.
            override_spec_augment: If True, applies spec-augment overrides from `kwargs`.
            override_lora: If True, updates LoRA settings.
            feat_extract_lr: Optional LR override for feature extractor parameters.
            encoder_lr: Optional LR override for encoder parameters.
            use_lora: Whether LoRA adapters should be active.
            lora_components: Module names where LoRA should be applied.
            lora_rank: LoRA rank.
            lora_alpha: LoRA scaling coefficient.
            lora_dropout: LoRA dropout probability.
            lora_merge_weights: Whether LoRA weights are merged at inference.
            **kwargs: Extra keyword args forwarded to override handlers.

        Returns:
            None.
        """
        if lora_components is None:
            lora_components = ["q_proj", "v_proj"]

        if override_spec_augment:
            logging.info(f"overriding speech augment with args={kwargs}")
            self.change_spec_augment(**kwargs)

        if override_dropouts:
            logging.info(f"overriding hf model dropouts with args={kwargs}")
            self.change_dropouts(**kwargs)

        if override_lora:
            logging.info("overriding LoRA config")
            self.change_lora(
                use_lora=use_lora,
                lora_components=lora_components,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_merge_weights=lora_merge_weights,
            )

        self.feat_extract_lr = feat_extract_lr
        self.encoder_lr = encoder_lr

    def change_spec_augment(
        self,
        apply_spec_augment: bool = True,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        **kwargs: Any,
    ) -> None:
        """Sets SpecAugment-related fields in the HF model config.

        Args:
            apply_spec_augment: Enables or disables SpecAugment.
            mask_time_prob: Time-mask sampling probability.
            mask_time_length: Time-mask length in frames.
            mask_time_min_masks: Minimum number of time masks.
            mask_feature_prob: Feature-mask sampling probability.
            mask_feature_length: Feature-mask length in channels.
            mask_feature_min_masks: Minimum number of feature masks.
            **kwargs: Unused extra keyword arguments for API compatibility.

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

    def change_lora(
        self,
        use_lora: bool = False,
        lora_components: Optional[List[str]] = None,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
    ) -> None:
        """Updates LoRA configuration and optionally creates LoRA layers via PEFT.

        Args:
            use_lora: Target LoRA enabled/disabled state.
            lora_components: Module names where LoRA should be applied.
            lora_rank: LoRA rank.
            lora_alpha: LoRA scaling coefficient.
            lora_dropout: LoRA dropout probability.
            lora_merge_weights: Whether LoRA weights are merged at inference.

        Returns:
            None.
        """
        if lora_components is None:
            lora_components = ["q_proj", "v_proj"]

        if use_lora and not self.use_lora:
            self._make_lora_layers(
                lora_components,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )
        elif not use_lora and self.use_lora:
            self._merge_lora_and_unload()
        elif use_lora and self.use_lora:
            logging.info(
                "LoRA already enabled, keeping existing adapter configuration."
            )

        self.use_lora = use_lora
        self.lora_components = lora_components
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_merge_weights = lora_merge_weights

    def _make_lora_layers(
        self,
        lora_components: List[str],
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> None:
        """Injects LoRA adapters into selected layers using HuggingFace PEFT.

        Args:
            lora_components: Module names where LoRA should be applied.
            lora_rank: LoRA rank.
            lora_alpha: LoRA scaling coefficient.
            lora_dropout: LoRA dropout probability.

        Returns:
            None.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "PEFT is required to use LoRA in HFWav2VecBase. Install it with `pip install peft`."
            ) from e

        if hasattr(self.hf_model, "peft_config"):
            logging.info("LoRA adapters are already present, skipping reinjection.")
            return

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_components,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.hf_model = get_peft_model(self.hf_model, peft_config)

        num_lora_layers = sum(
            1
            for _, module in self.hf_model.named_modules()
            if hasattr(module, "lora_A")
        )
        logging.info("count of LoRA-injected layers = %d", num_lora_layers)
        assert num_lora_layers > 0, "did not inject any LoRA layers"
        self._sync_lora_merge_state()

    def change_dropouts(self, **kwargs: Any) -> None:
        """Abstract dropout override hook implemented by subclasses.

        Args:
            **kwargs: Dropout override parameters.

        Returns:
            None.
        """
        pass  # needs to be overloaded

    def freeze_feature_encoder(self) -> None:
        """Freezes feature-extractor parameters of the HF backbone.

        Args:
            None.

        Returns:
            None.
        """
        self.hf_model.freeze_feature_encoder()

    def freeze_except_lora(self, bias: Optional[str] = None) -> None:
        """Freezes all parameters except LoRA (and optional bias parameters).

        Args:
            bias: Bias handling mode. Valid values: `none`, `all`, `lora_only`.

        Returns:
            None.
        """
        bias = "none" if bias is None else bias
        if bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be one of {'none', 'all', 'lora_only'}")

        for p in self.hf_model.parameters():
            p.requires_grad = False

        lora_module_prefixes = set()
        for name, param in self.hf_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                lora_module_prefixes.add(name.split(".lora_", 1)[0])

        if bias == "all":
            for name, param in self.hf_model.named_parameters():
                if name.endswith("bias"):
                    param.requires_grad = True
        elif bias == "lora_only":
            for name, param in self.hf_model.named_parameters():
                if not name.endswith("bias"):
                    continue
                module_name = name.rsplit(".", 1)[0]
                if module_name in lora_module_prefixes:
                    param.requires_grad = True

    def _merge_lora_adapters(self) -> None:
        """Merges LoRA adapters into base weights while preserving adapter params."""
        if self._lora_is_merged:
            return

        if hasattr(self.hf_model, "merge_adapter"):
            self.hf_model.merge_adapter()
            self._lora_is_merged = True
            return

        merged_any = False
        for module in self.hf_model.modules():
            merge_fn = getattr(module, "merge", None)
            if callable(merge_fn):
                merge_fn()
                merged_any = True
        if merged_any:
            self._lora_is_merged = True
            return

        logging.warning("Unable to merge LoRA adapters for current model type.")

    def _unmerge_lora_adapters(self) -> None:
        """Unmerges LoRA adapters from base weights, restoring separate adapter path."""
        if not self._lora_is_merged:
            return

        if hasattr(self.hf_model, "unmerge_adapter"):
            self.hf_model.unmerge_adapter()
            self._lora_is_merged = False
            return

        unmerged_any = False
        for module in self.hf_model.modules():
            unmerge_fn = getattr(module, "unmerge", None)
            if callable(unmerge_fn):
                unmerge_fn()
                unmerged_any = True
        if unmerged_any:
            self._lora_is_merged = False
            return

        logging.warning("Unable to unmerge LoRA adapters for current model type.")

    def _disable_lora_adapters(self) -> None:
        """Disables LoRA adapters without merging when supported by PEFT."""
        if hasattr(self.hf_model, "disable_adapter"):
            self.hf_model.disable_adapter()
            return
        if hasattr(self.hf_model, "disable_adapter_layers"):
            self.hf_model.disable_adapter_layers()
            return
        logging.warning("Unable to disable LoRA adapters for current model type.")

    def _merge_lora_and_unload(self) -> None:
        """Merges LoRA into base weights and removes PEFT adapter wrappers."""
        if hasattr(self.hf_model, "merge_and_unload"):
            self.hf_model = self.hf_model.merge_and_unload()
            self._lora_is_merged = False
            return

        raise RuntimeError(
            "Current PEFT model does not expose `merge_and_unload`, cannot remove "
            "adapter wrappers. Upgrade PEFT or keep use_lora=True."
        )

    def _sync_lora_merge_state(self) -> None:
        """Applies reversible LoRA merge policy based on mode and configuration."""
        if not self.use_lora or not self.lora_merge_weights:
            return

        if self.training:
            self._unmerge_lora_adapters()
        else:
            self._merge_lora_adapters()

    def train(self, mode: bool = True):
        """Sets training mode and synchronizes reversible LoRA merge state."""
        super().train(mode)
        self._sync_lora_merge_state()
        return self

    def _hf_backbone_model(self) -> nn.Module:
        """Returns the underlying HF backbone module, unwrapping PEFT wrappers if present."""
        if hasattr(self.hf_model, "get_base_model"):
            try:
                return self.hf_model.get_base_model()
            except Exception:
                pass
        return self.hf_model

    def has_param_groups(self) -> bool:
        """Checks whether optimizer parameter grouping should be customized.

        Args:
            None.

        Returns:
            True when LR overrides require explicit parameter groups.
        """
        return (
            self.feat_extract_lr is not None
            or self.encoder_lr is not None
            or self.bias_weight_decay is not None
        )

    def trainable_feat_extract_params(self, bias: bool = True):
        """Returns trainable feature-extractor and feature-projection parameters.

        Args:
            bias: If False, excludes bias parameters.

        Returns:
            Generator of trainable parameters from feature extractor/projection.
        """
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.feature_extractor.named_parameters():
            if not param.requires_grad:
                continue
            if not bias and name.endswith("bias"):
                continue
            yield param

        for name, param in hf_model.feature_projection.named_parameters():
            if not param.requires_grad:
                continue
            if not bias and name.endswith("bias"):
                continue
            yield param

    def trainable_encoder_params(self, bias: bool = True):
        """Returns trainable encoder parameters and optional HF adapter parameters.

        Args:
            bias: If False, excludes bias parameters.

        Returns:
            Generator of trainable parameters from encoder and optional adapter.
        """
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if not bias and name.endswith("bias"):
                continue
            yield param

        if getattr(hf_model, "adapter", None) is not None:
            for name, param in hf_model.adapter.named_parameters():
                if not param.requires_grad:
                    continue
                if not bias and name.endswith("bias"):
                    continue
                yield param

    def trainable_feat_extract_bias(self):
        """Returns trainable bias parameters from feature extractor/projection.

        Args:
            None.

        Returns:
            Generator of trainable bias parameters from feature extractor/projection.
        """
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.feature_extractor.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

        for name, param in hf_model.feature_projection.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

    def trainable_encoder_bias(self):
        """Returns trainable bias parameters from encoder and optional adapter.

        Args:
            None.

        Returns:
            Generator of trainable bias parameters from encoder/adapter.
        """
        hf_model = self._hf_backbone_model()
        for name, param in hf_model.encoder.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

        if getattr(hf_model, "adapter", None) is not None:
            for name, param in hf_model.adapter.named_parameters():
                if param.requires_grad and name.endswith("bias"):
                    yield param

    def trainable_bias(self):
        """Returns all trainable bias parameters in the model.

        Args:
            None.

        Returns:
            Generator of trainable bias parameters.
        """
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith("bias"):
                yield param

    def trainable_param_groups(
        self,
    ) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
        """Builds trainable parameter groups with LR and bias-decay overrides.

        Args:
            None.

        Returns:
            Either a flat parameter iterable or a list of optimizer param groups.
        """
        if not self.has_param_groups():
            return self.trainable_parameters()

        separate_bias = self.bias_weight_decay is not None
        include_bias_in_regular = not separate_bias

        feat_params = list(
            self.trainable_feat_extract_params(bias=include_bias_in_regular)
        )
        enc_params = list(self.trainable_encoder_params(bias=include_bias_in_regular))

        param_groups: List[Dict[str, Any]] = []
        if self.feat_extract_lr == self.encoder_lr:
            regular_params = feat_params + enc_params
            if regular_params:
                group: Dict[str, Any] = {"params": regular_params}
                if self.encoder_lr is not None:
                    group["lr"] = self.encoder_lr
                param_groups.append(group)
        else:
            if feat_params:
                feat_group: Dict[str, Any] = {"params": feat_params}
                if self.feat_extract_lr is not None:
                    feat_group["lr"] = self.feat_extract_lr
                param_groups.append(feat_group)

            if enc_params:
                enc_group: Dict[str, Any] = {"params": enc_params}
                if self.encoder_lr is not None:
                    enc_group["lr"] = self.encoder_lr
                param_groups.append(enc_group)

        if separate_bias:
            feat_bias = list(self.trainable_feat_extract_bias())
            enc_bias = list(self.trainable_encoder_bias())
            if self.feat_extract_lr == self.encoder_lr:
                bias_params = feat_bias + enc_bias
                if bias_params:
                    bias_group: Dict[str, Any] = {
                        "params": bias_params,
                        "weight_decay": self.bias_weight_decay,
                    }
                    if self.encoder_lr is not None:
                        bias_group["lr"] = self.encoder_lr
                    param_groups.append(bias_group)
            else:
                if feat_bias:
                    feat_bias_group: Dict[str, Any] = {
                        "params": feat_bias,
                        "weight_decay": self.bias_weight_decay,
                    }
                    if self.feat_extract_lr is not None:
                        feat_bias_group["lr"] = self.feat_extract_lr
                    param_groups.append(feat_bias_group)

                if enc_bias:
                    enc_bias_group: Dict[str, Any] = {
                        "params": enc_bias,
                        "weight_decay": self.bias_weight_decay,
                    }
                    if self.encoder_lr is not None:
                        enc_bias_group["lr"] = self.encoder_lr
                    param_groups.append(enc_bias_group)

        return param_groups

    @property
    def hf_config(self) -> Any:
        """Returns the underlying HuggingFace model config object.

        Args:
            None.

        Returns:
            The HuggingFace configuration object.
        """
        return self.hf_model.config

    def _normalize(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalizes the audio to have zero mean and unit variance."""
        if x_mask is None:
            x = x - x.mean(dim=1, keepdim=True)
            std = torch.sqrt((x**2).mean(dim=1, keepdim=True) + 1e-7)
            x = x / std
        else:
            x_mask = x_mask.to(dtype=x.dtype)
            x_samples = torch.mean(x_mask, dim=1, keepdim=True)
            x_mean = torch.mean(x * x_mask, dim=1, keepdim=True) / x_samples
            x2_mean = torch.mean(x**2 * x_mask, dim=1, keepdim=True) / x_samples
            std = torch.sqrt(x2_mean - x_mean**2 + 1e-7)
            x = (x - x_mean) / std
        return x

    def _preprocess(
        self, x: torch.Tensor, x_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepares input audio to be used by a Wav2Vec-style model."""
        x_mask = seq_lengths_to_mask(x_lengths, x.size(-1), dtype=torch.long)
        if self.normalize_input:
            x = self._normalize(x, x_mask)

        if not self.use_input_attention_mask:
            x_mask = None

        return x, x_mask

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
        chunk_length: float = 0,
        detach_chunks: bool = True,
    ) -> Any:
        r"""Forward function for Wav2Vec-style models.

        Uses chunked evaluation for long utterances when `chunk_length > 0` and
        the input duration exceeds that chunk length.

        Args:
          x: input audio of shape = (batch, sequence_length).
          x_lengths: lengths of the audio waveforms in samples with shape = (batch,).
          return_attentions: whether or not to return the attentions tensors of
            all attention layers.
          return_hid_states: whether or not to return the hidden states of all layers.
          chunk_length: chunk size in seconds.

        Returns:
          Dictionary with:
            last_hidden_state: sequence of hidden-states at the output of the last
                layer of the model (torch.FloatTensor of shape
                (batch_size, sequence_length, hidden_size)).
            extract_features: sequence of extracted feature vectors of the last
                convolutional layer of the model. (torch.FloatTensor of shape
                (batch_size, sequence_length, conv_dim[-1])
            hidden_states: hidden-states of the model at the output of each layer
                plus the initial embedding outputs (tuple(torch.FloatTensor)).
            attentions: Attentions weights after the attention softmax, used to
                compute the weighted average in the self-attention heads
                (tuple(torch.FloatTensor)).
        """
        if chunk_length == 0 or x.size(1) < chunk_length * self.sample_frequency:
            return self.forward_impl(x, x_lengths, return_attentions, return_hid_states)
        else:
            return self.forward_long_impl(
                x,
                x_lengths,
                return_attentions,
                return_hid_states,
                chunk_length,
                detach_chunks,
            )

    def forward_impl(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
    ) -> Any:
        r"""Forward function for wav2vec style models.

        Args:
          x: input audio of shape = (batch, sequence_length).
          x_lengths: lengths of the audio waveforms in samples with shape = (batch,).
          return_attentions: whether or not to return the attentions tensors of
            all attention layers.
          return_hid_states: whether or not to return the hidden states of all layers.

        Returns:
          Dictionary with:
            last_hidden_state: sequence of hidden-states at the output of the last
                layer of the model (torch.FloatTensor of shape
                (batch_size, sequence_length, hidden_size)).
            extract_features: sequence of extracted feature vectors of the last
                convolutional layer of the model. (torch.FloatTensor of shape
                (batch_size, sequence_length, conv_dim[-1])
            hidden_states: hidden-states of the model at the output of each layer
                plus the initial embedding outputs (tuple(torch.FloatTensor)).
            attentions: Attentions weights after the attention softmax, used to
                compute the weighted average in the self-attention heads
                (tuple(torch.FloatTensor)).
        """
        max_in_length = x.size(-1)
        x, x_mask = self._preprocess(x, x_lengths)
        output = self.hf_model(
            x,
            x_mask,
            output_attentions=return_attentions,
            output_hidden_states=return_hid_states,
        )
        max_out_length = output.last_hidden_state.size(1)
        feat_lengths = (
            None
            if x_lengths is None
            else scale_seq_lengths(x_lengths, max_out_length, max_in_length)
        )
        output["hidden_states_lengths"] = feat_lengths

        return output

    def forward_long_impl(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
        chunk_length: float = 120.0,
        detach_chunks: bool = True,
    ) -> Any:
        r"""Forward function for long utterances that do not fit in GPU memory.

        Args:
          x: input audio of shape = (batch, sequence_length).
          x_lengths: lengths of the audio waveforms in samples with shape = (batch,).
          return_attentions: whether or not to return the attentions tensors of
            all attention layers.
          return_hid_states: whether or not to return the hidden states of all layers.
          chunk_length: chunk size in seconds.

        Returns:
          Dictionary with:
            last_hidden_state: sequence of hidden-states at the output of the last
                layer of the model (torch.FloatTensor of shape
                (batch_size, sequence_length, hidden_size)).
            extract_features: sequence of extracted feature vectors of the last
                convolutional layer of the model. (torch.FloatTensor of shape
                (batch_size, sequence_length, conv_dim[-1])
            hidden_states: hidden-states of the model at the output of each layer
                plus the initial embedding outputs (tuple(torch.FloatTensor)).
            attentions: Attentions weights after the attention softmax, used to
                compute the weighted average in the self-attention heads
                (tuple(torch.FloatTensor)).
        """
        max_in_length = x.size(-1)
        x, x_mask = self._preprocess(x, x_lengths)
        # we transform the chunk length from seconds to samples,
        # making sure that the chunk_length corresponds to an integer number of output samples.
        chunk_frames = max(
            1, int(chunk_length * self.sample_frequency) // self.frame_shift
        )
        chunk_length = chunk_frames * self.frame_shift
        num_chunks = (x.size(1) + chunk_length - 1) // chunk_length
        left_context, right_context = self.context
        max_out_length = self.max_out_length(x.size(1))
        start = 0
        outputs = []
        for i in range(num_chunks):
            if i < num_chunks - 1:
                start_i = max(start - left_context, 0)
            else:
                # last chunk has special treatment, we forward pass
                # a chunk with chunk_length size ending at the end.
                # but we will just use the output frames that don't overlap
                # with the second last chunk.
                start_i = max(x.size(1) - chunk_length - left_context, 0)

            stop_i = min(start + chunk_length + right_context, x.size(1))
            x_i = x[:, start_i:stop_i]
            x_mask_i = None if x_mask is None else x_mask[:, start_i:stop_i]
            output_i = self.hf_model(
                x_i,
                x_mask_i,
                output_attentions=return_attentions,
                output_hidden_states=return_hid_states,
            )

            if i < num_chunks - 1:
                start_out_i = max(
                    output_i.last_hidden_state.size(1)
                    - chunk_frames
                    - self.right_encoder_context,
                    0,
                )
                stop_out_i = start_out_i + chunk_frames
            else:
                # we just use the frames that do not overlap
                # with the second last chunk
                remaining_frames = max_out_length - i * chunk_frames
                start_out_i = -remaining_frames
                stop_out_i = output_i.last_hidden_state.size(1)

            output_i.last_hidden_state = output_i.last_hidden_state[
                :, start_out_i:stop_out_i
            ]
            if detach_chunks:
                output_i.last_hidden_state.detach_()

            if return_hid_states:
                output_i.hidden_states = [
                    h[:, start_out_i:stop_out_i] for h in output_i.hidden_states
                ]
                if detach_chunks:
                    output_i.hidden_states = [
                        h.detach() for h in output_i.hidden_states
                    ]

            outputs.append(output_i)
            start += chunk_length

        # concatenate outputs from different chunks
        output = outputs[0]
        output.last_hidden_state = torch.cat(
            [o.last_hidden_state for o in outputs], dim=1
        )
        if return_hid_states:
            hidden_states = []
            for j in range(len(outputs[0].hidden_states)):
                hidden_states_j = torch.cat(
                    [o.hidden_states[j] for o in outputs], dim=1
                )
                hidden_states.append(hidden_states_j)
            output.hidden_states = hidden_states

        if return_attentions:
            attentions = []
            for j in range(len(outputs[0].attentions)):
                attentions_j = [o.attentions[j] for o in outputs]
                attentions.append(attentions_j)
            output.attentions = attentions

        feat_lengths = (
            None
            if x_lengths is None
            else scale_seq_lengths(x_lengths, max_out_length, max_in_length)
        )
        output["hidden_states_lengths"] = feat_lengths
        return output

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Returns the configuration arguments for the object in a dictionary."""

        config = {
            "pretrained_model_path": self.pretrained_model_path,
            "normalize_input": self.normalize_input,
            "use_input_attention_mask": self.use_input_attention_mask,
            "cache_dir": self.cache_dir,
            "force_download": self.force_download,
            "revision": self.revision,
            "drop_layers_gt": self.drop_layers_gt,
            "ignore_pretrained": self.ignore_pretrained,
            "override_dropouts": self.override_dropouts,
            "override_spec_augment": self.override_spec_augment,
            "left_encoder_context": self.left_encoder_context,
            "right_encoder_context": self.right_encoder_context,
            "sample_frequency": self.sample_frequency,
            "feat_extract_lr": self.feat_extract_lr,
            "encoder_lr": self.encoder_lr,
            "use_lora": self.use_lora,
            "lora_components": self.lora_components,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_merge_weights": self.lora_merge_weights,
            "bias_weight_decay": self.bias_weight_decay,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def save(self, file_path: PathLike) -> None:
        """Saves the model to disk."""
        self.ignore_pretrained = True
        super().save(file_path)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword args to those accepted by `__init__`.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            A dictionary containing only valid constructor arguments.
        """
        return filter_func_args(HFWav2VecBase.__init__, kwargs)

    @staticmethod
    def _add_lr_args(parser: ArgumentParser) -> None:
        """Adds learning-rate related CLI arguments to a parser.

        Args:
            parser: Parser to update.

        Returns:
            None.
        """
        parser.add_argument(
            "--feat-extract-lr",
            default=None,
            type=float,
            help=(
                "lr for conv feature extractor, it serves to set a lr "
                "different than the global one."
            ),
        )
        parser.add_argument(
            "--encoder-lr",
            default=None,
            type=float,
            help=(
                "lr for transformer encoder, it serves to set a lr "
                "different than the global one."
            ),
        )

    @staticmethod
    def _add_lora_args(parser: ArgumentParser) -> None:
        """Adds LoRA-related CLI arguments to a parser.

        Args:
            parser: Parser to update.

        Returns:
            None.
        """
        parser.add_argument(
            "--use-lora",
            default=False,
            action=ActionYesNo,
            help="use low-rank adapters",
        )
        parser.add_argument(
            "--lora-components",
            default=["q_proj", "v_proj"],
            nargs="+",
            choices=[
                "k_proj",
                "q_proj",
                "v_proj",
                "out_proj",
                "intermediate_dense",
                "output_dense",
            ],
            help="list of components where we apply LoRA, e.g., [Wq, Wv]",
        )
        parser.add_argument("--lora-rank", default=4, help="rank of LoRA")
        parser.add_argument("--lora-alpha", default=8, type=int, help="scale for LoRA")
        parser.add_argument("--lora-dropout", default=0.0, help="dropout rate for LoRA")
        parser.add_argument(
            "--lora-merge-weights",
            default=True,
            action=ActionYesNo,
            help="lora weights are merged with the pretrained weights at inference.",
        )

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Adds class-construction CLI arguments.

        Args:
            parser: Parser to update.
            prefix: Optional prefix for nested parser registration.
            skip: Optional set of variable-style argument names to omit
                (for example, `pretrained_model_path`).

        Returns:
            None.
        """
        skip = set() if skip is None else set(skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if _use_arg("pretrained_model_path"):
            parser.add_argument(
                "--pretrained-model-path",
                default=None,
                help=("file path or HuggingFace Hub path to pre-trained model"),
            )

        if _use_arg("normalize_input"):
            parser.add_argument(
                "--normalize-input",
                default=True,
                action=ActionYesNo,
                help=("whether or not to zero-mean unit-variance normalize the input"),
            )
        if _use_arg("use_input_attention_mask"):
            parser.add_argument(
                "--use-input-attention-mask",
                default=False,
                action=ActionYesNo,
                help=("whether we should input an attention mask to the wav2vec model"),
            )
        if _use_arg("cache_dir"):
            parser.add_argument(
                "--cache-dir",
                default="./.cache/hyperion_hf",
                help=(
                    "path to a directory in which a downloaded pretrained model "
                    "configuration should be cached if the standard cache should not be used"
                ),
            )
        if _use_arg("force_download"):
            parser.add_argument(
                "--force-download",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether or not to force the (re-)download the model weights "
                    "and configuration files and override thecached versions if they exist"
                ),
            )
        if _use_arg("revision"):
            parser.add_argument(
                "--revision",
                default="main",
                help=(
                    "the specific model version to use. It can be a branch name, "
                    "a tag name, or a commit id. "
                ),
            )
        if _use_arg("drop_layers_gt"):
            parser.add_argument(
                "--drop-layers-gt",
                default=None,
                type=int,
                help=("drop encoder layers greater than this value."),
            )
        if _use_arg("override_dropouts"):
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        if _use_arg("override_spec_augment"):
            parser.add_argument(
                "--override-spec-augment",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the spec augment config. passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        if _use_arg("left_encoder_context"):
            parser.add_argument(
                "--left-encoder-context",
                default=16,
                type=int,
                help=(
                    "past context frames used by the transformer encoder "
                    "when the signal is evaluated chunk by chunk."
                ),
            )
        if _use_arg("right_encoder_context"):
            parser.add_argument(
                "--right-encoder-context",
                default=16,
                type=int,
                help=(
                    "future context frames used by the transformer encoder "
                    "when the signal is evaluated chunk by chunk."
                ),
            )

        if _use_arg("feat_extract_lr"):
            parser.add_argument(
                "--feat-extract-lr",
                default=None,
                type=float,
                help=(
                    "lr for conv feature extractor, it serves to set a lr "
                    "different than the global one."
                ),
            )
        if _use_arg("encoder_lr"):
            parser.add_argument(
                "--encoder-lr",
                default=None,
                type=float,
                help=(
                    "lr for transformer encoder, it serves to set a lr "
                    "different than the global one."
                ),
            )
        if _use_arg("use_lora"):
            parser.add_argument(
                "--use-lora",
                default=False,
                action=ActionYesNo,
                help="use low-rank adapters",
            )
        if _use_arg("lora_components"):
            parser.add_argument(
                "--lora-components",
                default=["q_proj", "v_proj"],
                nargs="+",
                choices=[
                    "k_proj",
                    "q_proj",
                    "v_proj",
                    "out_proj",
                    "intermediate_dense",
                    "output_dense",
                ],
                help="list of components where we apply LoRA, e.g., [Wq, Wv]",
            )
        if _use_arg("lora_rank"):
            parser.add_argument("--lora-rank", default=4, help="rank of LoRA")
        if _use_arg("lora_alpha"):
            parser.add_argument(
                "--lora-alpha", default=8, type=int, help="scale for LoRA"
            )
        if _use_arg("lora_dropout"):
            parser.add_argument(
                "--lora-dropout", default=0.0, help="dropout rate for LoRA"
            )
        if _use_arg("lora_merge_weights"):
            parser.add_argument(
                "--lora-merge-weights",
                default=True,
                action=ActionYesNo,
                help="lora weights are merged with the pretrained weights at inference.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword args to those accepted by `change_config`.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            A dictionary containing only valid finetuning arguments.
        """
        return filter_func_args(HFWav2VecBase.change_config, kwargs)

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Adds finetuning-time CLI arguments.

        Args:
            parser: Parser to update.
            prefix: Optional prefix for nested parser registration.
            skip: Optional set of variable-style argument names to omit
                (for example, `override_dropouts`).

        Returns:
            None.
        """
        skip = set() if skip is None else set(skip)

        def _use_arg(var_name: str) -> bool:
            return var_name not in skip

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if _use_arg("override_dropouts"):
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        if _use_arg("override_spec_augment"):
            parser.add_argument(
                "--override-spec-augment",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the spec augment config. passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        if _use_arg("override_lora"):
            parser.add_argument(
                "--override-lora",
                default=False,
                action=ActionYesNo,
                help=("whether to change the config of LoRA layers in the model."),
            )

        if _use_arg("feat_extract_lr"):
            parser.add_argument(
                "--feat-extract-lr",
                default=None,
                type=float,
                help=(
                    "lr for conv feature extractor, it serves to set a lr "
                    "different than the global one."
                ),
            )
        if _use_arg("encoder_lr"):
            parser.add_argument(
                "--encoder-lr",
                default=None,
                type=float,
                help=(
                    "lr for transformer encoder, it serves to set a lr "
                    "different than the global one."
                ),
            )
        if _use_arg("use_lora"):
            parser.add_argument(
                "--use-lora",
                default=False,
                action=ActionYesNo,
                help="use low-rank adapters",
            )
        if _use_arg("lora_components"):
            parser.add_argument(
                "--lora-components",
                default=["q_proj", "v_proj"],
                nargs="+",
                choices=[
                    "k_proj",
                    "q_proj",
                    "v_proj",
                    "out_proj",
                    "intermediate_dense",
                    "output_dense",
                ],
                help="list of components where we apply LoRA, e.g., [Wq, Wv]",
            )
        if _use_arg("lora_rank"):
            parser.add_argument("--lora-rank", default=4, help="rank of LoRA")
        if _use_arg("lora_alpha"):
            parser.add_argument(
                "--lora-alpha", default=8, type=int, help="scale for LoRA"
            )
        if _use_arg("lora_dropout"):
            parser.add_argument(
                "--lora-dropout", default=0.0, help="dropout rate for LoRA"
            )
        if _use_arg("lora_merge_weights"):
            parser.add_argument(
                "--lora-merge-weights",
                default=True,
                action=ActionYesNo,
                help="lora weights are merged with the pretrained weights at inference.",
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
