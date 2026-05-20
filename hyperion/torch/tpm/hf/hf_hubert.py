"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import HubertConfig, HubertModel

from ...utils.ddp import ddp_get_rank, ddp_wait_for_all_procs
from .hf_wav2vec_base import HFWav2VecBase


class HFHubert(HFWav2VecBase):
    r"""This is wrapper over HuggingFace Hubert model.
        See documentation: https://huggingface.co/docs/transformers/main/en/model_doc/hubert

        This wrapper makes the HuggingFace model to have the same interface
        as other hyperion models. It also adds extra functionalities.

        The config. parameters are the same as in the HuggingFace HubertConfig class.

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
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layerdrop: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        feat_extract_norm: str = "group",
        feat_proj_dropout: float = 0.0,
        feat_extract_activation: Union[str, Callable] = "gelu",
        conv_dim: Tuple[int] = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: Tuple[int] = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: Tuple[int] = (10, 3, 3, 3, 3, 3, 3),
        conv_bias: bool = False,
        num_conv_pos_embeddings: int = 128,
        num_conv_pos_embedding_groups: int = 16,
        do_stable_layer_norm: bool = False,
        apply_spec_augment: bool = True,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
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
        """Initializes the HuggingFace HuBERT wrapper.

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
            ignore_pretrained=ignore_pretrained,
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

        if pretrained_model_path is not None and not ignore_pretrained:
            rank = ddp_get_rank()
            if rank == 0:
                logging.info(f"Downloading HF model from {pretrained_model_path}")
                # rank 0 downloads the model from HF web
                self.hf_model = HubertModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision,
                )
            # all ranks wait until the model is downloaded
            ddp_wait_for_all_procs()
            if rank > 0:
                # the rest of ranks should read the configuration from the cache.
                self.hf_model = HubertModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    revision=revision,
                )
            ddp_wait_for_all_procs()
            self.hf_model.config.layerdrop = 0.0
            self.change_config(
                override_dropouts=self.override_dropouts,
                override_spec_augment=self.override_spec_augment,
                hidden_dropout=hidden_dropout,
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                feat_proj_dropout=feat_proj_dropout,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
            )
        else:
            hf_config = HubertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout=hidden_dropout,
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                feat_proj_dropout=feat_proj_dropout,
                layerdrop=0.0,  # layerdrop,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                feat_extract_norm=feat_extract_norm,
                feat_extract_activation=feat_extract_activation,
                conv_dim=conv_dim,
                conv_stride=conv_stride,
                conv_kernel=conv_kernel,
                conv_bias=conv_bias,
                num_conv_pos_embeddings=num_conv_pos_embeddings,
                num_conv_pos_embedding_groups=num_conv_pos_embedding_groups,
                do_stable_layer_norm=do_stable_layer_norm,
                apply_spec_augment=apply_spec_augment,
                mask_time_prob=mask_time_prob,
                mask_time_length=mask_time_length,
                mask_time_min_masks=mask_time_min_masks,
                mask_feature_prob=mask_feature_prob,
                mask_feature_length=mask_feature_length,
                mask_feature_min_masks=mask_feature_min_masks,
            )
            self.hf_model = HubertModel(hf_config)

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
        feat_proj_dropout: float = 0.1,
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
        import transformers.models.hubert.modeling_hubert as t

        self.hf_model.config.hidden_dropout = hidden_dropout
        self.hf_model.config.activation_dropout = activation_dropout
        self.hf_model.config.attention_dropout = attention_dropout
        self.hf_model.config.feat_proj_dropout = feat_proj_dropout

        self.hf_model.feature_projection.dropout.p = feat_proj_dropout
        for module in self.hf_model.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = hidden_dropout

        for module in self.hf_model.encoder.modules():
            if isinstance(module, t.HubertAttention):
                module.dropout = attention_dropout
            if isinstance(module, t.HubertFeedForward):
                module.intermediate_dropout.p = activation_dropout

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

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Returns the configuration arguments for the object in a dictionary."""
        config = self.hf_model.config.to_dict()
        config = self.filter_args(**config)
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
            "hidden_act",
            "hidden_dropout",
            "activation_dropout",
            "attention_dropout",
            "feat_proj_dropout",
            "layerdrop",
            "initializer_range",
            "layer_norm_eps",
            "feat_extract_norm",
            "feat_extract_activation",
            "conv_dim",
            "conv_stride",
            "conv_kernel",
            "conv_bias",
            "num_conv_pos_embeddings",
            "num_conv_pos_embedding_groups",
            "do_stable_layer_norm",
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
