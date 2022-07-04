"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import logging
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
from typing import Optional, Tuple, Union, List, Callable

import torch
import torch.nn as nn

from transformers import WavLMModel, WavLMConfig

from ...utils.ddp import ddp_wait_for_all_procs, ddp_get_rank
from .hf_wav2vec_base import HFWav2VecBase


class HFWavLM(HFWav2VecBase):
    r"""This is wrapper over HuggingFace WavLM model.
        See documentation: https://huggingface.co/docs/transformers/model_doc/wavlm

        This wrapper makes the HugginFace model to have the same interface
        as other hyperion models. It also add extra functionalities.

        The config. parameters are the same as in the HuggingFace WavLMConfig class.

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
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, defaults to 10):
            length of vector span along the time axis.
        mask_time_min_masks (`int`, defaults to 2),:
            the minimum number of masks of length `mask_time_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, defaults to 0.0):
            percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, defaults to 10):
            length of vector span along the feature axis.
        mask_feature_min_masks (`int`, defaults to 0):
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        add_adapter (`bool`, defaults to `False`):
            whether a convolutional network should be stacked on top of the WavLM Encoder. Can be very useful for
            warm-starting WavLM for SpeechEncoderDecoder models.
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
        cache_dir (str or os.PathLike): path to a directory in which a downloaded pretrained
            model configuration should be cached if the standard cache should not be used.
        force_download (`bool`, defaults to `False`): whether or not to force the (re-)download
            the model weights and configuration files and override the
            cached versions if they exist.
        resume_download (`bool`, defaults to `False`): whether or not to delete incompletely
            received files. Will attempt to resume the download if such a file exists.
        revision(`str`, defaults to `"main"`): the specific model version to use.
            It can be a branch name, a tag name, or a commit id.
        ignore_pretrained (`bool` defaults to False): if True, it ignores the pretrained_model_path
            and inits the model from the configuration. This is set to True for models that have already
            been finetuned.
        override_dropouts (`bool` defaults to False): if True, it ingnores the dropout probs. in the pretrained model
            and uses the ones passed as arguments.
        override_spec_augment (`bool` defaults to False): if True, it ingnores the spec. augment.
            configuration in the pretrained model and uses the ones passed in the arguments.
        left_encoder_context (`int`): past context frames used by the transformer encoder when the signal is evaluated
          chunk by chunk, if it is too long to fit in GPU.
        right_encoder_context: (`int`): future context frames used by the transformer encoder.
        sample_frequency: (`int`) waveform sample frequency used to train the model.
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
        add_adapter: bool = False,
        adapter_kernel_size: int = 3,
        adapter_stride: int = 2,
        num_adapter_layers: int = 3,
        output_hidden_size: Optional[int] = None,
        cache_dir: Union[str, os.PathLike] = "./.cache/hyperion_hf",
        force_download: bool = False,
        resume_download: bool = False,
        revision: str = "main",
        drop_layers_gt: Optional[int] = None,
        ignore_pretrained: bool = False,
        override_dropouts: bool = False,
        override_spec_augment: bool = False,
        left_encoder_context: int = 16,
        right_encoder_context: int = 16,
        sample_frequency: int = 16000,
    ):

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
        )

        if pretrained_model_path is not None and not ignore_pretrained:
            rank = ddp_get_rank()
            if rank == 0:
                logging.info(f"Downloading HF model from {pretrained_model_path}")
                # rank 0 downloads the model from HF web
                self.hf_model = WavLMModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    revision=revision,
                )
            # all ranks wait until the model is downloaded
            ddp_wait_for_all_procs()
            if rank > 0:
                # the rest of ranks should read the configuration from the cache.
                self.hf_model = WavLMModel.from_pretrained(
                    pretrained_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    resume_download=False,
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
            hf_config = WavLMConfig(
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
                add_adapter=add_adapter,
                adapter_kernel_size=adapter_kernel_size,
                adapter_stride=adapter_stride,
                num_adapter_layers=num_adapter_layers,
                output_hidden_size=output_hidden_size,
            )
            self.hf_model = WavLMModel(hf_config)

        if drop_layers_gt is not None:
            self.drop_upper_layers(drop_layers_gt)

        self.ignore_pretrained = True

    @property
    def num_encoder_layers(self):
        return self.hf_config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.hf_config.hidden_size

    def drop_upper_layers(self, max_layers: int):
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

    def get_config(self):
        """Returns the configuration arguments for the object in a dictionary."""
        config = self.hf_model.config.to_dict()
        config = self.filter_args(**config)
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
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
            "add_adapter",
            "adapter_kernel_size",
            "adapter_stride",
            "num_adapter_layers",
            "output_hidden_size",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_base)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_class_args(parser)

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
        parser.add_argument(
            "--hidden-size",
            default=768,
            type=int,
            help=("dimensionality of the encoder layers and the pooler layer."),
        )
        parser.add_argument(
            "--num-hidden-layers",
            default=12,
            type=int,
            help=("number of hidden layers in the Transformer encoder"),
        )
        parser.add_argument(
            "--num-attention-heads",
            default=12,
            type=int,
            help=(
                "number of attention heads for "
                "each attention layer in the Transformer encoder"
            ),
        )
        parser.add_argument(
            "--intermediate-size",
            default=3072,
            type=int,
            help=(
                "dimensionality of the " "feed-forward layer in the Transformer encoder"
            ),
        )
        parser.add_argument(
            "--hidden-act",
            default="gelu",
            choices=["gelu", "relu", "selu", "gelu_new"],
            help=(
                "the non-linear "
                "activation function (function or string) in the encoder and pooler"
            ),
        )
        parser.add_argument(
            "--hidden-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "fully connected layers in the embeddings, encoder, and pooler"
            ),
        )
        parser.add_argument(
            "--activation-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "intermediate layer in feedforward transformer layers"
            ),
        )
        parser.add_argument(
            "--attention-dropout",
            default=0.1,
            type=float,
            help=("the dropout ratio for the attention probabilities"),
        )
        parser.add_argument(
            "--layerdrop",
            default=0.1,
            type=float,
            help=("prob. of dropping a layer"),
        )
        parser.add_argument(
            "--initializer-range",
            default=0.02,
            type=float,
            help=(
                "the standard deviation of the "
                "truncated_normal_initializer for initializing all weight matrices"
            ),
        )
        parser.add_argument(
            "--layer-norm-eps",
            default=1e-12,
            type=float,
            help=(
                "the standard deviation of the "
                "truncated_normal_initializer for initializing all weight matrices"
            ),
        )
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
        parser.add_argument(
            "--feat-proj-dropout",
            default=0.1,
            type=float,
            help=("the dropout probability for output of the feature encoder"),
        )
        parser.add_argument(
            "--feat-extract-activation",
            default="gelu",
            choices=["gelu", "relu", "selu", "gelu_new"],
            help=(
                "the non-linear activation function (function or string) in the 1D "
                "convolutional layers of the feature extractor"
            ),
        )
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
        parser.add_argument(
            "--conv-stride",
            default=[5, 2, 2, 2, 2, 2, 2],
            nargs="+",
            type=int,
            help=(
                "a tuple of integers defining the stride of each 1D convolutional layer in the feature encoder"
            ),
        )
        parser.add_argument(
            "--conv-kernel",
            default=[10, 3, 3, 3, 3, 3, 3],
            nargs="+",
            type=int,
            help=(
                "a tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder"
            ),
        )
        parser.add_argument(
            "--conv-bias",
            default=False,
            action=ActionYesNo,
            help=("whether the 1D convolutional layers have a bias"),
        )
        parser.add_argument(
            "--num-conv-pos-embeddings",
            default=128,
            type=int,
            help=(
                "number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional "
                "embeddings layer"
            ),
        )
        parser.add_argument(
            "--num-conv-pos-embedding-groups",
            default=16,
            type=int,
            help=("number of groups of 1D convolutional positional embeddings layer"),
        )
        parser.add_argument(
            "--do-stable-layer-norm",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to apply *stable* layer norm architecture of the Transformer encoder"
            ),
        )
        parser.add_argument(
            "--apply-spec-augment",
            default=True,
            action=ActionYesNo,
            help=(
                "whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder"
            ),
        )
        parser.add_argument(
            "--mask-time-prob",
            default=0.05,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the time axis which will be masked"
            ),
        )
        parser.add_argument(
            "--mask-time-length",
            default=10,
            type=int,
            help=("length of vector span along the time axis"),
        )
        parser.add_argument(
            "--mask-time-min-masks",
            default=2,
            type=int,
            help=(
                "the minimum number of masks of length `mask_time_length` generated along the time axis"
            ),
        )
        parser.add_argument(
            "--mask-feature-prob",
            default=0.0,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked"
            ),
        )
        parser.add_argument(
            "--mask-feature-length",
            default=10,
            type=int,
            help=(" length of vector span along the feature axis"),
        )
        parser.add_argument(
            "--mask-feature-min-masks",
            default=0,
            type=int,
            help=(
                "The minimum number of masks of length `mask_feature_length` generated along the feature axis"
            ),
        )
        parser.add_argument(
            "--add-adapter",
            default=False,
            action=ActionYesNo,
            help=(
                "whether a convolutional network should be stacked on top of the WavLM Encoder"
            ),
        )
        parser.add_argument(
            "--adapter-kernel-size",
            default=3,
            type=int,
            help=("kernel size of the convolutional layers in the adapter network"),
        )
        parser.add_argument(
            "--adapter-stride",
            default=2,
            type=int,
            help=("stride of the convolutional layers in the adapter network"),
        )
        parser.add_argument(
            "--num-adapter-layers",
            default=3,
            type=int,
            help=(
                "number of convolutional layers that should be used in the adapter network"
            ),
        )
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
    def filter_finetune_args(**kwargs):
        args_base = HFWav2VecBase.filter_args(**kwargs)
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
    def add_finetune_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        HFWav2VecBase.add_finetune_args(parser)
        parser.add_argument(
            "--hidden-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "fully connected layers in the embeddings, encoder, and pooler"
            ),
        )
        parser.add_argument(
            "--activation-dropout",
            default=0.1,
            type=float,
            help=(
                "the dropout probability for all "
                "intermediate layer in feedforward transformer layers"
            ),
        )
        parser.add_argument(
            "--attention-dropout",
            default=0.1,
            type=float,
            help=("the dropout ratio for the attention probabilities"),
        )
        parser.add_argument(
            "--apply-spec-augment",
            default=True,
            action=ActionYesNo,
            help=(
                "whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder"
            ),
        )
        parser.add_argument(
            "--mask-time-prob",
            default=0.05,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the time axis which will be masked"
            ),
        )
        parser.add_argument(
            "--mask-time-length",
            default=10,
            type=int,
            help=("length of vector span along the time axis"),
        )
        parser.add_argument(
            "--mask-time-min-masks",
            default=2,
            type=int,
            help=(
                "the minimum number of masks of length `mask_time_length` generated along the time axis"
            ),
        )
        parser.add_argument(
            "--mask-feature-prob",
            default=0.0,
            type=float,
            help=(
                "percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked"
            ),
        )
        parser.add_argument(
            "--mask-feature-length",
            default=10,
            type=int,
            help=(" length of vector span along the feature axis"),
        )
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
