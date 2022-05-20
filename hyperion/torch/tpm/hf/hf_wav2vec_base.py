"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import logging
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

from ...torch_model import TorchModel
from ...utils import seq_lengths_to_mask, scale_seq_lengths
from ...utils.ddp import ddp_wait_for_all_procs, ddp_get_rank


class HFWav2VecBase(TorchModel):
    """Base class for Wav2Vec style models (Wav2Vec2, Hubert, WavLM, ...) in HuggingFace.

    This class includes the proprocessing steps, common to all models.

    Attributes:
        pretrained_model_path (`str`, or os.PathLike, defaults to None): file path or
            HuggingFace Hub path to pre-trained model.
        normalize_input (`bool`, defaults to True): whether or not to zero-mean unit-variance
            normalize the input.
        use_input_attention_mask (`bool`, defaults to False): whether we should input an
            attention mask to the wav2vec model.
        cache_dir (str or os.PathLike): path to a directory in which a downloaded pretrained
            model configuration should be cached if the standard cache should not be used.
        force_download (`bool`, defaults to `False`): whether or not to force the (re-)download
            the model weights and configuration files and override the
            cached versions if they exist.
        resume_download (`bool`, defaults to `False`): whether or not to delete incompletely
            received files. Will attempt to resume the download if such a file exists.
        revision(`str`, defaults to `"main"`): the specific model version to use.
            It can be a branch name, a tag name, or a commit id.
        drop_layers_gt (`int` defaults to None): drop encoder layers greater than this value (in [1, num_encoder_layers]).
            If None, the model is not changed.
        ignore_pretrained (`bool` defaults to False): if True, it ignores the pretrained_model_path
            and inits the model from the configuration. This is set to True for models that have already
            been finetuned.
        override_dropouts (`bool` defaults to False): if True, it ingnores the dropout probs. in the pretrained model
            and uses the ones passed as arguments.
        override_spec_augment (`bool` defaults to False): if True, it ingnores the spec. augment.
            configuration in the pretrained model and uses the ones passed in the arguments.
    """

    def __init__(
        self,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None,
        normalize_input: bool = True,
        use_input_attention_mask: bool = False,
        cache_dir: Union[str, os.PathLike] = "./.cache/hyperion_hf",
        force_download: bool = False,
        resume_download: bool = False,
        revision: str = "main",
        drop_layers_gt: Optional[int] = None,
        ignore_pretrained: bool = False,
        override_dropouts: bool = False,
        override_spec_augment: bool = False,
    ):
        super().__init__()
        self.pretrained_model_path = pretrained_model_path
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.resume_download = resume_download
        self.revision = revision
        self.drop_layers_gt = drop_layers_gt
        self.ignore_pretrained = ignore_pretrained
        self.override_dropouts = override_dropouts
        self.override_spec_augment = override_spec_augment

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
                        resume_download=resume_download,
                        revision=revision,
                    )
                except:
                    # if fails, we try to download full processor config
                    processor = Wav2Vec2Processor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
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
                        resume_download=False,
                        revision=revision,
                    )
                except:
                    # if fails, we try to download full processor config
                    processor = Wav2Vec2Processor.from_pretrained(
                        pretrained_model_path,
                        cache_dir=cache_dir,
                        force_download=False,
                        resume_download=False,
                        revision=revision,
                    )
                    feature_extractor = processor.feature_extractor

            ddp_wait_for_all_procs()
            normalize_input = feature_extractor.do_normalize
            use_input_attention_mask = feature_extractor.return_attention_mask

        self.normalize_input = normalize_input
        self.use_input_attention_mask = use_input_attention_mask

    def __deepcopy__(self, memo):
        """Reimplementation of deepcopy for Hugging Face models.
        The weight_norm in the Conv. Pos. Encoder of Wav2Vec models make the default deepcopy to fail.
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

    def change_hyperparams(self, **kwargs):
        if self.override_spec_augment:
            self.change_spec_augment(**kwargs)

        if self.override_dropouts:
            self.change_dropouts(**kwargs)

    def change_spec_augment(
        self,
        apply_spec_augment: bool = True,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        **kwargs,
    ):
        self.hf_model.config.apply_spec_augment = apply_spec_augment
        self.hf_model.config.mask_time_prob = mask_time_prob
        self.hf_model.config.mask_time_length = mask_time_length
        self.hf_model.config.mask_time_min_masks = mask_time_min_masks
        self.hf_model.config.mask_feature_prob = mask_feature_prob
        self.hf_model.config.mask_feature_length = mask_feature_length
        self.hf_model.config.mask_feature_min_masks = mask_feature_min_masks

    def change_dropouts(self, **kwargs):
        pass  # needs to be overloaded

    def freeze_feature_encoder(self):
        self.hf_model.freeze_feature_encoder()

    @property
    def hf_config(self):
        return self.hf_model.config

    def _normalize(self, x, x_mask=None):
        """Normalizes the audio to have zero mean and unit variance."""
        if x_mask is None:
            x = x - x.mean(dim=1, keepdim=True)
            std = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-7)
            x = x / std
        else:
            x_mask = x_mask.to(dtype=x.dtype)
            x_samples = torch.mean(x_mask, dim=1, keepdim=True)
            x_mean = torch.mean(x * x_mask, dim=1, keepdim=True) / x_samples
            x2_mean = torch.mean(x ** 2 * x_mask, dim=1, keepdim=True) / x_samples
            std = torch.sqrt(x2_mean - x_mean ** 2 + 1e-7)
            x = (x - x_mean) / std
        return x

    def _preprocess(self, x, x_lengths=None):
        """Prepares input audio to be used as input to wav2vec style model."""
        x_mask = seq_lengths_to_mask(x_lengths, x.size(-1), dtype=torch.long)
        if self.normalize_input:
            x = self._normalize(x, x_lengths)

        if self.use_input_attention_mask:
            x_mask = None

        return x, x_mask

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.LongTensor] = None,
        return_attentions: bool = False,
        return_hid_states: bool = False,
    ):
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

    def get_config(self):
        """Returns the configuration arguments for the object in a dictionary."""

        config = {
            "pretrained_model_path": self.pretrained_model_path,
            "normalize_input": self.normalize_input,
            "use_input_attention_mask": self.use_input_attention_mask,
            "cache_dir": self.cache_dir,
            "force_download": self.force_download,
            "resume_download": self.resume_download,
            "revision": self.revision,
            "drop_layers_gt": self.drop_layers_gt,
            "ignore_pretrained": self.ignore_pretrained,
            "override_dropouts": self.override_dropouts,
            "override_spec_augment": self.override_spec_augment,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save(self, file_path: str):
        """Saves the model to disk."""
        self.ignore_pretrained = True
        self.save(file_path)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "pretrained_model_path",
            "normalize_input",
            "use_input_attention_mask",
            "cache_dir",
            "force_download",
            "resume_download",
            "revision",
            "drop_layers_gt",
            "ignore_pretrained",
            "override_dropouts",
            "override_spec_augment",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--pretrained-model-path",
            default=None,
            help=("file path or HuggingFace Hub path to pre-trained model"),
        )
        parser.add_argument(
            "--normalize-input",
            default=True,
            action=ActionYesNo,
            help=("whether or not to zero-mean unit-variance normalize the input"),
        )
        parser.add_argument(
            "--use-input-attention-mask",
            default=False,
            action=ActionYesNo,
            help=("whether we should input an attention mask to the wav2vec model"),
        )
        parser.add_argument(
            "--cache-dir",
            default="./.cache/hyperion_hf",
            help=(
                "path to a directory in which a downloaded pretrained model "
                "configuration should be cached if the standard cache should not be used"
            ),
        )
        parser.add_argument(
            "--force-download",
            default=False,
            action=ActionYesNo,
            help=(
                "whether or not to force the (re-)download the model weights "
                "and configuration files and override thecached versions if they exist"
            ),
        )
        parser.add_argument(
            "--resume-download",
            default=False,
            action=ActionYesNo,
            help=(
                "whether or not to delete incompletely received files. "
                "Will attempt to resume the download if such a file exists"
            ),
        )
        parser.add_argument(
            "--revision",
            default="main",
            help=(
                "the specific model version to use. It can be a branch name, "
                "a tag name, or a commit id. "
            ),
        )
        parser.add_argument(
            "--drop-layers-gt",
            default=None,
            type=int,
            help=("drop encoder layers greater than this value."),
        )
        parser.add_argument(
            "--override-dropouts",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to use the dropout probabilities passed in the "
                "arguments instead of the defaults in the pretrained model."
            ),
        )
        parser.add_argument(
            "--override-spec-augment",
            default=False,
            action=ActionYesNo,
            help=(
                "whether to use the spec augment config. passed in the "
                "arguments instead of the defaults in the pretrained model."
            ),
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
