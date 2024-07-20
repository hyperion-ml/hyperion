"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from turtle import right
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from ....utils.misc import filter_func_args
from ...layers import LoRAFactory
from ...torch_model import TorchModel
from ...utils import scale_seq_lengths, seq_lengths_to_mask
from ...utils.ddp import ddp_get_rank, ddp_wait_for_all_procs


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
        left_encoder_context (`int`): past context frames used by the transformer encoder when the signal is evaluated
          chunk by chunk, if it is too long to fit in GPU.
        right_encoder_context: (`int`): future context frames used by the transformer encoder.
        sample_frequency: (`int`) waveform sample frequency used to train the model.
        feat_extract_lr: learning rate for conv feature extractor, serves to set a lr different than the global one.
        encoder_lr: learning rate for the wav2vec encoder, serves to set a lr different than the global one.
        use_lora: use low-rank adapters
        lora_components: list of components where we apply LoRA, eg [Wq, Wv]
        lora_rank: rank of LoRA
        lora_alpha: scale for LoRA
        lora_dropout: dropout rate for LoRA
        lora_merge_weights: lora weights are merged with the pretrained weights at inference.
    """

    def __init__(
        self,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None,
        normalize_input: bool = True,
        use_input_attention_mask: bool = False,
        cache_dir: Union[str, os.PathLike] = "./.cache/hyperion_hf",
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
        lora_components: List[str] = ["q_proj", "v_proj"],
        lora_rank: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
        resume_download: bool = False,  # deprecated
    ):
        super().__init__()
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
                except:
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
                except:
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

    @property
    def feature_encoder_context(self):
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
    def frame_shift(self):
        if self._frame_shift is not None:
            return self._frame_shift

        total_stride = 1
        for stride in self.hf_model.config.conv_stride:
            total_stride *= stride

        self._frame_shift = total_stride
        return total_stride

    @property
    def context(self):
        left, right = self.feature_encoder_context
        left += self.left_encoder_context
        right += self.right_encoder_context
        return left, right

    def max_out_length(self, max_in_length):
        return self.hf_model._get_feat_extract_output_lengths(max_in_length).item()
        # left_context, right_context = self.feature_encoder_context
        # max_in_length = max_in_length - left_context - right_context
        # return max_in_length // self.frame_shift

    def out_lengths(self, in_lengths):
        return self.hf_model._get_feat_extract_output_lengths(in_lengths)
        # left_context, right_context = self.feature_encoder_context
        # in_lengths = in_lengths - left_context - right_context
        # return torch.div(in_lengths, self.frame_shift, rounding_mode="floor")

    def out_shape(self, in_shape):
        out_length = self.max_out_length(in_shape[1])
        C = self.hf_model.config.hidden_size
        return (in_shape[0], out_length, C)

    def change_config(
        self,
        override_dropouts: bool,
        override_spec_augment: bool,
        override_lora: bool = False,
        feat_extract_lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        use_lora: bool = False,
        lora_components: List[str] = ["q_proj", "v_proj"],
        lora_rank: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
        **kwargs,
    ):
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
        **kwargs,
    ):
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
        lora_components: List[str] = ["q_proj", "v_proj"],
        lora_rank: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_merge_weights: bool = True,
    ):
        if not self.use_lora:
            if use_lora:
                self._make_lora_layers(
                    lora_components,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                    lora_merge_weights,
                )
                pass
            else:
                # TODO
                pass
        else:
            if use_lora:
                # TODO
                pass
            else:
                # TODO
                pass

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
        lora_merge_weights: bool,
    ):
        counts = {k: 0 for k in lora_components}
        self._recursive_replace_layer_by_lora(
            self.hf_model,
            counts,
            lora_components,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_merge_weights,
        )
        for k, v in counts.items():
            logging.info("count of LoRA layers for %s = %d", k, v)
            assert v > 0, f"did not make any {k} LoRA"

    @staticmethod
    def _recursive_replace_layer_by_lora(
        model: nn.Module,
        counts: dict,
        lora_components: List[str],
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_merge_weights: bool,
    ):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                HFWav2VecBase._recursive_replace_layer_by_lora(
                    module,
                    counts,
                    lora_components,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                    lora_merge_weights,
                )
            if isinstance(module, nn.Linear) and name in lora_components:
                lora_layer = LoRAFactory.create_from_pretrained(
                    module,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                setattr(model, name, lora_layer)
                counts[name] += 1

    def change_dropouts(self, **kwargs):
        pass  # needs to be overloaded

    def freeze_feature_encoder(self):
        self.hf_model.freeze_feature_encoder()

    def freeze_except_lora(self, bias=None):
        bias = "none" if bias is None else bias
        from ...layers.lora import mark_only_lora_as_trainable

        mark_only_lora_as_trainable(self.hf_model, bias=bias)

    def has_param_groups(self):
        return self.feat_extract_lr is not None or self.encoder_lr is not None

    def trainable_param_groups(self):
        if not self.has_param_groups():
            return self.trainable_parameters()

        if self.feat_extract_lr == self.encoder_lr:
            return [{"params": self.trainable_parameters(), "lr": self.encoder_lr}]

        param_groups = [
            {"params": self.hf_model.feature_extractor.parameters()},
            {"params": self.hf_model.feature_projection.parameters()},
            {"params": self.hf_model.encoder.parameters()},
        ]
        if self.hf_model.adapter is not None:
            param_groups.append({"params": self.hf_model.adapter.parameters()})

        if self.feat_extract_lr is not None:
            param_groups[0]["lr"] = self.feat_extract_lr
            param_groups[1]["lr"] = self.feat_extract_lr

        if self.encoder_lr is not None:
            param_groups[2]["lr"] = self.encoder_lr
            if len(param_groups) == 4:
                param_groups[3]["lr"] = self.encoder_lr

        return param_groups

    @property
    def hf_config(self):
        return self.hf_model.config

    def _normalize(self, x, x_mask=None):
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

    def _preprocess(self, x, x_lengths=None):
        """Prepares input audio to be used as input to wav2vec style model."""
        x_mask = seq_lengths_to_mask(x_lengths, x.size(-1), dtype=torch.long)
        if self.normalize_input:
            x = self._normalize(x, x_mask)

        if self.use_input_attention_mask:
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
    ):
        r"""Forward function for long utterances that do not fit in GPU memory.

        Args:
          x: input audio of shape = (batch, sequence_length).
          x_lengths: lengths of the audio waveforms in samples with shape = (batch,).
          return_attentions: whether or not to return the attentions tensors of
            all attention layers.
          return_hid_states: whether or not to return the hidden states of all layers.
          chunk_size: chunk size in seconds.

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
        # if ddp_get_rank() == 0:
        #     lora_layer = self.hf_model.encoder.layers[0].attention.v_proj
        # print(
        #     "lora\nw=",
        #     lora_layer.weight[:3, :3],
        #     "\na=",
        #     lora_layer.lora_A[:3, :3],
        #     "\nb=",
        #     lora_layer.lora_B[:3, :3],
        #     "\n",
        #     "merged=",
        #     lora_layer.merged,
        #     "training=",
        #     lora_layer.training,
        #     flush=True,
        # )
        # assert self.training == lora_layer.training
        # assert self.training == (not lora_layer.merged)
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
    ):
        r"""Forward function for long utterances that do not fit in GPU memory.

        Args:
          x: input audio of shape = (batch, sequence_length).
          x_lengths: lengths of the audio waveforms in samples with shape = (batch,).
          return_attentions: whether or not to return the attentions tensors of
            all attention layers.
          return_hid_states: whether or not to return the hidden states of all layers.
          chunk_size: chunk size in seconds.

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
        # output0 = self.forward_impl(x, x_lengths)
        # mol0 = output0.last_hidden_state.size(1)
        print("long", flush=True)
        max_in_length = x.size(-1)
        x, x_mask = self._preprocess(x, x_lengths)
        # we transform the chunk length from seconds to samples,
        # making sure that the chunk_length corresponds to an integer number of output samples.
        chunk_frames = int(chunk_length * self.sample_frequency) // self.frame_shift
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
            x_mask_i = None if x_mask is None else x_mask[start_i:stop_i]
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

    def get_config(self):
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
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save(self, file_path: str):
        """Saves the model to disk."""
        self.ignore_pretrained = True
        self.save(file_path)

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(HFWav2VecBase.__init__, kwargs)

    @staticmethod
    def _add_lr_args(parser):
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
    def _add_lora_args(parser):
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
            help="list of components where we apply LoRA, eg [Wq, Wv]",
        )
        parser.add_argument("--lora-rank", default=4, help="rank of LoRA")
        parser.add_argument("--lora-alpha", default=1.0, help="scale for LoRA")
        parser.add_argument("--lora-dropout", default=0.0, help="dropout rate for LoRA")
        parser.add_argument(
            "--lora-merge-weights",
            default=True,
            action=ActionYesNo,
            help="lora weights are merged with the pretrained weights at inference.",
        )

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
        # parser.add_argument(
        #     "--resume-download",
        #     default=False,
        #     action=ActionYesNo,
        #     help=(
        #         "whether or not to delete incompletely received files. "
        #         "Will attempt to resume the download if such a file exists"
        #     ),
        # )
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
        parser.add_argument(
            "--left-encoder-context",
            default=16,
            type=int,
            help=(
                "past context frames used by the transformer encoder "
                "when the signal is evaluated chunk by chunk."
            ),
        )
        parser.add_argument(
            "--right-encoder-context",
            default=16,
            type=int,
            help=(
                "future context frames used by the transformer encoder "
                "when the signal is evaluated chunk by chunk."
            ),
        )

        HFWav2VecBase._add_lr_args(parser)
        HFWav2VecBase._add_lora_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        return filter_func_args(HFWav2VecBase.change_config, kwargs)
        # valid_args = (
        #     "override_dropouts",
        #     "override_spec_augment",
        # )
        # args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        # return args

    @staticmethod
    def add_finetune_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

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
        parser.add_argument(
            "--override-lora",
            default=False,
            action=ActionYesNo,
            help=("whether to change the config of LoRA layers in the model."),
        )

        HFWav2VecBase._add_lr_args(parser)
        HFWav2VecBase._add_lora_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
