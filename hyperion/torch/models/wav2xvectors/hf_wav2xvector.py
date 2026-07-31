"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...hyper_torch_model import HyperTorchModel
from ...narchs import FeatFuserMVN
from ...utils import collate_seqs_1d, collate_seqs_2d, remove_silence


class HFWav2XVector(HyperTorchModel):
    """Abstract Base class for x-vector models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       feat_fuser: Dictionary to build feature fuser object.
       xvector: x-vector model object.
       feat_fusion_start: the input to x-vector model will fuse the wav2vec layers from "feat_fusion_start" to
                          the wav2vec "num_layers".
    """

    def __init__(
        self,
        hf_feats: nn.Module,
        feat_fuser: Dict[str, Any],
        xvector: nn.Module,
        feat_fusion_start: int = 0,
    ) -> None:
        """Initializes the HF-based x-vector wrapper.

        Args:
          hf_feats: Hugging Face feature extractor module.
          feat_fuser: Configuration dictionary for ``FeatFuserMVN``.
          xvector: Backend x-vector model that consumes fused features.
          feat_fusion_start: First HF layer index used by the feature fuser.

        Returns:
          None.
        """
        super().__init__()
        self.hf_feats = hf_feats
        self.xvector = xvector
        self.feat_fusion_start = feat_fusion_start
        self._hf_context = contextlib.nullcontext()
        self._make_fuser(feat_fuser)

    def _make_fuser(self, feat_fuser: Dict[str, Any]) -> None:
        """Builds the feature-fusion module based on HF extractor dimensions.

        Args:
          feat_fuser: Configuration dictionary for ``FeatFuserMVN``.

        Returns:
          None.
        """
        num_feats = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        feat_dim = self.hf_feats.hidden_size
        feat_fuser["feat_fuser"]["num_feats"] = num_feats
        feat_fuser["feat_fuser"]["feat_dim"] = feat_dim
        self.feat_fuser = FeatFuserMVN(**feat_fuser)

    @property
    def sample_frequency(self) -> int:
        """Sampling rate expected by the HF feature extractor.

        Returns:
          Sampling frequency in Hz.
        """
        return self.hf_feats.sample_frequency

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Computes class-prototype affinity from the x-vector backend.

        Returns:
          Affinity tensor produced by the x-vector model.
        """
        return self.xvector.compute_prototype_affinity()

    def update_loss_margin(self, epoch: int) -> None:
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.xvector.update_loss_margin(epoch)

    def rebuild_output_layer(
        self,
        num_classes: Optional[int] = None,
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 10,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
    ) -> None:
        """Rebuilds the classification/output layer in the x-vector backend.

        Args:
          num_classes: Number of target classes. If ``None``, backend defaults are used.
          loss_type: Loss head type, e.g. ``"arc-softmax"``.
          cos_scale: Scale applied to cosine logits.
          margin: Margin value for margin-based softmax variants.
          margin_warmup_epochs: Number of epochs used for margin warmup.
          intertop_k: Number of hardest impostor classes for inter-top margin.
          intertop_margin: Additional margin for inter-top classes.
          num_subcenters: Number of subcenters per class.

        Returns:
          None.
        """
        self.xvector.rebuild_output_layer(
            num_classes=num_classes,
            loss_type=loss_type,
            cos_scale=cos_scale,
            margin=margin,
            margin_warmup_epochs=margin_warmup_epochs,
            intertop_k=intertop_k,
            intertop_margin=intertop_margin,
            num_subcenters=num_subcenters,
        )

    def forward_feats(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor],
        return_feat_layers: Optional[List[int]] = None,
        chunk_length: int = 0,
        detach_chunks: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        """Runs HF feature extraction and fuses selected hidden layers.

        Args:
          x: Input waveform tensor with shape ``(batch, num_samples)``.
          x_lengths: Optional valid lengths for ``x``.
          return_feat_layers: Optional HF hidden-layer indices to return.
          chunk_length: Optional chunk length used by the HF extractor.
          detach_chunks: If ``True``, detaches chunk outputs in chunked extraction.

        Returns:
          Tuple containing fused features ``(batch, feat_dim, time)``, optional
          selected HF hidden features, and fused feature lengths.
        """
        return_hid_states = (
            False
            if return_feat_layers is None and self.feat_fuser.fuser_type == "last"
            else True
        )
        with self._hf_context:
            hf_output = self.hf_feats(
                x,
                x_lengths,
                return_hid_states=return_hid_states,
                chunk_length=chunk_length,
                detach_chunks=detach_chunks,
            )
        feat_lengths = hf_output["hidden_states_lengths"]
        if return_hid_states:
            hid_feats = hf_output["hidden_states"]
            hid_feats = hid_feats[self.feat_fusion_start :]
        else:
            hid_feats = [hf_output["last_hidden_state"]]

        feats, feat_lengths = self.feat_fuser(hid_feats, feat_lengths)
        feats = feats.transpose(1, 2)
        if return_feat_layers is not None:
            # add hidden feats from wav2vec to the output. We transpose to be (batch, C, time)
            # as the hidden features of the x-vector encoder.
            hid_feats = [
                f.transpose(1, 2)
                for i, f in enumerate(hid_feats)
                if i in return_feat_layers
            ]
        else:
            hid_feats = None

        return feats, hid_feats, feat_lengths

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_feat_layers: Optional[List[int]] = None,
        return_enc_layers: Optional[List[int]] = None,
        return_classif_layers: Optional[List[int]] = None,
        return_logits: bool = True,
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """Forward function. If returns the logits posteriors of the classes.
        It can also returns the hidden representations in the wav2vec feature extractor,
        the x-vector encoder and the
        classification head. In this case the ouput variable is a dictionary.

        Args:
          x: Input waveform tensor with shape ``(batch, num_samples)``.
          x_lengths: Time lengths of the waveform tensor with shape ``(batch,)``.
          y: target classes torch.long tensor with shape=(batch,)
          return_feat_layers: HF hidden-layer indices to return. If ``None``, no HF
            hidden states are added to the output.
          return_enc_layers: Encoder layers to return from the backend x-vector.
          return_classif_layers: Classification-head layers to return from the backend x-vector.
          return_logits: If ``True``, the backend output includes logits.

        Returns:
          Tensor with class logits with shape=(batch, num_classes) or
          dictionary with ``logits``, ``h_enc``, ``h_classif``, and optionally ``h_feats``.
        """
        feats, hid_feats, feat_lengths = self.forward_feats(
            x, x_lengths, return_feat_layers
        )
        output = self.xvector(
            feats,
            feat_lengths,
            y,
            return_enc_layers=return_enc_layers,
            return_classif_layers=return_classif_layers,
            return_logits=return_logits,
        )

        if not return_feat_layers:
            return output

        output.h_feats = hid_feats
        return output

    def extract_embed(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        vad_samples: Optional[List[torch.Tensor]] = None,
        hf_chunk_length: int = 0,
        xvec_chunk_length: int = 0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ) -> torch.Tensor:
        """Extracts speaker embeddings for full utterances.

        Args:
          x: Input waveform tensor with shape ``(batch, num_samples)``.
          x_lengths: Optional valid lengths for ``x``.
          vad_samples: Optional per-utterance voiced sample indices for silence removal.
          hf_chunk_length: Chunk length used by HF feature extraction.
          xvec_chunk_length: Chunk length used by x-vector embedding extraction.
          embed_layer: Optional x-vector embedding layer selector.
          detach_chunks: If ``True``, detaches chunk outputs in chunked extraction.

        Returns:
          Embedding tensor returned by ``xvector.extract_embed``.
        """
        if vad_samples is not None:
            x, x_lengths = remove_silence(x, vad_samples, x_lengths)

        feats, _, feat_lengths = self.forward_feats(
            x, x_lengths, chunk_length=hf_chunk_length, detach_chunks=detach_chunks
        )
        xvec_chunk_length = int(
            xvec_chunk_length
            * self.hf_feats.sample_frequency
            * feats.size(-1)
            // x.size(-1)
        )
        return self.xvector.extract_embed(
            feats, feat_lengths, xvec_chunk_length, embed_layer, detach_chunks
        )

    def extract_embed_slidwin(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        vad_t_start: Optional[List[torch.Tensor]] = None,
        vad_t_end: Optional[List[torch.Tensor]] = None,
        win_length: float = 1.0,
        win_shift: float = 0.25,
        chunk_length: float = 0.0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Extracts sliding-window embeddings with optional VAD time constraints.

        Args:
          x: Input waveform tensor with shape ``(batch, num_samples)``.
          x_lengths: Optional valid lengths for ``x``.
          vad_t_start: Optional voiced-region start times per utterance in seconds.
          vad_t_end: Optional voiced-region end times per utterance in seconds.
          win_length: Sliding window duration in seconds.
          win_shift: Sliding window shift in seconds.
          chunk_length: Max accumulated audio duration (seconds) before extraction.
          embed_layer: Optional x-vector embedding layer selector.
          detach_chunks: If ``True``, detaches chunk outputs after extraction.

        Returns:
          Tuple with padded window embeddings, embedding lengths, output window
          start times, and output window end times.
        """
        if vad_t_start is not None:
            assert vad_t_end is not None
            assert len(vad_t_start) == len(vad_t_end)

        assert win_length >= win_shift

        x_strided = []
        embed2x_mappings = []
        out_t_start = []
        out_t_end = []
        embeds = []
        accum_length = 0.0
        for i in range(x.shape[0]):
            x_i = x[i]
            x_length_i = len(x_i) if x_lengths is None else x_lengths[i]
            if vad_t_start is None:
                t_start_i = [0.0]
                t_end_i = [x_length_i / self.sample_frequency]
            else:
                t_start_i = vad_t_start[i]
                t_end_i = vad_t_end[i]

            out_t_start_i = []
            out_t_end_i = []
            for t_start_ij, t_end_ij in zip(t_start_i, t_end_i):
                cur_t_start = t_start_ij
                num_wins_ij = max(
                    1, int((t_end_ij - t_start_ij - win_length + win_shift) / win_shift)
                )
                if num_wins_ij > 1:
                    out_t_center_ij = (
                        win_shift * torch.arange(0, num_wins_ij)
                        + t_start_ij
                        + win_length / 2
                    )
                    out_t_start_ij = out_t_center_ij - win_shift / 2
                    out_t_end_ij = out_t_center_ij + win_shift / 2
                    out_t_start_ij[0] = t_start_ij
                    out_t_end_ij[-1] = t_end_ij
                else:
                    out_t_start_ij = torch.as_tensor([t_start_ij])
                    out_t_end_ij = torch.as_tensor([t_end_ij])

                for win in range(num_wins_ij):
                    cur_t_end = min(cur_t_start + win_length, t_end_ij)
                    cur_sample_start = int(cur_t_start * self.sample_frequency)
                    cur_sample_end = min(
                        int(cur_t_end * self.sample_frequency), x.size(1)
                    )
                    x_ij = x_i[cur_sample_start:cur_sample_end]
                    x_strided.append(x_ij)
                    embed2x_mappings.append(i)
                    accum_length += cur_t_end - cur_t_start
                    if chunk_length > 0 and accum_length >= chunk_length:
                        x_strided, x_strided_lengths = collate_seqs_1d(x_strided)
                        embeds_chunk = self.extract_embed(
                            x_strided,
                            x_strided_lengths,
                            embed_layer=embed_layer,
                            detach_chunks=detach_chunks,
                        )
                        if detach_chunks:
                            embeds_chunk = embeds_chunk.detach()

                        embeds.append(embeds_chunk)
                        del x_strided
                        x_strided = []
                        accum_length = 0.0

                    cur_t_start += win_shift

                out_t_start_i.append(out_t_start_ij)
                out_t_end_i.append(out_t_end_ij)

            out_t_start_i = torch.cat(out_t_start_i)
            out_t_end_i = torch.cat(out_t_end_i)
            out_t_start.append(out_t_start_i)
            out_t_end.append(out_t_end_i)

        if x_strided:
            x_strided, x_strided_lengths = collate_seqs_1d(x_strided)
            embeds_chunk = self.extract_embed(
                x_strided,
                x_strided_lengths,
                embed_layer=embed_layer,
                detach_chunks=detach_chunks,
            )
            if detach_chunks:
                embeds_chunk = embeds_chunk.detach()
            embeds.append(embeds_chunk)

        embeds = torch.cat(embeds, axis=0)
        embed2x_mappings = torch.as_tensor(embed2x_mappings)
        out_embeds = []
        for i in range(x.shape[0]):
            idx = embed2x_mappings == i
            out_embeds.append(embeds[idx])

        out_embeds, embeds_lengths = collate_seqs_2d(out_embeds)
        return out_embeds, embeds_lengths, out_t_start, out_t_end

    def freeze_feat_fuser(self) -> None:
        """Freezes feature-fuser parameters.

        Returns:
          None.
        """
        self.feat_fuser.freeze()

    def freeze_hf_feats(self) -> None:
        """Freezes all HF feature-extractor parameters.

        Returns:
          None.
        """
        self.hf_feats.freeze()

    def freeze_hf_feature_encoder(self) -> None:
        """Freezes the low-level HF feature encoder only.

        Returns:
          None.
        """
        self.hf_feats.freeze_feature_encoder()

    def freeze_hf_except_lora(self, bias: Optional[str] = None) -> None:
        """Freezes HF parameters except LoRA (and optionally bias) parameters.

        Args:
          bias: Bias-freezing mode passed to the HF wrapper.

        Returns:
          None.
        """
        self.hf_feats.freeze_except_lora(bias)

    def has_param_groups(self) -> bool:
        """Checks whether custom optimizer parameter groups are required.

        Returns:
          ``True`` when either HF extractor or x-vector defines parameter groups.
        """
        return self.hf_feats.has_param_groups() or self.xvector.has_param_groups()

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        """Builds trainable optimizer parameter groups for this composite model.

        Returns:
          List of optimizer parameter-group dictionaries.
        """
        if not self.has_param_groups():
            return [{"params": self.trainable_parameters()}]

        param_groups = self.hf_feats.trainable_param_groups()
        param_groups.append({"params": self.feat_fuser.trainable_parameters()})
        param_groups.extend(self.xvector.trainable_param_groups())
        return param_groups

    def set_train_mode(self, mode: str) -> None:
        """Sets train mode and applies the corresponding freeze/unfreeze policy.

        Args:
          mode: Train mode name. Must be one of ``valid_train_modes()``.

        Returns:
          None.

        Raises:
          ValueError: If ``mode`` is unknown.
        """
        if mode == self._train_mode:
            return

        xvector_mode = "full"
        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
            xvector_mode = "frozen"
        elif mode == "ft-embed-affine":
            self.unfreeze()
            self.freeze_feat_fuser()
            self.freeze_hf_feats()
            self.xvector.freeze_preembed_layers()
            xvector_mode = "ft-embed-affine"
        elif mode in ["ft-xvector", "ft-xvector-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
            self.freeze_feat_fuser()
        elif mode in ["hf-feats-frozen", "hf-feats-frozen-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
        elif mode == "hf-feat-extractor-frozen":
            self.unfreeze()
            self.freeze_hf_feature_encoder()
        elif mode == "hf-lora":
            self.unfreeze()
            self.freeze_hf_except_lora()
        elif mode == "hf-all-bias-lora":
            self.unfreeze()
            self.freeze_hf_except_lora(bias="all")
        elif mode == "hf-lora-with-bias":
            self.unfreeze()
            self.freeze_hf_except_lora(bias="lora_only")
        else:
            raise ValueError(f"invalid train_mode={mode}")

        if self.xvector.head_type == "dino":
            self.xvector.classif_net.freeze_output_g()

        logging.info("train mode set to %s", mode)

        if "nograd" in mode or mode == "ft-embed-affine":
            logging.info("using torch.no_grad for hf_feats")
            self._hf_context = torch.no_grad()
        else:
            self._hf_context = contextlib.nullcontext()

        self.xvector.set_train_mode(xvector_mode)
        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Implements train-mode transitions used by ``HyperTorchModel.train``.

        Args:
          train_mode: Internal train mode name.

        Returns:
          None.

        Raises:
          ValueError: If ``train_mode`` is unknown.
        """
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode == "ft-embed-affine":
            self.hf_feats.train()
            self.feat_fuser.train()
            self.xvector._train("ft-embed-affine")
        elif train_mode in [
            "ft-xvector",
            "hf-feats-frozen",
            "ft-xvector-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
            "hf-lora",
            "hf-all-bias-lora",
            "hf-lora-with-bias",
        ]:
            self.hf_feats.train()
            self.feat_fuser.train()
            self.xvector._train("full")
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Lists supported training modes.

        Returns:
          Supported train-mode names.
        """
        return [
            "full",
            "frozen",
            "ft-embed-affine",
            "ft-xvector",
            "hf-feats-frozen",
            "ft-xvector-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
            "hf-lora",
            "hf-all-bias-lora",
            "hf-lora-with-bias",
        ]

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters input kwargs to constructor-supported keys.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Returns:
          Dictionary containing only supported constructor keys.
        """
        valid_args = (
            "hf_feats",
            "feat_fuser",
            "xvector",
            "feat_fusion_start",
            # "feat_fusion_method",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self) -> Dict[str, Any]:
        """Serializes model configuration.

        Returns:
          Configuration dictionary for recreating this model.
        """
        hf_cfg = self.hf_feats.get_config()
        fuser_cfg = self.feat_fuser.get_config()
        xvec_cfg = self.xvector.get_config()
        del hf_cfg["class_name"]
        del fuser_cfg["class_name"]
        del xvec_cfg["class_name"]
        config = {
            "hf_feats": hf_cfg,
            "feat_fuser": fuser_cfg,
            "xvector": xvec_cfg,
            "feat_fusion_start": self.feat_fusion_start,
            # "feat_fusion_method": self.feat_fusion_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, hf_feats: Dict[str, Any], xvector: Dict[str, Any]) -> None:
        """Applies runtime configuration updates to child modules.

        Args:
          hf_feats: Configuration updates for the HF feature extractor.
          xvector: Configuration updates for the x-vector backend.

        Returns:
          None.
        """
        logging.info("changing hf wav2xvector config")
        self.hf_feats.change_config(**hf_feats)
        self.xvector.change_config(**xvector)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Registers CLI arguments for this class.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
          skip: Unused compatibility argument for shared class-arg API.

        Returns:
          None.
        """
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        FeatFuserMVN.add_class_args(parser, prefix="feat_fuser")

        parser.add_argument(
            "--feat-fusion-start",
            default=0,
            type=int,
            help=(
                "the input to x-vector model will fuse the wav2vec layers from feat_fusion_start to"
                "the wav2vec num_layers"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
