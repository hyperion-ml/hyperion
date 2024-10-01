"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import AudioFeatsMVN
from ...torch_model import TorchModel
from ...utils import collate_seqs_1d, collate_seqs_2d, remove_silence


class Wav2XVector(TorchModel):
    """Base class for models that integrate the acoustic feature extractor and and x-vector model that takes acoustic features as input.

    Attributes:
      feats: feature extractor object of class AudioFeatsMVN or dictionary of options to instantiate AudioFeatsMVN object.
      xvector: x-vector model object.
    """

    def __init__(self, feats, xvector):
        super().__init__()

        if isinstance(feats, dict):
            feats = AudioFeatsMVN.filter_args(**feats)
            feats["trans"] = True
            feats = AudioFeatsMVN(**feats)
        else:
            assert isinstance(feats, AudioFeatsMVN)

        self.feats = feats
        self.xvector = xvector
        self._feats_context = contextlib.nullcontext()

    @property
    def sample_frequency(self):
        return self.feats.sample_frequency

    # def clone(self):
    #     # weight normalized layers cannot be copied with deepcopy,
    #     # we remove them to clone and put them back later
    #     modules, cloned_modules = self.xvector.before_cloning()
    #     new_self = super().clone()
    #     self.xvector.after_cloning(*modules)
    #     new_self.xvector.after_cloning(*cloned_modules)
    #     return new_self

    def compute_prototype_affinity(self):
        return self.xvector.compute_prototype_affinity()

    def update_loss_margin(self, epoch: int):
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
    ):
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

    def change_config(self, xvector):
        logging.info("changing wav2xvector config")
        self.xvector.change_config(**xvector)

    def cancel_output_layer_grads(self):
        self.xvector.cancel_output_layer_grads()

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        vad_samples: Optional[torch.Tensor] = None,
        vad_feats: Optional[torch.Tensor] = None,
        enc_layers: Optional[List[int]] = None,
        classif_layers: Optional[List[int]] = None,
        return_output: bool = True,
    ):
        with self._feats_context:
            if vad_samples is not None:
                x, x_lengths = remove_silence(x, vad_samples, x_lengths)

            feats, feat_lengths = self.feats(x, x_lengths)
            if vad_feats is not None:
                feats, feat_lengths = remove_silence(feats, vad_feats, feat_lengths)

        n = torch.sum(~torch.isfinite(feats))
        if n > 0:
            print(
                "feats",
                n,
                torch.sum(torch.isnan(feats)),
                torch.sum(torch.any(torch.isnan(x), dim=-1)),
                x.dtype,
                feats.dtype,
                flush=True,
            )
        return self.xvector(
            feats, feat_lengths, y, enc_layers, classif_layers, return_output
        )

    def extract_embed(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        vad_samples: Optional[torch.Tensor] = None,
        vad_feats: Optional[torch.Tensor] = None,
        chunk_length: float = 0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ):
        with self._feats_context:
            if vad_samples is not None:
                x, x_lengths = remove_silence(x, vad_samples, x_lengths)

            feats, feat_lengths = self.feats(x, x_lengths)
            if vad_feats is not None:
                feats, feat_lengths = remove_silence(feats, vad_feats, feat_lengths)

            chunk_length = int(
                chunk_length * feats.shape[-1] / (x.shape[-1] / self.sample_frequency)
            )

        return self.xvector.extract_embed(
            feats, feat_lengths, chunk_length, embed_layer, detach_chunks
        )

    def extract_embed_slidwin(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        vad_t_starts: Optional[List[torch.Tensor]] = None,
        vad_t_ends: Optional[List[torch.Tensor]] = None,
        win_length: float = 1.0,
        win_shift: float = 0.25,
        chunk_length: float = 0.0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ):
        if vad_t_starts is not None:
            assert vad_t_ends is not None
            assert len(vad_t_starts) == len(vad_t_ends)

        x_strided = []
        embed2x_mappings = []
        out_t_starts = []
        out_t_ends = []
        embeds = []
        accum_length = 0.0
        for i in range(x.shape[0]):
            x_i = x[i]
            x_length_i = len(x_i) if x_lengths is None else x_lengths[i]
            if vad_t_starts is None:
                t_start_i = [0.0]
                t_end_i = [x_length_i / self.sample_frequency]
            else:
                t_start_i = vad_t_starts[i]
                t_end_i = vad_t_ends[i]

            out_t_start_i = []
            out_t_end_i = []
            for t_start_ij, t_end_ij in zip(t_start_i, t_end_i):
                cur_t_start = t_start_ij
                num_wins_ij = (t_end_ij - t_start_ij) // win_shift
                out_t_center_ij = torch.arange(win_length / 2, t_end_ij, win_shift)
                out_t_start_ij = out_t_center_ij - win_shift / 2
                out_t_end_ij = out_t_center_ij + win_shift / 2
                out_t_start_ij[0] = t_start_ij
                out_t_end_ij[-1] = t_end_ij
                for win in range(num_wins_ij):
                    cur_t_end = min(cur_t_start + win_length, t_end_ij)
                    cur_sample_start = cur_t_start * self.sample_frequency
                    cur_sample_end = min(cur_t_end * self.sample_frequency, x.size(1))
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
                        x_strided = []
                        accum_length = 0.0

                    out_t_start_i.append(out_t_start_ij)
                    out_t_end_i.append(out_t_end_ij)
                    cur_t_start += win_shift

            out_t_start_i = torch.cat(out_t_start_i)
            out_t_end_i = torch.cat(out_t_end_i)
            out_t_starts.append(out_t_start_i)
            out_t_ends.append(out_t_end_i)

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
        return out_embeds, embeds_lengths, out_t_starts, out_t_ends

    def trainable_param_groups(self):
        param_groups = self.xvector.trainable_param_groups()
        return param_groups

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return
        logging.info("setting Wav2XVector train mode to %s", mode)
        if mode == "full-feats-grad":
            self._feats_context = contextlib.nullcontext()
            xvector_mode = "full"
        else:
            logging.info("using torch.no_grad for feats")
            self._feats_context = torch.no_grad()
            xvector_mode = mode

        logging.info(
            "setting Wav2XVector XVector object train mode to %s", xvector_mode
        )
        self.xvector.set_train_mode(xvector_mode)
        self._train_mode = mode

    def _train(self, train_mode: str):
        self.feats.train()
        if train_mode in ["frozen"]:
            super()._train(train_mode)
        elif train_mode in ["full-feats-grad", "full"]:
            self.xvector._train("full")
        elif train_mode == "ft-embed-affine":
            self.xvector._train(train_mode)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes():
        return [
            "full",
            "frozen",
            "ft-embed-affine",
            "full-feats-grad",
        ]

    def get_config(self):
        feat_cfg = self.feats.get_config()
        xvector_cfg = self.xvector.get_config()
        config = {
            "feats": feat_cfg,
            "xvector": xvector_cfg,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        """Filters Wav2XVector class arguments from arguments dictionary.

        Args:
          kwargs: Arguments dictionary.

        Returns:
          Dictionary with SpecAugment options.
        """
        valid_args = (
            "feats",
            "xvector",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds Wav2XVector options common to all child classes to parser.

        Args:
          parser: Arguments parser
          prefix: Options prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AudioFeatsMVN.add_class_args(parser, prefix="feats")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
