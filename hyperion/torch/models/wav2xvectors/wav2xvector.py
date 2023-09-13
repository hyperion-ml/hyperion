"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import contextlib
import logging

from jsonargparse import ActionParser, ArgumentParser

import torch
import torch.nn as nn

from ...narchs import AudioFeatsMVN
from ...torch_model import TorchModel
from ...utils import remove_silence


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

    def compute_prototype_affinity(self):
        return self.xvector.compute_prototype_affinity()

    def update_loss_margin(self, epoch):
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.xvector.update_loss_margin(epoch)

    def rebuild_output_layer(
        self,
        num_classes=None,
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=10,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
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

    def forward(
        self,
        x,
        x_lengths=None,
        y=None,
        vad_samples=None,
        vad_feats=None,
        enc_layers=None,
        classif_layers=None,
        return_output=True,
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
        x,
        x_lengths=None,
        vad_samples=None,
        vad_feats=None,
        chunk_length=0,
        embed_layer=None,
        detach_chunks=False,
    ):

        with self._feats_context:
            if vad_samples is not None:
                x, x_lengths = remove_silence(x, vad_samples, x_lengths)

            feats, feat_lengths = self.feats(x, x_lengths)
            if vad_feats is not None:
                feats, feat_lengths = remove_silence(feats, vad_feats, feat_lengths)

            chunk_length = int(chunk_length * feats.shape[1] / x.shape[-1])

        return self.xvector.extract_embed(
            feats, feat_lengths, chunk_length, embed_layer, detach_chunks
        )

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full-feats-grad":
            self._feats_context = contextlib.nullcontext()
            xvector_mode = "full"
        else:
            logging.info("using torch.no_grad for feats")
            self._feats_context = torch.no_grad()

        self.xvector.set_train_mode(xvector_mode)
        self._train_mode = mode

    def _train(self, train_mode: str):

        self.feats.train()
        if train_mode in ["frozen"]:
            super()._train(train_mode)
        elif train_mode in ["full-feats-grad", "full"]:
            self.xvector._train("full")
        elif train_mode == "ft-embed-affine":
            self.xvector._train("ft-embed_affine")
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
