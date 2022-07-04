"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from ...torch_model import TorchModel
from ...narchs import AudioFeatsMVN
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

    def compute_prototype_affinity(self):
        return self.xvector.compute_prototype_affinity()

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

        if vad_samples is not None:
            x, x_lengths = remove_silence(x, x_lengths)
        feats, feat_lengths = self.feats(x, x_lengths)
        if vad_feats is not None:
            feats, feat_lengths = remove_silence(feats, feat_lengths)

        # feat_lengths = torch.div(x_lengths * feats.size(-1), x.size(-1))
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

        if vad_samples is not None:
            x, x_lengths = remove_silence(x, x_lengths)
        feats, feat_lengths = self.feats(x, x_lengths)
        if vad_feats is not None:
            feats, feat_lengths = remove_silence(feats, feat_lengths)

        feats = feats.transpose(1, 2)
        return self.xvector.extract_embed(
            feats, feat_lengths, chunk_length, embed_layer, detach_chunks
        )

    def train_mode(self, mode="ft-embed-affine"):
        self.xvector.train_mode(mode)

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
    def filter_args(*kwargs):
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
