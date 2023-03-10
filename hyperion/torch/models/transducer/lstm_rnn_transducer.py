"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Dict, Optional, Union
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch

from ...torch_model import TorchModel
from ..narchs import RNNTransducerDecoder


class RNNTransducer(TorchModel):
    """ Base-class for RNN-T in
    "Sequence Transduction with Recurrent Neural Networks"
    https://arxiv.org/pdf/1211.3711.pdf

    Attributes:
      encoder: Encoder network module
      decoder: RNN-T Decoder config. dictionary or module.
    """

    def __init__(
        self,
        encoder: TorchModel,
        decoder: Union[Dict, RNNTransducerDecoder],
    ):
        super().__init__()
        assert isinstance(encoder, TorchModel)
        if isinstance(decoder, dict):
            decoder = RNNTransducerDecoder(**decoder)
        else:
            assert isinstance(decoder, RNNTransducerDecoder)

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: k2.RaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
          x: input features with shape = (N, T, C)
          x_lengths: feature number for frames with shape = (N,)
          y: ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
        Returns:
          - Token logits with shape = (N, vocab_size)
          - RNN-T loss.
        """
        assert x.ndim == 3, x.shape
        assert x_lengths.ndim == 1, x_lengths.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lengths.size(0) == y.dim0

        x, x_lengths = self.encoder(x, x_lengths)
        assert torch.all(x_lengths > 0)

        logits, loss = self.decoder(x, x_lengths, y)
        return logits, loss

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        self._train_mode = mode

    def _train(self, train_mode: str):
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes():
        return ["full", "frozen"]

    def get_config(self):
        dec_cfg = self.decoder.get_config()
        config = {
            "decoder": dec_cfg,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):

        # get arguments for pooling
        decoder_args = RNNTransducerDecoder.filter_args(**kwargs["decoder"])
        args["decoder"] = decoder_args
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducerDecoder.add_class_args(parser, prefix="decoder")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    def change_config(
        self,
        decoder,
    ):
        logging.info("changing transducer config")
        self.decoder.change_config(**decoder)

    @staticmethod
    def filter_finetune_args(**kwargs):
        # get arguments for pooling
        decoder_args = Decoder.filter_finetune_args(**kwargs["decoder"])
        args["decoder"] = decoder_args
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducerDecoder.add_finetune_args(parser, prefix="decoder")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
