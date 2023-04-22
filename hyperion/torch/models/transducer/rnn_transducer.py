"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch

from ....utils import HypDataClass
from ....utils.misc import filter_func_args
from ...narchs import RNNTransducerDecoder
from ...torch_model import TorchModel


@dataclass
class RNNTransducerOutput(HypDataClass):

    loss: torch.Tensor
    loss_simple: Optional[torch.Tensor] = None
    loss_pruned: Optional[torch.Tensor] = None
    h_feats: Optional[List[torch.Tensor]] = None


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
        encoder: Union[TorchModel, None],
        decoder: Union[Dict, RNNTransducerDecoder],
    ):
        super().__init__()
        if encoder is not None:
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
        y: Union[Dict, k2.RaggedTensor],
    ) -> RNNTransducerOutput:
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
        assert torch.all(
            x_lengths[:-1] >= x_lengths[1:]
        ), f"x_lengths={x_lengths}"  # check x_lengths are sorted

        if self.encoder is not None:
            x, x_lengths = self.encoder(x, x_lengths)
            assert torch.all(x_lengths > 0)

        dec_output = self.decoder(x, x_lengths, y)
        output = RNNTransducerOutput(*dec_output)
        return output

    def infer(self,
              x: torch.Tensor,
              x_lengths: torch.Tensor,
              decoding_method="time_sync_beam_search",
              beam_width: int = 5,
              max_sym_per_frame: int = 3,
              max_sym_per_utt: int = 1000) -> List[List[int]]:
        """
        ASR tokens inference
        Args:
          x: input features with shape = (N, T, C)
          x_lengths: feature number for frames with shape = (N,)
          decoding_method: greedy, time_sync_beam_search or align_length_sync_beam_search
          max_sym_per_frame: maximum number of symbols RNN-T can emit in 1 frame.
          max_sym_per_utt: maximimum number of symbols in a single utterance.
        Returns:
          List of list of integer indexes of the recognizer's symbols.
        """
        assert x.ndim == 3, x.shape
        assert x_lengths.ndim == 1, x_lengths.shape
        assert x.size(0) == x_lengths.size(0)

        if self.encoder is not None:
            x, x_lengths = self.encoder(x, x_lengths)
            assert torch.all(x_lengths > 0)

        batch_size = x.size(0)
        y = []
        for i in range(batch_size):
            x_i = x[i:i + 1, :x_lengths[i]]
            y_i = self.decoder.decode(x_i,
                                      method=decoding_method,
                                      beam_width=beam_width,
                                      max_sym_per_frame=max_sym_per_frame,
                                      max_sym_per_utt=max_sym_per_utt)
            y.append(y_i)

        return y

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
        if self.encoder is None:
            enc_cfg = None
        else:
            enc_cfg = self.encoder.get_config()
            del enc_cfg["class_name"]

        dec_cfg = self.decoder.get_config()
        del dec_cfg["class_name"]
        config = {
            "encoder": enc_cfg,
            "decoder": dec_cfg,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):
        # get arguments for pooling
        args = {}
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
        decoder: Dict,
    ):
        logging.info("changing decoder config")
        self.decoder.change_config(**decoder)

    @staticmethod
    def filter_finetune_args(**kwargs):
        args = {}
        decoder_args = RNNTransducerDecoder.filter_finetune_args(**kwargs["decoder"])
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

    @staticmethod
    def add_infer_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--decoding-method",
                            default="time_sync_beam_search",
                            choices=[
                                "greedy", "time_sync_beam_search",
                                "align_length_sync_beam_search"
                            ])

        parser.add_argument("--beam-width",
                            default=5,
                            type=int,
                            help="beam width for beam search")
        parser.add_argument("--max-sym-per-frame",
                            default=3,
                            type=int,
                            help="max symbols RNN-T can emit in 1 frame")
        parser.add_argument("--max-sym-per-utt",
                            default=1000,
                            type=int,
                            help="max symbols RNN-T can emit in 1 frame")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs):
        return filter_func_args(RNNTransducer.infer, kwargs)
