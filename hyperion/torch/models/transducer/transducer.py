# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note we use `rnnt_loss` from torchaudio, which exists only in
torchaudio >= v0.10.0. It also means you have to use torch >= v1.10.0
"""
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import logging
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional
from .encoder_interface import EncoderInterface

from ...torch_model import TorchModel
from hyperion.utils.text import add_sos
# from .conformer import Conformer
from .decoder import Decoder
from .joiner import Joiner


class Transducer(TorchModel):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """
    def __init__(
        self,
        vocab_size,
        blank_id,
        # conformer_enc,
        decoder,
        joiner,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        # assert isinstance(encoder, EncoderInterface)
        # assert hasattr(decoder, "blank_id")

        decoder["blank_id"] = blank_id
        decoder["vocab_size"] = vocab_size
        joiner["out_dims"] = vocab_size

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.decoder = Decoder(**decoder)
        self.joiner = Joiner(**joiner)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
        Returns:
          Return the transducer loss.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        #  wav2vec2 works as encoder
        # encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        encoder_out = x
        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        decoder_out, _ = self.decoder(sos_y_padded)

        logits = self.joiner(encoder_out, decoder_out)

        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)

        assert hasattr(torchaudio.functional, "rnnt_loss"), (
            f"Current torchaudio version: {torchaudio.__version__}\n"
            "Please install a version >= 0.10.0")

        x_lens = x_lens.to(torch.int32)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded.to(torch.int32),
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
        )

        return logits, loss

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        elif mode == "ft-embed-affine":
            self.unfreeze()
            self.freeze_preembed_layers()
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
        return ["full", "frozen", "ft-embed-affine"]

    def get_config(self):
        dec_cfg = self.decoder.get_config()
        join_cfg = self.joiner.get_config()

        config = {
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
            "decoder": dec_cfg,
            "joiner": join_cfg,
        }

        # base_config = super().get_config()
        return dict(list(config.items()))

    @staticmethod
    def filter_args(**kwargs):

        # get arguments for pooling
        decoder_args = Decoder.filter_args(**kwargs["decoder"])
        joiner_args = Joiner.filter_args(**kwargs["joiner"])

        valid_args = ()
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        args["decoder"] = decoder_args
        args["joiner"] = joiner_args
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Decoder.add_class_args(parser, prefix="decoder")
        Joiner.add_class_args(parser, prefix="joiner")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    def change_config(self, 
        decoder,
        # joiner,
        ):
        logging.info("changing transducer config")
        self.decoder.change_config(**decoder)
        # self.joiner.change_config(**joiner)

    @staticmethod
    def filter_finetune_args(**kwargs):
        # get arguments for pooling
        decoder_args = Decoder.filter_finetune_args(**kwargs["decoder"])
        # joiner_args = Joiner.filter_finetune_args(**kwargs["joiner"])

        valid_args = (
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        args["decoder"] = decoder_args
        # args["joiner"] = joiner_args
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        Decoder.add_finetune_args(parser, prefix="decoder")
        # Joiner.add_finetune_args(parser, prefix="joiner")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
    add_argparse_finetune_args = add_finetune_args


