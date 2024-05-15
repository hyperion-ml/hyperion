"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...torch_model import TorchModel
from ...utils import remove_silence
from ..transducer import RNNTransducer, RNNTransducerOutput


class HFWav2RNNTransducer(TorchModel):
    """Abstract Base class for RNN-T transducer models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       transducer: transducer model object.
       feat_fusion_start: the input to x-vector model will fuse the wav2vec layers from "feat_fusion_start" to
                          the wav2vec "num_layers".
       feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used.
    """

    def __init__(
        self,
        hf_feats: TorchModel,
        transducer: Union[Dict, TorchModel],
        feat_fusion_start: int = 0,
        feat_fusion_method: str = "weighted-avg",
    ):

        super().__init__()
        self.hf_feats = hf_feats
        if isinstance(transducer, dict):
            transducer["decoder"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer["encoder"] = None
            transducer = RNNTransducer(**transducer)
        else:
            assert isinstance(transducer, RNNTransducer)
            if transducer.encoder is None:
                assert transducer.decoder.in_feats == hf_feats.hidden_size

        self.transducer = transducer
        self.feat_fusion_start = feat_fusion_start
        self.feat_fusion_method = feat_fusion_method
        self._hf_context = contextlib.nullcontext()
        self._make_fuser()

    def _make_fuser(self):
        if self.feat_fusion_method == "last":
            self.feat_fuser = None
            return

        num_layers = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        layer_dim = self.hf_feats.hidden_size
        if self.feat_fusion_method == "weighted-avg":
            self.feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif self.feat_fusion_method == "linear":
            self.feat_fuser = nn.Linear(num_layers, 1, bias=False)
            self.feat_fuser.weight.data = torch.ones(1, num_layers) / num_layers
        elif self.feat_fusion_method == "cat":
            self.feat_fuser = nn.Linear(num_layers * layer_dim, layer_dim, bias=False)

    def _fuse_hid_feats(self, hid_feats):
        """Fuses the hidden features from the Wav2Vec model.

        Args:
          hid_feats: list of hidden features Tensors from Wav2Vec model.

        Returns:
          Tensor of fused features (batch, channels, time)
        """
        if len(hid_feats) == 1:
            # There is only one layer of features
            return hid_feats[0]

        hid_feats = hid_feats[self.feat_fusion_start :]
        if self.feat_fusion_method == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats).squeeze(dim=-1)
        elif self.feat_fusion_method == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats)
        elif self.feat_fusion_method == "last":
            feats = hid_feats[-1]

        return feats

    def forward_feats(
        self, x, x_lengths, return_feat_layers=None, chunk_length=0, detach_chunks=False
    ):
        return_hid_states = (
            False
            if return_feat_layers is None and self.feat_fusion_method == "last"
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
            feats = self._fuse_hid_feats(hid_feats)
        else:
            hid_feats = None
            feats = hf_output["last_hidden_state"]

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
        x,
        x_lengths=None,
        y=None,
        return_feat_layers=None,
        # return_enc_layers=None,
        return_logits=True,
    ):
        """Forward function. If returns the logits posteriors of the classes.
        It can also returns the hidden representations in the wav2vec feature extractor,
        the x-vector encoder and the
        classification head. In this case the ouput variable is a dictionary.

        Args:
          x: input features tensor with shape=(batch, in_feats, time)
          x_lengths: time lengths of the features with shape=(batch,)
          y: target classes torch.long tensor with shape=(batch,)
          return_feat_layers: list of integers indicating, which wav2vec layers
                             we should return. If None, no wav2vec layers are returned.
          return_enc_layers: list of integers indicating, which encoder layers
                             we should return. If None, no encoder layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Dataclass with losses, "h_enc" (list of hidden encoder layers),
          "h_feats" (wav2vec features)
        """
        feats, hid_feats, feat_lengths = self.forward_feats(
            x, x_lengths, return_feat_layers
        )

        feats = feats.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)
        output = self.transducer(
            feats,
            feat_lengths,
            y,
        )

        if return_feat_layers:
            output.h_feats = hid_feats

        return output

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        decoding_method="time_sync_beam_search",
        beam_width: int = 5,
        max_sym_per_frame: int = 3,
        max_sym_per_utt: int = 1000,
    ):
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

        feats, _, feat_lengths = self.forward_feats(x, x_lengths)

        feats = feats.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

        y = self.transducer.infer(
            feats,
            feat_lengths,
            decoding_method=decoding_method,
            beam_width=beam_width,
            max_sym_per_frame=max_sym_per_frame,
            max_sym_per_utt=max_sym_per_utt,
        )
        return y

    def freeze_feat_fuser(self):
        if self.feat_fuser is None:
            return

        if self.feat_fusion_method == "weighted-avg":
            self.feat_fuser.requires_grad = False
            return

        for param in self.feat_fuser.parameters():
            param.requires_grad = False

    def freeze_hf_feats(self):
        self.hf_feats.freeze()

    def freeze_hf_feature_encoder(self):
        self.hf_feats.freeze_feature_encoder()

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        elif mode in ["ft-transducer", "ft-transducer-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
            self.freeze_feat_fuser()
        elif mode in ["hf-feats-frozen", "hf-feats-frozen-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
        elif mode == "hf-feat-extractor-frozen":
            self.unfreeze()
            self.freeze_hf_feature_encoder()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        logging.info("train mode set to %s", mode)

        if "nograd" in mode:
            logging.info("using torch.no_grad for hf_feats")
            self._hf_context = torch.no_grad()
        else:
            self._hf_context = contextlib.nullcontext()

        self._train_mode = mode

    def _train(self, train_mode: str):

        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode in [
            "ft-transducer",
            "hf-feats-frozen",
            "ft-transducer-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
        ]:
            self.hf_feats.train()
            self.transducer._train("full")
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes():
        return [
            "full",
            "frozen",
            "ft-embed-affine",
            "ft-transducer",
            "hf-feats-frozen",
            "ft-transducer-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
        ]

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "hf_feats",
            "transducer",
            "feat_fusion_start",
            "feat_fusion_method",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self):
        hf_cfg = self.hf_feats.get_config()
        tran_cfg = self.transducer.get_config()
        del hf_cfg["class_name"]
        del tran_cfg["class_name"]
        config = {
            "hf_feats": hf_cfg,
            "transducer": tran_cfg,
            "feat_fusion_start": self.feat_fusion_start,
            "feat_fusion_method": self.feat_fusion_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, hf_feats, transducer):
        logging.info("changing hf wav2transducer config")
        self.hf_feats.change_config(**hf_feats)
        self.transducer.change_config(**transducer)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--feat-fusion-start",
            default=0,
            type=int,
            help="""
            the input to x-vector model will fuse the wav2vec 
            layers from feat_fusion_start to
            the wav2vec num_layers""",
        )
        parser.add_argument(
            "--feat-fusion-method",
            default="weighted-avg",
            choices=["weighted-avg", "linear", "cat", "last"],
            help=(
                "method to fuse the hidden layers from the wav2vec model "
                "in [weighted-avg, linear, cat, last]"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    @staticmethod
    def add_infer_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducer.add_infer_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs):
        return RNNTransducer.filter_infer_args(**kwargs)
