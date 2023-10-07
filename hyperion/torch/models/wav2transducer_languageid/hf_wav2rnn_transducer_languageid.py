"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils import HypDataClass
from ...torch_model import TorchModel
from ...utils import remove_silence
from ..transducer import RNNTransducer, RNNTransducerOutput
from ..xvectors import ResNet1dXVector as ResNet1dLanguageID

@dataclass
class RNNTransducerLanguageIDOutput(HypDataClass):
    loss: torch.Tensor  # Total loss
    loss_transducer: torch.Tensor  # Loss from the transducer
    loss_lid: torch.Tensor  # Loss from the language ID
    loss_embed: Optional[torch.Tensor] = None  # Loss from the embedding
    loss_reg_lid: Optional[torch.Tensor] = None  # Regularization loss from the language ID
    loss_reg_transducer: Optional[torch.Tensor] = None  # Regularization loss from the transducer
    loss_transducer_simple: Optional[torch.Tensor] = None  # Simple loss from the transducer, if available
    loss_transducer_pruned: Optional[torch.Tensor] = None  # Pruned loss from the transducer, if available
    h_feats: Optional[List[torch.Tensor]] = None  # Hidden features, if available
    logits: Optional[torch.Tensor] = None  # Logits from languageid, if available


class HFWav2RNNTransducerLanguageID(TorchModel):
    """Abstract Base class for combined transducer language identification models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       transducer: transducer model object.
       languageid: language identification model object.
       feat_fusion_start: the input to the combined model will fuse the wav2vec layers from "feat_fusion_start" to
                          the wav2vec "num_layers".
       feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used.
    """

    def __init__(self,
                 hf_feats: TorchModel,
                 transducer: Union[Dict, TorchModel],
                 languageid: Union[Dict, TorchModel],
                 feat_fusion_start_transducer: int = 0,
                 feat_fusion_start_lid: int = 0,
                 feat_fusion_method_transducer: str = "weighted-avg",
                 feat_fusion_method_lid: str = "weighted-avg",
                 loss_lid_type: str = "weightedCE",
                 loss_class_weight: Optional[torch.Tensor] = None,
                 loss_class_weight_exp= 1.0,
                 loss_weight_transducer: float = 0.005,
                 loss_weight_lid: float = 1.0,
                 lid_length: float = 3.0,
                 ):

        super().__init__()
        self.hf_feats = hf_feats
        if isinstance(transducer, dict):
            transducer["decoder"]["in_feats"] = hf_feats.hidden_size
            #transducer["joiner"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer["encoder"] = None
            transducer = RNNTransducer(**transducer)
        else:
            assert isinstance(transducer, RNNTransducer)
            if transducer.encoder is None:
                assert transducer.decoder.in_feats == hf_feats.hidden_size
                #assert transducer.joiner.in_feats == hf_feats.hidden_size

        if isinstance(languageid, dict):
            languageid["resnet_enc"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in languageid:
                del languageid["class_name"]
            languageid = ResNet1dLanguageID(**languageid)
        else:
            assert isinstance(languageid, ResNet1dLanguageID)
            assert languageid.encoder_net.in_feats == hf_feats.hidden_size


        self.transducer = transducer
        self.languageid = languageid
        self.feat_fusion_start_transducer = feat_fusion_start_transducer
        self.feat_fusion_start_lid = feat_fusion_start_lid
        self.feat_fusion_method_transducer = feat_fusion_method_transducer
        self.feat_fusion_method_lid = feat_fusion_method_lid
        self.loss_lid_type = loss_lid_type
        self.loss_class_weight = loss_class_weight
        self.loss_class_weight_exp = loss_class_weight_exp

        if loss_lid_type == "CE" or loss_lid_type is None:
            self.loss_lid = nn.CrossEntropyLoss()
        elif loss_lid_type == "weightedCE":
            self.loss_lid = nn.CrossEntropyLoss(weight=torch.tensor(loss_class_weight.values, dtype=torch.float)**(-loss_class_weight_exp))
            logging.info(torch.tensor(loss_class_weight.values)**(-loss_class_weight_exp))
        elif loss_lid_type == "focal_loss":
            self.loss_lid = FocalLoss(alpha=torch.tensor(loss_class_weight.values)**(-loss_class_weight_exp), gamma=2, size_average=True)

        self.loss_weight_transducer = loss_weight_transducer
        self.loss_weight_lid = loss_weight_lid
        self.lid_length = lid_length
        self._hf_context = contextlib.nullcontext()
        self.transducer_fuser = self._make_fuser(self.feat_fusion_method_transducer, self.feat_fusion_start_transducer)
        self.languageid_fuser = self._make_fuser(self.feat_fusion_method_lid, self.feat_fusion_start_lid)

    def _make_fuser(self, method, start):
        if method == "last":
            feat_fuser = None
            return feat_fuser
        num_layers = self.hf_feats.num_encoder_layers + 1 - start
        layer_dim = self.hf_feats.hidden_size
        if method == "weighted-avg":
            feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif method == "linear":
            feat_fuser = nn.Linear(num_layers, 1, bias=False)
            feat_fuser.weight.data = torch.ones(1,
                                                     num_layers) / num_layers
        elif method == "cat":
            feat_fuser = nn.Linear(num_layers * layer_dim,
                                        layer_dim,
                                        bias=False)

        return feat_fuser


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

        hid_feats = hid_feats[self.feat_fusion_start_transducer:]
        if self.feat_fusion_method_transducer == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_transducer_weights = nn.functional.softmax(self.transducer_fuser, dim=-1)
            norm_lid_weights = nn.functional.softmax(self.languageid_fuser, dim=-1)
            feats_transducer = torch.sum(hid_feats * norm_transducer_weights, dim=-1)
            feats_languageid = torch.sum(hid_feats * norm_lid_weights, dim=-1)
        elif self.feat_fusion_method_transducer == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats_transducer = self.transducer_fuser(hid_feats).squeeze(dim=-1)
            feats_languageid = self.languageid_fuser(hid_feats).squeeze(dim=-1)
        elif self.feat_fusion_method_transducer == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats_transducer = self.transducer_fuser(hid_feats)
            feats_languageid = self.languageid_fuser(hid_feats)
        elif self.feat_fusion_method_transducer == "last":
            feats = hid_feats[-1]

        return feats_transducer, feats_languageid

    def forward_feats(self,
                      x,
                      x_lengths,
                      return_feat_layers=None,
                      chunk_length=0,
                      detach_chunks=False):
        return_hid_states = (False if return_feat_layers is None
                             and self.feat_fusion_method_transducer == "last" else True)
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
            feats_transducer, feats_languageid = self._fuse_hid_feats(hid_feats)
            # feats_languageid = self._fuse_hid_feats(hid_feats, self.feat_fusion_method_lid, self.languageid_fuser)
        else:
            hid_feats = None
            feats_transducer = hf_output["last_hidden_state"]
            feats_languageid = hf_output["last_hidden_state"]

        feats_transducer = feats_transducer.transpose(1, 2)
        feats_languageid = feats_languageid.transpose(1, 2)
        if return_feat_layers is not None:
            # add hidden feats from wav2vec to the output. We transpose to be (batch, C, time)
            # as the hidden features of the x-vector encoder.
            hid_feats = [
                f.transpose(1, 2) for i, f in enumerate(hid_feats)
                if i in return_feat_layers
            ]
        else:
            hid_feats = None

        return feats_transducer, feats_languageid, hid_feats, feat_lengths
            
    # def languageid_chunk(self, feats, lengths):
    #     sr = self.hf_feats.get_config()["sample_frequency"]
    #     strides = self.hf_feats.get_config()["conv_stride"]
        
    #     total_stride = torch.prod(torch.tensor(strides, dtype=torch.float32))

    #     chunk_length = int(self.lid_length * sr / total_stride)

    #     # Check if all samples are longer than chunk_length
    #     if any(len < chunk_length for len in lengths):
    #         return feats

    #     start_indices = [torch.randint(0, len - chunk_length + 1, (1,)).item() for len in lengths]

    #     chunks = torch.stack([feats[i, :, start:start + chunk_length] for i, start in enumerate(start_indices)])

    #     return chunks


    def forward(
        self,
        x,
        x_lengths=None,
        text=None,
        languageid=None,
        return_feat_layers=None,
        return_enc_layers=None,
        return_classif_layers=None,
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
        feats_transducer, feats_languageid, hid_feats, feat_lengths = self.forward_feats(
            x, x_lengths, return_feat_layers)

        lid_len = int(self.lid_length * 50)
        min_len = torch.min(feat_lengths).item()
        if min_len > lid_len:
            lid_start = torch.randint(0, min_len - lid_len + 1, (1,)).item()
            feats_languageid = feats_languageid[:, :, lid_start: lid_start + lid_len]


        # feats_languageid = self.languageid_chunk(feats_languageid, feat_lengths)

        feats_transducer = feats_transducer.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)
            
        logits = self.languageid(
            feats_languageid,
            None,
            languageid,
            return_enc_layers=return_enc_layers,
            return_classif_layers=return_classif_layers,
            return_logits=return_logits,
        )

        # loss_lid = nn.CrossEntropyLoss()(logits, languageid)
        loss_lid = self.loss_lid(logits, languageid)
        
        trans_output = self.transducer(
            feats_transducer,
            feat_lengths,
            text,
        )


        if return_feat_layers:
            trans_output.h_feats = hid_feats
        output = RNNTransducerLanguageIDOutput(loss=self.loss_weight_transducer * trans_output.loss + self.loss_weight_lid * loss_lid, 
                                                loss_transducer=trans_output.loss, 
                                                loss_lid=loss_lid,
                                                loss_transducer_simple=trans_output.loss_simple, 
                                                loss_transducer_pruned=trans_output.loss_pruned,
                                                h_feats=trans_output.h_feats,
                                                logits=logits if return_logits else None)
        return output

    def infer(self,
              x: torch.Tensor,
              x_lengths: torch.Tensor,
              decoding_method="time_sync_beam_search",
              beam_width: int = 5,
              max_sym_per_frame: int = 3,
              max_sym_per_utt: int = 1000):
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

        feats_transducer, feats_languageid, _, feat_lengths = self.forward_feats(x, x_lengths)
        # logging.info(f"feat_lengths: {feat_lengths}")
        # logging.info(f"feats_transducer.shape: {feats_transducer.shape}")
        # logging.info(f"feats_languageid.shape: {feats_languageid.shape}")
        # logging.info(f"feats_transducer: {feats_transducer}")
        # logging.info(f"feats_languageid: {feats_languageid}")
        lid = self.languageid(
            feats_languageid.float(),
            None,
            None,
            return_enc_layers=None,
            return_classif_layers=None,
            return_logits=True,
        )


        feats_transducer = feats_transducer.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

        text = self.transducer.infer(feats_transducer,
                                  feat_lengths,
                                  decoding_method=decoding_method,
                                  beam_width=beam_width,
                                  max_sym_per_frame=max_sym_per_frame,
                                  max_sym_per_utt=max_sym_per_utt)

        return text, lid

    # def freeze_feat_fuser(self):
    #     if self.feat_fuser is None:
    #         return

    #     if self.feat_fusion_method_transducer == "weighted-avg":
    #         self.feat_fuser.requires_grad = False
    #         return

    #     for param in self.feat_fuser.parameters():
    #         param.requires_grad = False

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
            "feat_fusion_start_transducer",
            "feat_fusion_start_lid",
            "feat_fusion_method_transducer",
            "feat_fusion_method_lid",
            "loss_lid_type",
            "loss_class_weight",
            "loss_class_weight_exp",
            "loss_weight_transducer",
            "loss_weight_lid",
            "languageid",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self):
        hf_cfg = self.hf_feats.get_config()
        tran_cfg = self.transducer.get_config()
        lid_cfg = self.languageid.get_config()
        del hf_cfg["class_name"]
        del tran_cfg["class_name"]
        del lid_cfg["class_name"]
        config = {
            "hf_feats": hf_cfg,
            "transducer": tran_cfg,
            "languageid": lid_cfg,
            "feat_fusion_start_transducer": self.feat_fusion_start_transducer,
            "feat_fusion_start_lid": self.feat_fusion_start_lid,
            "feat_fusion_method_transducer": self.feat_fusion_method_transducer,
            "feat_fusion_method_lid": self.feat_fusion_method_lid,
            "loss_lid_type": self.loss_lid_type,
            "loss_class_weight": self.loss_class_weight,
            "loss_class_weight_exp": self.loss_class_weight_exp,
            "loss_weight_transducer": self.loss_weight_transducer,
            "loss_weight_lid": self.loss_weight_lid,
            "lid_length": self.lid_length,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def change_config(self, hf_feats, transducer, languageid):
    def change_config(self, loss_weight_transducer, loss_weight_lid, lid_length, hf_feats, transducer, languageid):
        logging.info("changing hf wav2transducer config")

        self.loss_weight_transducer = loss_weight_transducer
        self.loss_weight_lid = loss_weight_lid
        self.lid_length = lid_length
        self.loss_reg_weight_transducer = loss_reg_weight_transducer
        self.loss_reg_weight_lid = loss_reg_weight_lid
        
        self.hf_feats.change_config(**hf_feats)
        self.transducer.change_config(**transducer)
        self.languageid.change_config(**languageid)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--feat-fusion-start-transducer",
            default=0,
            type=int,
            help="""
            the input to transducer model will fuse the wav2vec 
            layers from feat_fusion_start_transducer to
            the wav2vec num_layers""",
        )
        parser.add_argument(
            "--feat-fusion-start-lid",
            default=0,
            type=int,
            help="""
            the input to lid model will fuse the wav2vec 
            layers from feat_fusion_start_lid to
            the wav2vec num_layers""",
        )

        parser.add_argument(
            "--feat-fusion-method-transducer",
            default="weighted-avg",
            choices=["weighted-avg", "linear", "cat", "last"],
            help=("method to fuse the hidden layers from the wav2vec model "
                  "in [weighted-avg, linear, cat, last]"),
        )
        parser.add_argument(
            "--feat-fusion-method-lid",
            default="weighted-avg",
            choices=["weighted-avg", "linear", "cat", "last"],
            help=("method to fuse the hidden layers from the wav2vec model "
                  "in [weighted-avg, linear, cat, last]"),
        )

        parser.add_argument(
            "--loss-lid-type",
            default="weightedCE",
            choices=["CE", "weightedCE", "focal_loss"],
            help=("loss type for language identification"),
        )
        parser.add_argument(
            "--loss-class-weight",
            default=None,
            type=str,
            help=("class weight for language identification"),
        )
        parser.add_argument(
            "--loss-class-weight-exp",
            default=1.0,
            type=float,
            help=("class weight exponent for language identification"),
        )
        parser.add_argument(
            "--loss-weight-transducer",
            default=0.005,
            type=float,
            help="""
            The weight of the transducer loss
            """,
        )

        parser.add_argument(
            "--loss-weight-lid",
            default=1.0,
            type=float,
            help="""
            The weight of the lid loss
            """,
        )

        parser.add_argument(
            "--lid-length",
            default=3.0,
            type=float,
            help="""
            The length of the chunks for language id
            """,
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
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs):
        return RNNTransducer.filter_infer_args(**kwargs)
