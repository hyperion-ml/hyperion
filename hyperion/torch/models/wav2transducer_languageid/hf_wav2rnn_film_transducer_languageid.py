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
from ..transducer import RNNTransducer, RNNFiLMTransducer, RNNTransducerOutput
from .hf_wav2rnn_transducer_languageid import RNNTransducerLanguageIDOutput
from ..xvectors import ResNet1dXVector as ResNet1dLanguageID
from ...layer_blocks import FiLM


class HFWav2RNNFiLMTransducerLanguageID(TorchModel):
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
                 feat_fusion_method_transducer: str = "film-weighted-avg",
                 feat_fusion_method_lid: str = "weighted-avg",
                 loss_lid_type: str = "weightedCE",
                 loss_class_weight: Optional[torch.Tensor] = None,
                 loss_class_weight_exp= 1.0,
                 loss_weight_transducer: float = 0.005,
                 loss_weight_lid: float = 1.0,
                 loss_weight_embed: float = 0.005,
                 loss_reg_weight_transducer: float = 0.0,
                 loss_reg_weight_lid: float = 0.0,
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
            transducer = RNNFiLMTransducer(**transducer)
        else:
            assert isinstance(transducer, RNNFiLMTransducer)
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
        self.loss_weight_embed = loss_weight_embed
        self.loss_reg_weight_transducer = loss_reg_weight_transducer
        self.loss_reg_weight_lid = loss_reg_weight_lid
        self.lid_length = lid_length
        self._hf_context = contextlib.nullcontext()
        self.transducer_fuser, self.film, self.lid_film = self._make_fuser(self.feat_fusion_method_transducer, self.feat_fusion_start_transducer)
        self.languageid_fuser, _, _ = self._make_fuser(self.feat_fusion_method_lid, self.feat_fusion_start_lid)

    def _make_fuser(self, method, start):
        feat_fuser = None
        film = None
        lid_film = None
        if method == "last":
            return feat_fuser, None, None
        num_layers = self.hf_feats.num_encoder_layers + 1 - start
        layer_dim = self.hf_feats.hidden_size
        if method == "film-weighted-avg":
            film = nn.ModuleList([FiLM(layer_dim, self.transducer.decoder.condition_size) for _ in range(num_layers)])
            lid_film = nn.ModuleList([FiLM(layer_dim, self.transducer.decoder.condition_size) for _ in range(num_layers)])
            feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif method == "film-fused-feature":
            feat_fuser = nn.Parameter(torch.zeros(num_layers))
            film = FiLM(layer_dim, self.transducer.decoder.condition_size)
            lid_film = FiLM(layer_dim, self.transducer.decoder.condition_size)
        elif method == "weighted-avg":
            feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif method == "linear":
            feat_fuser = nn.Linear(num_layers, 1, bias=False)
            feat_fuser.weight.data = torch.ones(1,
                                                     num_layers) / num_layers
        elif method == "cat":
            feat_fuser = nn.Linear(num_layers * layer_dim,
                                        layer_dim,
                                        bias=False)

        return feat_fuser, film, lid_film

    def _fuse_transducer_hid_feats(self, hid_feats, lang_condition):
        """Fuses the hidden features from the Wav2Vec model.

        Args:
          hid_feats: list of hidden features Tensors from Wav2Vec model.
          lang: language id Tensor.

        Returns:
          Tensor of fused features (batch, channels, time)
        """
        if len(hid_feats) == 1:
            # There is only one layer of features
            return hid_feats[0]

        if self.transducer.decoder.film_cond_type in ["one-hot", "lid_pred"]:
            lang_condition = self.transducer.decoder.lang_embedding(lang_condition)
        hid_feats = hid_feats[self.feat_fusion_start_transducer:]
        if self.feat_fusion_method_transducer == "film-weighted-avg":
            film_hid_feats = tuple(self.lid_film[i](hid_feats[i], lang_condition) for i in range(len(self.lid_film)))
            film_hid_feats = torch.stack(film_hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.transducer_fuser, dim=-1)
            feats = torch.sum(film_hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method_transducer == "film-fused-feature":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.transducer_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
            feats = self.lid_film(feats, lang_condition)
        elif self.feat_fusion_method_transducer == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.transducer_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method_transducer == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats = self.transducer_fuser(hid_feats).squeeze(dim=-1)
        elif self.feat_fusion_method_transducer == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats = self.transducer_fuser(hid_feats)
        elif self.feat_fusion_method_transducer == "last":
            feats = hid_feats[-1]

        return feats


    def _fuse_lid_hid_feats(self, hid_feats):
        """Fuses the hidden features from the Wav2Vec model.

        Args:
          hid_feats: list of hidden features Tensors from Wav2Vec model.

        Returns:
          Tensor of fused features (batch, channels, time)
        """
        if len(hid_feats) == 1:
            # There is only one layer of features
            return hid_feats[0]

        hid_feats = hid_feats[self.feat_fusion_start_lid:]
        if self.feat_fusion_method_lid == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.languageid_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method_lid == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats = self.languageid_fuser(hid_feats).squeeze(dim=-1)
        elif self.feat_fusion_method_lid == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats = self.languageid_fuser(hid_feats)
        elif self.feat_fusion_method_lid == "last":
            feats = hid_feats[-1]

        return feats

    def forward_lid_feats(self,
                      x,
                      x_lengths,
                      lang=None,
                      return_feat_layers=None,
                      chunk_length=0,
                      detach_chunks=False):
        with self._hf_context:
            hf_output = self.hf_feats(
                x,
                x_lengths,
                return_hid_states=True,
                chunk_length=chunk_length,
                detach_chunks=detach_chunks,
            )
        feat_lengths = hf_output["hidden_states_lengths"]
        
        hid_feats = hf_output["hidden_states"]
        feats = self._fuse_lid_hid_feats(hid_feats)
        

        feats = feats.transpose(1, 2)

        return feats, hid_feats, feat_lengths
            
    def compute_embed_loss(self, lang_embed, languageid):
        # comput the loss for the embeding between the film and lid_film
        lang_condition = self.transducer.decoder.lang_embedding(languageid)

        # for the encoder
        film_scale = self.film.linear_scale(lang_condition)
        lid_film_scale = self.lid_film.linear_scale(lang_embed)
        film_shift = self.film.linear_shift(lang_condition)
        lid_film_shift = self.lid_film.linear_shift(lang_embed)
        loss_embed_encode = torch.mean(torch.abs(film_scale - lid_film_scale)) + torch.mean(torch.abs(film_shift - lid_film_shift))

        # for the predictor
        loss_embed_predictor = 0
        for i in range(2):
            film_scale = self.transducer.decoder.predictor.rnn.films[i].linear_scale(lang_condition)
            lid_film_scale = self.transducer.decoder.predictor.rnn.lid_films[i].linear_scale(lang_embed)
            film_shift = self.transducer.decoder.predictor.rnn.films[i].linear_shift(lang_condition)
            lid_film_shift = self.transducer.decoder.predictor.rnn.lid_films[i].linear_shift(lang_embed)
            loss_embed_predictor += torch.mean(torch.abs(film_scale - lid_film_scale)) + torch.mean(torch.abs(film_shift - lid_film_shift))
            

        # for the joiner
        film_scale = self.transducer.decoder.joiner.film.linear_scale(lang_condition)
        lid_film_scale = self.transducer.decoder.joiner.lid_film.linear_scale(lang_embed)
        film_shift = self.transducer.decoder.joiner.film.linear_shift(lang_condition)
        lid_film_shift = self.transducer.decoder.joiner.lid_film.linear_shift(lang_embed)
        loss_embed_joiner = torch.mean(torch.abs(film_scale - lid_film_scale)) + torch.mean(torch.abs(film_shift - lid_film_shift))


        loss_embed = loss_embed_encode + loss_embed_predictor + loss_embed_joiner

        return loss_embed

    def forward(
        self,
        x,
        x_lengths=None,
        text=None,
        languageid=None,
        return_feat_layers=None,
        return_enc_layers=None,
        return_classif_layers=[0],
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
        feats_languageid, hid_feats, feat_lengths = self.forward_lid_feats(
            x, x_lengths, return_feat_layers)

        lid_len = int(self.lid_length * 50)
        min_len = torch.min(feat_lengths).item()
        if min_len > lid_len:
            lid_start = torch.randint(0, min_len - lid_len + 1, (1,)).item()
            feats_languageid = feats_languageid[:, :, lid_start: lid_start + lid_len]


        output = self.languageid(
            feats_languageid,
            None,
            languageid,
            return_enc_layers=return_enc_layers,
            return_classif_layers=return_classif_layers,
            return_logits=return_logits,
        )
        # output["h_classif"] = h_classif
        # output["logits"] = y_pred

        #loss_lid = self.loss_lid(lid_logits, languageid)
        loss_lid = self.loss_lid(output["logits"], languageid)
        # import pdb; pdb.set_trace()
        # logging.info(output["h_classif"])

        loss_embed = self.compute_embed_loss(output["h_classif"][0], languageid)
        
        # feats_transducer = self._fuse_transducer_hid_feats(hid_feats, lid_logits) # (N, T, C)
        feats_transducer = self._fuse_transducer_hid_feats(hid_feats, output["h_classif"][0]) # (N, T, C)
            
        trans_output = self.transducer(
            feats_transducer,
            feat_lengths,
            text,
            output["h_classif"][0]
            # lid_logits
        )

        if return_feat_layers:
            trans_output.h_feats = [
                f.transpose(1, 2) for i, f in enumerate(hid_feats)
                if i in return_feat_layers
            ]

        loss_reg_lid = 0
        if self.loss_reg_weight_lid > 0:
            loss_reg_lid = self.languageid.get_regularization_loss()
            
        loss_reg_transducer = 0
        if self.loss_reg_weight_transducer > 0:
            loss_reg_transducer = self.transducer.get_regularization_loss()



        output = RNNTransducerLanguageIDOutput(loss=self.loss_weight_transducer * trans_output.loss + self.loss_weight_lid * loss_lid + self.loss_weight_embed * loss_embed + self.loss_reg_weight_lid * loss_reg_lid + self.loss_reg_weight_transducer * loss_reg_transducer, 
                                                loss_transducer=trans_output.loss, 
                                                loss_lid=loss_lid,
                                                loss_embed=loss_embed,
                                                loss_transducer_simple=trans_output.loss_simple, 
                                                loss_transducer_pruned=trans_output.loss_pruned,
                                                h_feats=trans_output.h_feats,
                                                logits=output["logits"] if return_logits else None)
                                                # logits=lid_logits if return_logits else None)
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


        feats_languageid, hid_feats, feat_lengths = self.forward_lid_feats(
            x, x_lengths, None)
        # logging.info(f"feat_lengths: {feat_lengths}")
        # logging.info(f"feats_languageid.shape: {feats_languageid.shape}")
        # logging.info(f"feats_languageid: {feats_languageid}")


        output = self.languageid(
            feats_languageid,
            None,
            None,
            return_enc_layers=None,
            return_classif_layers=[0],
            return_logits=True,
        )

        # output = self.languageid(
        #     feats_languageid,
        #     feat_lengths,
        #     None,
        #     return_enc_layers=None,
        #     return_classif_layers=[0],
        #     return_logits=True,
        # )
        
        feats_transducer = self._fuse_transducer_hid_feats(hid_feats, output["h_classif"][0])  # (N, T, C)
            

        text = self.transducer.infer(feats_transducer,
                                  feat_lengths,
                                  lang=output["h_classif"][0],
                                  decoding_method=decoding_method,
                                  beam_width=beam_width,
                                  max_sym_per_frame=max_sym_per_frame,
                                  max_sym_per_utt=max_sym_per_utt)
                                  
        return text, output["logits"]

    def unfreeze_lid_film(self):
        for name, param in self.named_parameters():
            if "lid_film" in name:
                logging.info(f"unfreezing {name}")
                param.requires_grad = True

    def freeze_lid(self):
        self.languageid.freeze()

    def freeze_film(self):
        for name, param in self.named_parameters():
            # logging.info(f"parameter {name}")
            if "film" in name and "lid_film" not in name:
                logging.info(f"freezing {name}")
                param.requires_grad = False
            if "lang_embedding" in name:
                logging.info(f"freezing {name}")
                param.requires_grad = False

    def freeze_lid_feat_fuser(self):
        if self.languageid_fuser is None:
            return

        if self.feat_fusion_method_lid == "weighted-avg":
            self.languageid_fuser.requires_grad = False
            return

        for param in self.languageid_fuser.parameters():
            param.requires_grad = False

    def freeze_hf_feats(self):
        self.hf_feats.freeze()

    def freeze_hf_feature_encoder(self):
        self.hf_feats.freeze_feature_encoder()

    def set_train_mode(self, mode):
        logging.info("setting train mode to %s", mode)
        logging.info("train mode was %s", self._train_mode)
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        if mode == "freeze-gt-film":
            self.unfreeze()
            self.freeze_film()
        elif mode == "frozen":
            self.freeze()
        elif mode in ["ft-film", "ft-film-grad"]:
            self.freeze()
            self.unfreeze_lid_film()
        elif mode in ["ft-transducer", "ft-transducer-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
            self.freeze_film()
            self.freeze_lid_feat_fuser()
            self.freeze_lid()
            # self.unfreeze_lid_film()
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
                "ft-film",
                "freeze-gt-film",
                "ft-transducer",
                "hf-feats-frozen",
                "ft-film-grad",
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
            "freeze-gt-film",
            "ft-film",
            "ft-embed-affine",
            "ft-transducer",
            "ft-film-grad",
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
            "loss_weight_embed",
            "loss_reg_weight_transducer",
            "loss_reg_weight_lid",
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
            "loss_weight_embed": self.loss_weight_embed,
            "loss_reg_weight_transducer": self.loss_reg_weight_transducer,
            "loss_reg_weight_lid": self.loss_reg_weight_lid,
            "lid_length": self.lid_length,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, loss_weight_transducer, loss_weight_lid, loss_weight_embed, loss_reg_weight_transducer, loss_reg_weight_lid, lid_length, hf_feats, transducer, languageid):
        logging.info("changing hf wav2film_transducer_languageid config")

        self.loss_weight_transducer = loss_weight_transducer
        self.loss_weight_lid = loss_weight_lid
        self.loss_weight_embed = loss_weight_embed
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
            "--loss-weight-embed",
            default=0.005,
            type=float,
            help="""
            The weight of the embedding loss
            """,
        )
        parser.add_argument(
            "--loss-reg-weight-transducer",
            default=0.0,
            type=float,
            help="""
            The weight of the transducer regularization loss
            """,
        )
        parser.add_argument(
            "--loss-reg-weight-lid",
            default=0.0,
            type=float,
            help="""
            The weight of the lid regularization loss
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

        RNNFiLMTransducer.add_infer_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs):
        return RNNFiLMTransducer.filter_infer_args(**kwargs)
