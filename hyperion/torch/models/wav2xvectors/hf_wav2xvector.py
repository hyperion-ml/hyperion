"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import contextlib
import logging

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import FeatFuserMVN
from ...torch_model import TorchModel
from ...utils import remove_silence


class HFWav2XVector(TorchModel):
    """Abstract Base class for x-vector models that use a Hugging Face Model as feature extractor.

    Attributes:
       hf_feats: hugging face model wrapper object.
       feat_fuser: Dictionary to build feature fuser object.
       xvector: x-vector model object.
       feat_fusion_start: the input to x-vector model will fuse the wav2vec layers from "feat_fusion_start" to
                          the wav2vec "num_layers".
       feat_fusion_method: method to fuse the hidden layers from the wav2vec model, when more
                           than one layer is used (deprecated).
    """

    def __init__(
        self,
        hf_feats,
        feat_fuser,
        xvector,
        feat_fusion_start=0,
        # feat_fusion_method="weighted-avg",
    ):
        super().__init__()
        self.hf_feats = hf_feats
        self.xvector = xvector
        self.feat_fusion_start = feat_fusion_start
        # self.feat_fusion_method = feat_fusion_method
        self._hf_context = contextlib.nullcontext()
        self._make_fuser(feat_fuser)

    def _make_fuser(self, feat_fuser):
        num_feats = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        feat_dim = self.hf_feats.hidden_size
        feat_fuser["feat_fuser"]["num_feats"] = num_feats
        feat_fuser["feat_fuser"]["feat_dim"] = feat_dim
        self.feat_fuser = FeatFuserMVN(**feat_fuser)

    # def _make_fuser_legacy(self):
    #     if self.feat_fusion_method == "last":
    #         self.feat_fuser = None
    #         return

    #     num_layers = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
    #     layer_dim = self.hf_feats.hidden_size
    #     if self.feat_fusion_method == "weighted-avg":
    #         self.feat_fuser = nn.Parameter(torch.zeros(num_layers))
    #     elif self.feat_fusion_method == "linear":
    #         self.feat_fuser = nn.Linear(num_layers, 1, bias=False)
    #         self.feat_fuser.weight.data = torch.ones(1, num_layers) / num_layers
    #     elif self.feat_fusion_method == "cat":
    #         self.feat_fuser = nn.Linear(num_layers * layer_dim, layer_dim, bias=False)

    # def _fuse_hid_feats_legacy(self, hid_feats):
    #     """Fuses the hidden features from the Wav2Vec model.

    #     Args:
    #       hid_feats: list of hidden features Tensors from Wav2Vec model.

    #     Returns:
    #       Tensor of fused features (batch, channels, time)
    #     """
    #     if len(hid_feats) == 1:
    #         # There is only one layer of features
    #         return hid_feats[0]

    #     hid_feats = hid_feats[self.feat_fusion_start :]
    #     if self.feat_fusion_method == "weighted-avg":
    #         hid_feats = torch.stack(hid_feats, dim=-1)
    #         norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
    #         feats = torch.sum(hid_feats * norm_weights, dim=-1)
    #     elif self.feat_fusion_method == "linear":
    #         hid_feats = torch.stack(hid_feats, dim=-1)
    #         feats = self.feat_fuser(hid_feats).squeeze(dim=-1)
    #     elif self.feat_fusion_method == "cat":
    #         hid_feats = torch.cat(hid_feats, dim=-1)
    #         feats = self.feat_fuser(hid_feats)
    #     elif self.feat_fusion_method == "last":
    #         feats = hid_feats[-1]

    #     return feats

    @property
    def sample_frequency(self):
        return self.hf_feats.sample_frequency

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

    def forward_feats(
        self, x, x_lengths, return_feat_layers=None, chunk_length=0, detach_chunks=False
    ):
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

    # def forward_feats_legacy(
    #     self, x, x_lengths, return_feat_layers=None, chunk_length=0, detach_chunks=False
    # ):
    #     return_hid_states = (
    #         False
    #         if return_feat_layers is None and self.feat_fusion_method == "last"
    #         else True
    #     )
    #     with self._hf_context:
    #         hf_output = self.hf_feats(
    #             x,
    #             x_lengths,
    #             return_hid_states=return_hid_states,
    #             chunk_length=chunk_length,
    #             detach_chunks=detach_chunks,
    #         )
    #     feat_lengths = hf_output["hidden_states_lengths"]
    #     if return_hid_states:
    #         hid_feats = hf_output["hidden_states"]
    #         feats = self._fuse_hid_feats(hid_feats)
    #     else:
    #         hid_feats = None
    #         feats = hf_output["last_hidden_state"]

    #     feats = feats.transpose(1, 2)
    #     if return_feat_layers is not None:
    #         # add hidden feats from wav2vec to the output. We transpose to be (batch, C, time)
    #         # as the hidden features of the x-vector encoder.
    #         hid_feats = [
    #             f.transpose(1, 2)
    #             for i, f in enumerate(hid_feats)
    #             if i in return_feat_layers
    #         ]
    #     else:
    #         hid_feats = None

    #     return feats, hid_feats, feat_lengths

    def forward(
        self,
        x,
        x_lengths=None,
        y=None,
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
          return_enc_layers: list of integers indicating, which classification head layers
                             we should return. If None, no head layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Tensor with class logits with shape=(batch, num_classes) or
          Dictionary with "logits", "h_enc" (list of hidden encoder layers),
          "h_classif" (list hidden classification head layers), "h_feats" (wav2vec features)
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

        if not isinstance(output, dict):
            # if the xvector just returned the logits we put then into a dictionary
            # to append the hid feats later.
            output["logits"] = output

        output["h_feats"] = hid_feats
        return output

    def extract_embed(
        self,
        x,
        x_lengths=None,
        vad_samples=None,
        hf_chunk_length=0,
        xvec_chunk_length=0,
        embed_layer=None,
        detach_chunks=False,
    ):
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

    def freeze_feat_fuser(self):
        self.feat_fuser.freeze()
        # if self.feat_fuser is None:
        #     return

        # if self.feat_fusion_method == "weighted-avg":
        #     self.feat_fuser.requires_grad = False
        #     return

        # for param in self.feat_fuser.parameters():
        #     param.requires_grad = False

    def freeze_hf_feats(self):
        self.hf_feats.freeze()

    def freeze_hf_feature_encoder(self):
        self.hf_feats.freeze_feature_encoder()

    def freeze_hf_except_lora(self, bias=None):
        self.hf_feats.freeze_except_lora(bias)

    def has_param_groups(self):
        return self.hf_feats.has_param_groups() or self.xvector.has_param_groups()

    def trainable_param_groups(self):
        if not self.has_param_groups():
            return [{"params": self.trainable_parameters()}]

        param_groups = self.hf_feats.trainable_param_groups()
        param_groups.append({"params": self.feat_fuser.trainable_parameters()})
        param_groups.extend(self.xvector.trainable_param_groups())
        return param_groups

    def set_train_mode(self, mode):
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        elif mode == "ft-embed-affine":
            self.unfreeze()
            self.freeze_feat_fuser()
            self.freeze_hf_feats()
            self.xvector.freeze_preembed_layers()
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

        self._train_mode = mode

    def _train(self, train_mode: str):
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode == "ft-embed-affine":
            self.hf_feats.train()
            self.feat_fuser.train()
            self.xvector._train("ft-embed_affine")
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
    def valid_train_modes():
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
    def filter_args(**kwargs):
        valid_args = (
            "hf_feats",
            "feat_fuser",
            "xvector",
            "feat_fusion_start",
            # "feat_fusion_method",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self):
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

    def change_config(self, hf_feats, xvector):
        logging.info("changing hf wav2xvector config")
        self.hf_feats.change_config(**hf_feats)
        self.xvector.change_config(**xvector)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
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
        # parser.add_argument(
        #     "--feat-fusion-method",
        #     default="weighted-avg",
        #     choices=["weighted-avg", "linear", "cat", "last"],
        #     help=(
        #         "method to fuse the hidden layers from the wav2vec model "
        #         "in [weighted-avg, cat]"
        #     ),
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
