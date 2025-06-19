"""
 Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
#from enum import Enum
from typing import List, Dict, Optional, Union, Any

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HypDataClass
from ....utils.misc import filter_func_args
from ...narchs import ProjHead, TorchNALoader, QFormerV2
from ...torch_model import TorchModel
from ...utils import eval_nnet_by_chunks, scale_seq_lengths


@dataclass
class QVectorOutput(HypDataClass):
    loss: torch.Tensor
    logits: torch.Tensor
    qvector: torch.Tensor
    qmatrix: torch.Tensor
    # hidden_enc: Optional[List[torch.Tensor]] = None
    # h_classif: Optional[List[torch.Tensor]] = None
    # h_feats: Optional[List[torch.Tensor]] = None


class QVector(TorchModel):
    """q-Vector base class"""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_feats_agg_qformer: Union[Dict[str, Any], None],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Union[Dict[str, Any], None],
        num_output_feats_queries: int,
        qvector_dim: int,
        classif_head: Dict[str, Any],
        bias_weight_decay=None,
    ):
        super().__init__(bias_weight_decay=bias_weight_decay)

        # encoder network
        self.encoder = encoder

        assert num_hidden_feats_queries > 0 or num_output_feats_queries > 0
        self.num_hidden_feats_queries = num_hidden_feats_queries
        self.num_output_feats_queries = num_output_feats_queries
        self.qvector_dim = qvector_dim

        query_dim = None
        self.return_enc_layers = None
        self.enc_layers_dims = None
        if num_hidden_feats_queries > 0:
            query_dim = hidden_feats_agg_qformer["hidden_dim"]
            hidden_feats_agg_qformer["multilayer_input"] = True
            self.hidden_feats_queries = nn.Parameter()
            self.hidden_feats_agg_qformer = QFormerV2(**hidden_feats_agg_qformer)
        else:
            self.hidden_feats_queries = None
            self.hidden_feats_agg_qformer = None

        if output_feats_queries["num_layers"] > 0:
            if query_dim is not None:
                assert query_dim == output_feats_queries["hidden_dim"]

            self.output_feats_queries = nn.Parameter()
            output_feats_agg_qformer["multilayer_input"] = False
            self.output_feats_agg_qformer = QFormerV2(**output_feats_agg_qformer)
        else:
            self.output_feats_queries = None
            self.output_feats_agg_qformer = None

        qmatrix_dim = (num_hidden_feats_queries+num_output_feats_queries) * query_dim
        self.proj_head = ProjHead(in_feats=qmatrix_dim, out_feats=qvector_dim)
        classif_head["in_feats"] = qvector_dim
        self.classif_head = ClassifHead(**classif_head)


    @property
    def has_hidden_feats_agg(self):
        return self.hidden_feats_agg_qformer is not None

    @property
    def has_output_feats_agg(self):
        return self.output_feats_agg_qformer is not None

    def _infer_enc_layers_indeces_and_dims(self, qformer_cfg):
        raise NotImplementedError()

    @property
    def num_classes(self):
        return self.classif_net.num_classes

    @property
    def cos_scale(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.cos_scale
        elif self.head_type == XVectorHeadType.DINO:
            return 1
        else:
            raise ValueError

    @property
    def margin(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.margin
        else:
            return 0.0

    @property
    def margin_warmup_epochs(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.margin_warmup_epochs
        else:
            return 0

    @property
    def intertop_k(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.intertop_k
        else:
            return 0

    @property
    def intertop_margin(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.intertop_margin
        else:
            return 0.0

    @property
    def num_subcenters(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.num_subcenters
        else:
            return 0

    @property
    def loss_type(self):
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.loss_type
        elif self.head_type == XVectorHeadType.DINO:
            return self.classif_net.output_type
        else:
            raise ValueError()

    
    def update_loss_margin(self, epoch):
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.classif_net.update_margin(epoch)

    def _pre_enc(self, x):
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        return x

    def _post_enc(self, x, in_lengths=None, max_in_length=None):
        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        if in_lengths is not None:
            out_lengths = scale_seq_lengths(in_lengths, x.size(-1), max_in_length)
        else:
            out_lengths = None

        return x, out_lengths

    def forward(
        self,
        x,
        x_lengths=None,
        y=None,
        return_enc_layers=None,
        return_classif_layers=None,
        return_logits=True,
    ):
        """Forward function. If returns the logits posteriors of the classes.
        It can also returns the hidden representations in the encoder and
        classification head. In this case the ouput variable is a dictionary.

        Args:
          x: input features tensor with shape=(batch, in_feats, time).
          x_lengths: time lengths of the features with shape=(batch,).
          y: target classes torch.long tensor with shape=(batch,).
          return_enc_layers: list of integers indicating, which encoder layers
                             we should return. If None, no encoder layers are returned.
          return_enc_layers: list of integers indicating, which classification head layers
                             we should return. If None, no head layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Tensor with class logits with shape=(batch, num_classes) or
          Dictionary with "logits", "h_enc" (list of hidden encoder layers),
          "h_classif" (list hidden classification head layers).
        """

        if return_enc_layers is None and return_classif_layers is None:
            return self.forward_logits(x, x_lengths, y)

        return self.forward_hid_feats(
            x, x_lengths, y, return_enc_layers, return_classif_layers, return_logits
        )

    def forward_logits(self, x, x_lengths=None, y=None):
        """Forward function

        Args:
          x: input features tensor with shape=(batch, in_feats, time).
          x_lengths: time lengths of the features with shape=(batch,).
          y: target classes torch.long tensor with shape=(batch,).

        Returns:
          class logits tensor with shape=(batch, num_classes).
        """
        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        x = self.encoder_net(x)
        if isinstance(x, tuple):
            x = x[0]

        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
        p = self.pool_net(x, x_lengths=x_lengths)
        if not torch.all(torch.isfinite(p)):
            logging.warning("non-finite p-avg=%f", torch.mean(p))
        xvector = None
        if self.proj_head_net is not None:
            p = self.proj_head_net(p)
            xvector = p

        logits = self.classif_net(p, y)
        if not torch.all(torch.isfinite(logits)):
            logging.warning("non-finite y-avg=%f", torch.mean(logits))
        # return logits
        output = XVectorOutput(None, logits, xvector)
        return output

    def forward_hid_feats(
        self,
        x,
        x_lengths=None,
        y=None,
        return_enc_layers=None,
        return_classif_layers=None,
        return_logits=False,
    ):
        """forwards hidden representations in the x-vector network

        Args:
          x: input features tensor with shape=(batch, in_feats, time).
          x_lengths: time lengths of the features with shape=(batch,).
          y: target classes torch.long tensor with shape=(batch,).
          return_enc_layers: list of integers indicating, which encoder layers
                             we should return. If None, no encoder layers are returned.
          return_enc_layers: list of integers indicating, which classification head layers
                             we should return. If None, no head layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Dictionary with "logits", "h_enc" (list of hidden encoder layers),
          "h_classif" (list hidden classification head layers).
        """
        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        h_enc, x = self.encoder_net.forward_hid_feats(
            x, return_enc_layers, return_output=True
        )
        output = {"h_enc": h_enc}
        if not return_logits and return_classif_layers is None:
            return output

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        if self.proj_head_net is not None:
            p = self.proj_head_net(p)
        h_classif = self.classif_net.forward_hid_feats(
            p, y, return_classif_layers, return_logits=return_logits
        )
        if return_logits:
            h_classif, y_pred = h_classif
        else:
            y_pred = None

        if h_classif is not None:
            xvector = h_classif[0]
        else:
            xvector = None

        output = XVectorOutput(None, y_pred, xvector, h_enc, h_classif)
        return output

    # def forward_hid_feats(
    #     self,
    #     x,
    #     x_lengths=None,
    #     y=None,
    #     return_enc_layers=None,
    #     return_classif_layers=None,
    #     return_logits=False,
    # ):
    #     """forwards hidden representations in the x-vector network

    #     Args:
    #       x: input features tensor with shape=(batch, in_feats, time).
    #       x_lengths: time lengths of the features with shape=(batch,).
    #       y: target classes torch.long tensor with shape=(batch,).
    #       return_enc_layers: list of integers indicating, which encoder layers
    #                          we should return. If None, no encoder layers are returned.
    #       return_enc_layers: list of integers indicating, which classification head layers
    #                          we should return. If None, no head layers are returned.
    #       return_logits: if True, it adds the logits to the output dictionary.
    #     Returns:
    #       Dictionary with "logits", "h_enc" (list of hidden encoder layers),
    #       "h_classif" (list hidden classification head layers).
    #     """
    #     max_in_length = x.size(-1)
    #     x = self._pre_enc(x)
    #     h_enc, x = self.encoder_net.forward_hid_feats(
    #         x, return_enc_layers, return_output=True
    #     )
    #     output = {"h_enc": h_enc}
    #     if not return_logits and return_classif_layers is None:
    #         return output

    #     x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
    #     p = self.pool_net(x, x_lengths=x_lengths)
    #     if self.proj_head_net is not None:
    #         p = self.proj_head_net(p)
    #     h_classif = self.classif_net.forward_hid_feats(
    #         p, y, return_classif_layers, return_logits=return_logits
    #     )
    #     if return_logits:
    #         h_classif, y_pred = h_classif
    #         output["h_classif"] = h_classif
    #         output["logits"] = y_pred
    #         return output

    #     output["h_classif"] = h_classif
    #     return output

    def extract_embed_impl(
        self, x, x_lengths=None, chunk_length=0, embed_layer=None, detach_chunks=False
    ):
        if embed_layer is None:
            embed_layer = self.embed_layer

        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        if max_in_length <= chunk_length or chunk_length == 0:
            x = self.encoder_net(x, x_lengths=x_lengths)
            if isinstance(x, tuple):
                x = x[0]
        else:
            x = eval_nnet_by_chunks(
                x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
            )

            if x.device != self.device:
                x = x.to(self.device)

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        if self.proj_head_net is not None:
            return self.proj_head_net(p)

        y = self.classif_net.extract_embed(p, embed_layer)
        return y

    def extract_embed(
        self, x, x_lengths=None, chunk_length=0, embed_layer=None, detach_chunks=False
    ):

        if x.size(-1) <= chunk_length or chunk_length == 0:
            return self.extract_embed_impl(x, x_lengths, 0, embed_layer)
        else:
            e = []
            for i in range(x.size(0)):
                x_i = x[i : i + 1]
                if x_lengths is not None:
                    x_i = x_i[..., x_lengths[i]]

                e_i = self.extract_embed_impl(
                    x_i,
                    chunk_length=chunk_length,
                    embed_layer=embed_layer,
                    detach_chunks=detach_chunks,
                )
                e.append(e_i)

            return torch.cat(e, dim=0)

    def extract_embed_slidwin_legacy(
        self,
        x,
        win_length,
        win_shift,
        snip_edges=False,
        feat_frame_length=None,
        feat_frame_shift=None,
        chunk_length=0,
        embed_layer=None,
        detach_chunks=False,
    ):
        if feat_frame_shift is not None:
            # assume win_length/shift are in secs, transform to frames
            # pass feat times from msecs to secs
            feat_frame_shift = feat_frame_shift / 1000
            feat_frame_length = feat_frame_length / 1000

            # get length and shift in number of feature frames
            win_shift = win_shift / feat_frame_shift  # this can be a float
            win_length = (
                win_length - feat_frame_length + feat_frame_shift
            ) / feat_frame_shift
            assert win_shift > 0.5, "win-length should be longer than feat-frame-length"

        if embed_layer is None:
            embed_layer = self.embed_layer

        in_time = x.size(-1)
        x, _ = self._pre_enc(x)
        x = eval_nnet_by_chunks(
            x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
        )

        if x.device != self.device:
            x = x.to(self.device)

        x = self._post_enc(x)
        pin_time = x.size(-1)  # time dim before pooling
        downsample_factor = float(pin_time) / in_time
        p = self.pool_net.forward_slidwin(
            x,
            downsample_factor * win_length,
            downsample_factor * win_shift,
            snip_edges=snip_edges,
        )
        # (batch, pool_dim, time)

        p = p.transpose(1, 2).contiguous().view(-1, p.size(1))
        y = (
            self.classif_net.extract_embed(p, embed_layer)
            .view(x.size(0), -1, self.embed_dim)
            .transpose(1, 2)
            .contiguous()
        )

        return y

    def compute_slidwin_timestamps(
        self,
        num_windows,
        win_length,
        win_shift,
        snip_edges=False,
        feat_frame_length=25,
        feat_frame_shift=10,
        feat_snip_edges=False,
    ):
        P = self.compute_slidwin_left_padding(
            win_length,
            win_shift,
            snip_edges,
            feat_frame_length,
            feat_frame_shift,
            feat_snip_edges,
        )

        tstamps = (
            torch.as_tensor(
                [
                    [i * win_shift, i * win_shift + win_length]
                    for i in range(num_windows)
                ]
            )
            - P
        )
        tstamps[tstamps < 0] = 0
        return tstamps

    def compute_slidwin_left_padding(
        self,
        win_length,
        win_shift,
        snip_edges=False,
        feat_frame_length=25,
        feat_frame_shift=10,
        feat_snip_edges=False,
    ):
        # pass feat times from msecs to secs
        feat_frame_shift = feat_frame_shift / 1000
        feat_frame_length = feat_frame_length / 1000

        # get length and shift in number of feature frames
        H = win_shift / feat_frame_shift
        L = (win_length - feat_frame_length + feat_frame_shift) / feat_frame_shift
        assert L > 0.5, "win-length should be longer than feat-frame-length"

        # compute left padding in case of snip_edges is False
        if snip_edges:
            P1 = 0
        else:
            Q = (
                L - H
            ) / 2  # left padding in frames introduced by x-vector sliding window
            P1 = (
                Q * feat_frame_shift
            )  # left padding in secs introduced by x-vector sliding window

        if feat_snip_edges:
            # left padding introduced when computing acoustic feats
            P2 = 0
        else:
            P2 = (feat_frame_length - feat_frame_shift) / 2

        # total left padding
        return P1 + P2

    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        pool_cfg = PF.get_config(self.pool_net)
        config = {
            "encoder_cfg": enc_cfg,
            "pool_net": pool_cfg,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_embed_layers": self.num_embed_layers,
            "hid_act": self.hid_act,
            "loss_type": self.loss_type,
            "cos_scale": self.cos_scale,
            "margin": self.margin,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "intertop_k": self.intertop_k,
            "intertop_margin": self.intertop_margin,
            "num_subcenters": self.num_subcenters,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "head_norm_layer": self.head_norm_layer,
            "head_use_norm": self.head_use_norm,
            "head_use_in_norm": self.head_use_in_norm,
            "head_hid_dim": self.head_hid_dim,
            "head_bottleneck_dim": self.head_bottleneck_dim,
            "proj_head_use_norm": self.proj_head_use_norm,
            "proj_head_norm_before": self.proj_head_norm_before,
            "dropout_rate": self.dropout_rate,
            "embed_layer": self.embed_layer,
            "in_feats": self.in_feats,
            "proj_feats": self.proj_feats,
            "head_type": self.head_type,
            "bias_weight_decay": self.bias_weight_decay,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        encoder_net = TorchNALoader.load_from_cfg(cfg=cfg["encoder_cfg"])
        for k in "encoder_cfg":
            del cfg[k]

        model = cls(encoder_net, **cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    def change_config(
        self,
        override_output=False,
        override_dropouts=False,
        dropout_rate=0,
        num_classes=None,
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=10,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
        head_type=XVectorHeadType.XVECTOR,
    ):
        logging.info("changing x-vector config")
        if override_output:
            self.rebuild_output_layer(
                num_classes=num_classes,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                head_type=head_type,
            )

        if override_dropouts:
            logging.info("overriding x-vector dropouts")
            self.encoder_net.change_dropouts(dropout_rate)
            self.classif_net.change_dropouts(dropout_rate)

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
        head_type=XVectorHeadType.XVECTOR,
    ):

        if head_type != self.head_type:
            # only from dino to x-vector
            assert self.head_type == XVectorHeadType.DINO
            logging.info("transforming dino head into x-vector head")
            self.num_embed_layers = 1
            self.head_use_in_norm = (
                self.proj_head_use_norm and self.proj_head_norm_before
            )
            self.head_use_norm = (
                self.proj_head_use_norm and not self.proj_head_norm_before
            )
            self.classif_net = ClassifHead(
                self.proj_head_net.in_feats,
                num_classes,
                embed_dim=self.proj_head_net.out_feats,
                num_embed_layers=1,
                hid_act=None,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                norm_layer=self.head_norm_layer,
                use_norm=self.proj_head_use_norm,
                norm_before=self.norm_before,
                dropout_rate=self.dropout_rate,
                use_in_norm=self.head_use_in_norm,
            )

            if (
                self.classif_net.fc_blocks[0].linear.bias is not None
                and self.proj_head_net.proj.bias is not None
            ):
                self.classif_net.fc_blocks[0].linear.bias.data.copy_(
                    self.proj_head_net.proj.bias.data
                )

            self.classif_net.fc_blocks[0].linear.weight.data.copy_(
                self.proj_head_net.proj.weight.data
            )
            if self.head_use_norm:
                self.classif_net.fc_blocks[0].bn1.load_state_dict(
                    self.proj_head_net._norm_layer.state_dict()
                )
            del self.proj_head_net
            self.proj_head_net = None
            self.head_type = XVectorHeadType.XVECTOR
            return

        if (
            (self.num_classes is not None and self.num_classes != num_classes)
            or (self.loss_type != loss_type)
            or (
                loss_type == "subcenter-arc-softmax"
                and self.classif_net.num_subcenters != num_subcenters
            )
        ):
            # if we change the number of classes or the loss-type
            # we need to reinitiate the last layer
            logging.info("rebuilding output layer")
            self.classif_net.rebuild_output_layer(
                num_classes,
                loss_type,
                cos_scale,
                margin,
                margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
            )
            return

        # otherwise we just change the values of s, margin and margin_warmup
        self.classif_net.set_margin(margin)
        self.classif_net.set_margin_warmup_epochs(margin_warmup_epochs)
        self.classif_net.set_cos_scale(cos_scale)
        self.classif_net.set_intertop_k(intertop_k)
        self.classif_net.set_intertop_margin(intertop_margin)
        self.classif_net.set_num_subcenters(num_subcenters)

    def cancel_output_layer_grads(self):
        for p in self.classif_net.output.parameters():
            p.grad = None

    def freeze_preembed_layers(self):
        self.encoder_net.freeze()
        if self.proj is not None:
            self.proj.freeze()

        for param in self.pool_net.parameters():
            param.requires_grad = False

        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.freeze_layers(layer_list)

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

        if self.head_type == XVectorHeadType.DINO:
            self.classif_net.freeze_output_g()

        self._train_mode = mode

    def _train(self, train_mode: str):
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode == "ft-embed-affine":
            self.encoder_net.eval()
            if self.proj is not None:
                self.proj.eval()

            self.pool_net.eval()
            self.classif_net.train()
            layer_list = [l for l in range(self.embed_layer)]
            self.classif_net.put_layers_in_eval_mode(layer_list)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def compute_prototype_affinity(self):
        return self.classif_net.compute_prototype_affinity()

    @staticmethod
    def valid_train_modes():
        return ["full", "frozen", "ft-embed-affine"]

    @staticmethod
    def filter_args(**kwargs):
        # get arguments for pooling
        pool_args = PF.filter_args(**kwargs["pool_net"])
        args = filter_func_args(XVector.__init__, kwargs)
        args["pool_net"] = pool_args
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        PF.add_class_args(
            parser, prefix="pool_net", skip=["dim", "in_feats", "keepdim"]
        )

        parser.add_argument(
            "--head-type",
            default=XVectorHeadType.XVECTOR.value,
            choices=XVectorHeadType.choices(),
            help="type of classification head in [x-vector, dino]",
        )

        parser.add_argument(
            "--embed-dim", default=256, type=int, help=("x-vector dimension")
        )

        parser.add_argument(
            "--num-embed-layers",
            default=1,
            type=int,
            help=("number of layers in the classif head"),
        )

        try:
            parser.add_argument("--hid-act", default="relu", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--loss-type",
            default="arc-softmax",
            choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
            help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
        )

        parser.add_argument(
            "--cos-scale", default=64, type=float, help="scale for arcface"
        )

        parser.add_argument(
            "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
        )

        parser.add_argument(
            "--margin-warmup-epochs",
            default=10,
            type=float,
            help="number of epoch until we set the final margin",
        )

        parser.add_argument(
            "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
        )
        parser.add_argument(
            "--intertop-margin",
            default=0.0,
            type=float,
            help="margin for InterTopK penalty",
        )

        parser.add_argument(
            "--num-subcenters",
            default=2,
            type=int,
            help="number of subcenters in subcenter losses",
        )

        try:
            parser.add_argument(
                "--norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help="type of normalization layer for all components of x-vector network",
            )
        except:
            pass

        try:
            parser.add_argument(
                "--head-norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help=(
                    "type of normalization layer for classification head, "
                    "it overrides the value of the norm-layer parameter"
                ),
            )
        except:
            pass

        parser.add_argument(
            "--use-norm",
            default=True,
            action=ActionYesNo,
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton before activation",
        )

        parser.add_argument(
            "--head-use-norm",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton at the head",
        )
        parser.add_argument(
            "--head-use-in-norm",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton at the head input",
        )

        parser.add_argument(
            "--head-hid-dim",
            default=2048,
            type=int,
            help="bottleneck dim of DINO head",
        )

        parser.add_argument(
            "--head-bottleneck-dim",
            default=256,
            type=int,
            help="bottleneck dim of DINO head",
        )

        parser.add_argument(
            "--proj-head-use-norm",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton at projection head",
        )
        parser.add_argument(
            "--proj-head-norm-before",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton at the begining of projection head",
        )

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats",
                default=None,
                type=int,
                help=(
                    "input feature dimension, "
                    "if None it will try to infer from encoder network"
                ),
            )

        parser.add_argument(
            "--proj-feats",
            default=None,
            type=int,
            help=(
                "dimension of linear projection after encoder network, "
                "if None, there is not projection"
            ),
        )

        parser.add_argument(
            "--bias-weight-decay",
            default=None,
            type=float,
            help="biases weight decay, if None default it is used",
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
                help="xvector options",
            )

    @staticmethod
    def filter_finetune_args(**kwargs):
        args = filter_func_args(XVector.change_config, kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--override-output",
            default=False,
            action=ActionYesNo,
            help="changes the config of the output layer",
        )

        parser.add_argument(
            "--loss-type",
            default="arc-softmax",
            choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
            help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
        )

        parser.add_argument(
            "--cos-scale", default=64, type=float, help="scale for arcface"
        )

        parser.add_argument(
            "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
        )

        parser.add_argument(
            "--margin-warmup-epochs",
            default=10,
            type=float,
            help="number of epoch until we set the final margin",
        )

        parser.add_argument(
            "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
        )
        parser.add_argument(
            "--intertop-margin",
            default=0.0,
            type=float,
            help="margin for InterTopK penalty",
        )

        parser.add_argument(
            "--num-subcenters",
            default=2,
            type=int,
            help="number of subcenters in subcenter losses",
        )

        try:
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        except:
            pass

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs):
        return XVector.filter_finetune_args(**kwargs)

    @staticmethod
    def add_dino_teacher_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        try:
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        except:
            pass

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
