"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from enum import Enum
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo
from typing import Optional

import torch
import torch.nn as nn

from ...layers import GlobalPool1dFactory as PF
from ...layer_blocks import TDNNBlock
from ...narchs import ClassifHead, TorchNALoader
from ...torch_model import TorchModel
from ...utils import eval_nnet_by_chunks, scale_seq_lengths


class XVectorTrainMode(Enum):
    full = 0
    frozen = 1
    ft_embed_affine = 2


class XVector(TorchModel):
    """x-Vector base class"""

    def __init__(
        self,
        encoder_net,
        num_classes,
        pool_net="mean+stddev",
        embed_dim=256,
        num_embed_layers=1,
        hid_act={"name": "relu", "inplace": True},
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
        norm_layer=None,
        head_norm_layer=None,
        use_norm=True,
        norm_before=True,
        dropout_rate=0,
        embed_layer=0,
        in_feats=None,
        proj_feats=None,
    ):

        super().__init__()

        # encoder network
        self.encoder_net = encoder_net

        # infer input and output shapes of encoder network
        in_shape = self.encoder_net.in_shape()
        if len(in_shape) == 3:
            # encoder based on 1d conv or transformer
            in_feats = in_shape[1]
            out_shape = self.encoder_net.out_shape(in_shape)
            enc_feats = out_shape[1]
        elif len(in_shape) == 4:
            # encoder based in 2d convs
            assert (
                in_feats is not None
            ), "in_feats dimension must be given to calculate pooling dimension"
            in_shape = list(in_shape)
            in_shape[2] = in_feats
            out_shape = self.encoder_net.out_shape(tuple(in_shape))
            enc_feats = out_shape[1] * out_shape[2]

        self.in_feats = in_feats

        logging.info("encoder input shape={}".format(in_shape))
        logging.info("encoder output shape={}".format(out_shape))

        # add projection network to link encoder and pooling layers if proj_feats is not None
        self.proj = None
        self.proj_feats = proj_feats
        if proj_feats is not None:
            logging.info(
                "adding projection layer after encoder with in/out size %d -> %d",
                enc_feats,
                proj_feats,
            )

            self.proj = TDNNBlock(
                enc_feats, proj_feats, kernel_size=1, activation=None, use_norm=use_norm
            )

        # create pooling network
        # infer output dimension of pooling which is input dim for classification head
        if proj_feats is None:
            self.pool_net = self._make_pool_net(pool_net, enc_feats)
            pool_feats = int(enc_feats * self.pool_net.size_multiplier)
        else:
            self.pool_net = self._make_pool_net(pool_net, proj_feats)
            pool_feats = int(proj_feats * self.pool_net.size_multiplier)

        logging.info("infer pooling dimension %d", pool_feats)

        # if head_norm_layer is none we use the global norm_layer
        if head_norm_layer is None and norm_layer is not None:
            if norm_layer == "instance-norm" or norm_layer == "instance-norm-affine":
                head_norm_layer = "batch-norm"
            else:
                head_norm_layer = norm_layer

        # create classification head
        logging.info("making classification head net")
        self.classif_net = ClassifHead(
            pool_feats,
            num_classes,
            embed_dim=embed_dim,
            num_embed_layers=num_embed_layers,
            hid_act=hid_act,
            loss_type=loss_type,
            cos_scale=cos_scale,
            margin=margin,
            margin_warmup_epochs=margin_warmup_epochs,
            intertop_k=intertop_k,
            intertop_margin=intertop_margin,
            num_subcenters=num_subcenters,
            norm_layer=head_norm_layer,
            use_norm=use_norm,
            norm_before=norm_before,
            dropout_rate=dropout_rate,
        )

        self.hid_act = hid_act
        self.norm_layer = norm_layer
        self.head_norm_layer = head_norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.dropout_rate = dropout_rate
        self.embed_layer = embed_layer

    @property
    def pool_feats(self):
        return self.classif_net.in_feats

    @property
    def num_classes(self):
        return self.classif_net.num_classes

    @property
    def embed_dim(self):
        return self.classif_net.embed_dim

    @property
    def num_embed_layers(self):
        return self.classif_net.num_embed_layers

    @property
    def cos_scale(self):
        return self.classif_net.cos_scale

    @property
    def margin(self):
        return self.classif_net.margin

    @property
    def margin_warmup_epochs(self):
        return self.classif_net.margin_warmup_epochs

    @property
    def intertop_k(self):
        return self.classif_net.intertop_k

    @property
    def intertop_margin(self):
        return self.classif_net.intertop_margin

    @property
    def num_subcenters(self):
        return self.classif_net.num_subcenters

    @property
    def loss_type(self):
        return self.classif_net.loss_type

    def _make_pool_net(self, pool_net, enc_feats=None):
        """Makes the pooling block

        Args:
         pool_net: str or dict to pass to the pooling factory create function
         enc_feats: dimension of the features coming from the encoder

        Returns:
         GlobalPool1d object
        """
        print(pool_net, flush=True)
        if isinstance(pool_net, str):
            pool_net = {"pool_type": pool_net}

        if isinstance(pool_net, dict):
            if enc_feats is not None:
                pool_net["in_feats"] = enc_feats

            return PF.create(**pool_net)
        elif isinstance(pool_net, nn.Module):
            return pool_net
        else:
            raise Exception("Invalid pool_net argument")

    def update_loss_margin(self, epoch):
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
          epoch: epoch which is about to start
        """
        self.classif_net.update_margin(epoch)

    def _pre_enc(self, x):
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
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
        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        y = self.classif_net(p, y)
        return y

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
            x, return_enc_layers, return_logits=True
        )
        output = {"h_enc": h_enc}
        if not return_logits and return_classif_layers is None:
            return output

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        h_classif, y_pred = self.classif_net.forward_hid_feats(
            p, y, return_classif_layers, return_logits=return_logits
        )
        if return_logits:
            h_classif, y_pred = h_classif
            output["h_classif"] = h_classif
            output["logits"] = y_pred
            return output

        output["h_classif"] = h_classif
        return output

    def extract_embed(
        self, x, x_lengths=None, chunk_length=0, embed_layer=None, detach_chunks=False
    ):
        if embed_layer is None:
            embed_layer = self.embed_layer

        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        x = eval_nnet_by_chunks(
            x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
        )

        if x.device != self.device:
            x = x.to(self.device)

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        y = self.classif_net.extract_embed(p, embed_layer)
        return y

    def extract_embed_slidwin(
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
            "head_norm_layer": self.head_norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "dropout_rate": self.dropout_rate,
            "embed_layer": self.embed_layer,
            "in_feats": self.in_feats,
            "proj_feats": self.proj_feats,
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
    ):
        logging.info("changing x-vector config")
        self.rebuild_output_layer(
            num_classes=num_classes,
            loss_type=loss_type,
            cos_scale=cos_scale,
            margin=margin,
            margin_warmup_epochs=margin_warmup_epochs,
            intertop_k=intertop_k,
            intertop_margin=intertop_margin,
            num_subcenters=num_subcenters,
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
    ):
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

        if "wo_norm" in kwargs:
            kwargs["use_norm"] = not kwargs["wo_norm"]
            del kwargs["wo_norm"]

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        # get arguments for pooling
        pool_args = PF.filter_args(**kwargs["pool_net"])

        valid_args = (
            "num_classes",
            "embed_dim",
            "num_embed_layers",
            "hid_act",
            "loss_type",
            "cos_scale",
            "margin",
            "margin_warmup_epochs",
            "intertop_k",
            "intertop_margin",
            "num_subcenters",
            "use_norm",
            "norm_before",
            "in_feats",
            "proj_feats",
            "dropout_rate",
            "norm_layer",
            "head_norm_layer",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

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
            "--embed-dim", default=256, type=int, help=("x-vector dimension")
        )

        parser.add_argument(
            "--num-embed-layers",
            default=1,
            type=int,
            help=("number of layers in the classif head"),
        )

        try:
            parser.add_argument("--hid-act", default="relu6", help="hidden activation")
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
            "--wo-norm",
            default=False,
            action="store_true",
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-after",
            default=False,
            action="store_true",
            help="batch normalizaton after activation",
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
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
                help="xvector options",
            )

    @staticmethod
    def filter_finetune_args(**kwargs):
        valid_args = (
            "loss_type",
            "cos_scale",
            "margin",
            "margin_warmup_epochs",
            "intertop_k",
            "intertop_margin",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

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

    add_argparse_args = add_class_args
    add_argparse_finetune_args = add_finetune_args
