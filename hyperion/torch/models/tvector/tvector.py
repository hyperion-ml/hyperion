"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from ..layers import GlobalPool1dFactory as PF
from ..layer_blocks import TDNNBlock
from ...narchs import ClassifHead, ConformerEncoderV1, TorchNALoader
from ..torch_model import TorchModel
from ..utils import eval_nnet_by_chunks


class TXVector(TorchModel):
    """x-Vector base class"""

    def __init__(
        self,
        encoder_net,
        num_classes,
        conformer_net={},
        pool_net="mean+stddev",
        classif_net={},
        in_feats=None,
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

        # create conformer net
        if isinstance(conformer_net, nn.Module):
            self.conformer_net = conformer_net
        else:
            logging.info("making conformer net")
            conformer_net["in_layer_type"] = "linear"
            self.conformer_net = ConformerEncoderV1(
                enc_feats, in_time_dim=1, out_time_dim=1, **conformer_net
            )

        d_model = self.conformer_net.d_model
        self.pool_net = self._make_pool_net(pool_cfg, d_model)
        pool_feats = int(d_model * self.pool_net.size_multiplier)
        logging.info("infer pooling dimension %d", pool_feats)

        # create classification head
        if isinstance(classif_net, nn.Module):
            self.classif_net = classif_net
        else:
            logging.info("making classification head net")
            self.classif_net = ClassifHead(pool_feats, num_classes, **head_cfg)

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
    def s(self):
        return self.classif_net.s

    @property
    def margin(self):
        return self.classif_net.margin

    @property
    def margin_warmup_epochs(self):
        return self.classif_net.margin_warmup_epochs

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

    def _post_enc(self, x):
        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        return x

    def forward(
        self,
        x,
        y=None,
        enc_layers=None,
        classif_layers=None,
        return_output=True,
        use_amp=False,
    ):
        if enc_layers is None and classif_layers is None:
            return self.forward_output(x, y)

        h = self.forward_hid_feats(x, y, enc_layers, classif_layers, return_output)
        output = {}
        if enc_layers is not None:
            if classif_layers is None:
                output["h_enc"] = h
            else:
                output["h_enc"] = h[0]
        else:
            output["h_enc"] = []
        if classif_layers is not None:
            output["h_classif"] = h[1]
        else:
            output["h_classif"] = []
        if return_output:
            output["output"] = h[2]
        return output

    def forward_output(self, x, y=None):
        """Forward function

        Args:
          x: input features tensor with shape=(batch, in_feats, time)
          y: target classes torch.long tensor with shape=(batch,)

        Returns:
          class posteriors tensor with shape=(batch, num_classes)
        """
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.encoder_net(x)
        x = self.conformer_net(x)

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        p = self.pool_net(x)
        y = self.classif_net(p, y)
        return y

    def forward_hid_feats(
        self,
        x,
        y=None,
        enc_layers=None,
        conf_layers=None,
        classif_layers=None,
        return_output=False,
    ):
        """forwards hidden representations in the x-vector network"""

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        h_enc, x = self.encoder_net.forward_hid_feats(x, enc_layers, return_output=True)

        h_conf, x = self.conformer_net.forward_hid_feats(
            x, conf_layers, return_output=True
        )

        if not return_output and classif_layers is None:
            return h_enc

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        p = self.pool_net(x)
        h_classif = self.classif_net.forward_hid_feats(
            p, y, classif_layers, return_output=return_output
        )
        if return_output:
            h_classif, y = h_classif
            return h_enc, h_classif, y

        return h_enc, h_classif

    def extract_embed(self, x, chunk_length=0, embed_layer=None, detach_chunks=False):
        if embed_layer is None:
            embed_layer = self.embed_layer

        x = self._pre_enc(x)
        # if self.encoder_net.in_dim() == 4 and x.dim() == 3:
        #     x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = eval_nnet_by_chunks(
            x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
        )

        if x.device != self.device:
            x = x.to(self.device)

        x = self._post_enc(x)

        # if self.encoder_net.out_dim() == 4:
        #     x = x.view(x.size(0), -1, x.size(-1))

        # if self.proj is not None:
        #     x = self.proj(x)

        p = self.pool_net(x)
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
        x = self._pre_enc(x)
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
        conformer_cfg = self.conformer_net.get_config()
        classif_cfg = self.classif_net.get_config()

        config = {
            "encoder_cfg": enc_cfg,
            "num_classes": self.num_classes,
            "conformer_net": self.conformer_cfg,
            "pool_net": pool_cfg,
            "classif_net": self.classif_cfg,
            "in_feats": self.in_feats,
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

    def rebuild_output_layer(
        self,
        num_classes=None,
        loss_type="arc-softmax",
        s=64,
        margin=0.3,
        margin_warmup_epochs=10,
    ):
        if (self.num_classes is not None and self.num_classes != num_classes) or (
            self.loss_type != loss_type
        ):
            # if we change the number of classes or the loss-type
            # we need to reinitiate the last layer
            self.classif_net.rebuild_output_layer(
                num_classes, loss_type, s, margin, margin_warmup_epochs
            )
            return

        # otherwise we just change the values of s, margin and margin_warmup
        self.classif_net.set_margin(margin)
        self.classif_net.set_margin_warmup_epochs(margin_warmup_epochs)
        self.classif_net.set_s(s)

    def freeze_preembed_layers(self):
        self.encoder_net.freeze()
        if self.proj is not None:
            self.proj.freeze()

        for param in self.pool_net.parameters():
            param.requires_grad = False

        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.freeze_layers(layer_list)

    def train_mode(self, mode="ft-embed-affine"):
        if mode == "ft-full" or mode == "train":
            self.train()
            return

        self.encoder_net.eval()
        self.conformer_net.eval()
        self.pool_net.eval()
        self.classif_net.train()
        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.put_layers_in_eval_mode(layer_list)

    @staticmethod
    def filter_args(**kwargs):

        valid_args = ("num_classes", "in_feats")
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        # get arguments for conformer
        conformer_args = ConformerEncoderV1.filter_args(**kwargs["conformer_net"])
        args["corformer_net"] = conformer_args
        # get arguments for pooling
        pool_args = PF.filter_args(**kwargs["pool_net"])
        args["pool_net"] = pool_args
        # get arguments for classif head
        classif_args = ClassifHead.filter_args(**kwargs["classif_net"])
        args["classif_net"] = classif_args

        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        CoformerEncoderV1.add_class_args(parser, prefix="conformer_net")
        PF.add_class_args(
            parser, prefix="pool_net", skip=["dim", "in_feats", "keepdim"]
        )
        ClassifHead.add_class_args(parser, prefix="classif_net")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
                help="xvector options",
            )

    @staticmethod
    def filter_finetune_args(**kwargs):
        valid_args = ("loss_type", "s", "margin", "margin_warmup_epochs")
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

        parser.add_argument("--s", default=64, type=float, help="scale for arcface")

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
            "--num-subcenters",
            default=2,
            type=float,
            help="number of subcenters in subcenter losses",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
