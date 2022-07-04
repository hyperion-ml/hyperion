f"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from .xvector import XVector
from ...narchs import TransformerEncoderV1 as TE


class TransformerXVectorV1(XVector):
    """x-Vector with Transformer encoder.

    Attributes:
      in_feats: input features dimension
      num_classes: number of training classes
      enc_d_model: encoder blocks feature dimension
      num_enc_heads: number of heads
      num_enc_blocks: number of self attn blocks
      enc_att_type: string in ['scaled-dot-prod-att-v1', 'local-scaled-dot-prod-att-v1']
      enc_att_context: maximum context range for local attention
      enc_ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
      enc_d_ff: dimension of middle layer in feed_forward block
      enc_ff_kernel_size: kernel size for convolutional versions of ff block
      in_layer_type: input layer block type in ['linear','conv2d-sub', 'embed', None]
      enc_concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)
      pool_net: pooling block configuration string or dictionary of params
      embed_dim: x-vector  dimension
      num_embed_layers: number of hidden layers in classification head
      hid_act: hidden activation configuration string or dictionary
      loss_type: sofmax losss type string in ['softmax', 'arc-softmax', 'cos-softmax']
      cos_scale: s parameter in arc/cos-softmax losses
      margin: margin in arc/cos-sofmtax losses
      margin_warmup_epochs: number of epochs until we reach the maximum value for margin
      dropout_rate: dropout rate for ff block and classification head
      pos_dropout_rate: dropout rate for positional encoder
      att_dropout_rate: dropout rate for attention block


      use_norm: if True use batch/layer norm
      norm_before: if True, use layer norm before layers, otherwise after
      in_norm: add batchnorm at the input
      embed_layer: which layer to use to extract x-vectors
      proj_feats: add linear projection layer after the encoder to project feature dimension to proj_feats
    """

    def __init__(
        self,
        in_feats,
        num_classes,
        enc_d_model=512,
        num_enc_heads=4,
        num_enc_blocks=6,
        enc_att_type="scaled-dot-prod-v1",
        enc_att_context=25,
        enc_ff_type="linear",
        enc_d_ff=2048,
        enc_ff_kernel_size=1,
        in_layer_type="conv2d-sub",
        enc_concat_after=False,
        pool_net="mean+stddev",
        embed_dim=256,
        num_embed_layers=1,
        hid_act={"name": "relu6", "inplace": True},
        loss_type="arc-softmax",
        cos_scale=64,
        margin=0.3,
        margin_warmup_epochs=0,
        intertop_k=5,
        intertop_margin=0.0,
        num_subcenters=2,
        dropout_rate=0.1,
        pos_dropout_rate=0.1,
        att_dropout_rate=0.0,
        norm_layer=None,
        head_norm_layer=None,
        use_norm=True,
        norm_before=False,
        in_norm=False,
        embed_layer=0,
        proj_feats=None,
    ):

        logging.info("making transformer-v1 encoder network")
        encoder_net = TE(
            in_feats,
            enc_d_model,
            num_enc_heads,
            num_enc_blocks,
            att_type=enc_att_type,
            att_context=enc_att_context,
            ff_type=enc_ff_type,
            d_ff=enc_d_ff,
            ff_kernel_size=enc_ff_kernel_size,
            ff_dropout_rate=dropout_rate,
            pos_dropout_rate=pos_dropout_rate,
            att_dropout_rate=att_dropout_rate,
            in_layer_type=in_layer_type,
            norm_before=norm_before,
            concat_after=enc_concat_after,
            in_time_dim=-1,
            out_time_dim=-1,
        )

        super().__init__(
            encoder_net,
            num_classes,
            pool_net=pool_net,
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
            norm_layer=norm_layer,
            head_norm_layer=head_norm_layer,
            use_norm=use_norm,
            norm_before=norm_before,
            dropout_rate=dropout_rate,
            embed_layer=embed_layer,
            in_feats=None,
            proj_feats=proj_feats,
        )

    @property
    def enc_d_model(self):
        return self.encoder_net.d_model

    @property
    def num_enc_heads(self):
        return self.encoder_net.num_heads

    @property
    def num_enc_blocks(self):
        return self.encoder_net.num_blocks

    @property
    def enc_att_type(self):
        return self.encoder_net.att_type

    @property
    def enc_att_context(self):
        return self.encoder_net.att_context

    @property
    def enc_ff_type(self):
        return self.encoder_net.ff_type

    @property
    def enc_d_ff(self):
        return self.encoder_net.d_ff

    @property
    def enc_ff_kernel_size(self):
        return self.encoder_net.ff_kernel_size

    @property
    def pos_dropout_rate(self):
        return self.encoder_net.pos_dropout_rate

    @property
    def att_dropout_rate(self):
        return self.encoder_net.att_dropout_rate

    @property
    def in_layer_type(self):
        return self.encoder_net.in_layer_type

    @property
    def enc_concat_after(self):
        return self.encoder_net.concat_after

    @property
    def enc_ff_type(self):
        return self.encoder_net.ff_type

    # @property
    # def in_norm(self):
    #     return self.encoder_net.in_norm

    def get_config(self):
        """Gets network config
        Returns:
           dictionary with config params
        """
        base_config = super(TransformerXVectorV1, self).get_config()
        del base_config["encoder_cfg"]

        pool_cfg = self.pool_net.get_config()

        config = {
            "num_enc_blocks": self.num_enc_blocks,
            "in_feats": self.in_feats,
            "enc_d_model": self.enc_d_model,
            "num_enc_heads": self.num_enc_heads,
            "enc_att_type": self.enc_att_type,
            "enc_att_context": self.enc_att_context,
            "enc_ff_type": self.enc_ff_type,
            "enc_d_ff": self.enc_d_ff,
            "enc_ff_kernel_size": self.enc_ff_kernel_size,
            "pos_dropout_rate": self.pos_dropout_rate,
            "att_dropout_rate": self.att_dropout_rate,
            "in_layer_type": self.in_layer_type,
            "enc_concat_after": self.enc_concat_after,
        }
        #'in_norm': self.in_norm }

        config.update(base_config)
        return config

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        """Loads model from file"""
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        # fix to load old model
        if "d_enc_ff" in cfg:
            cfg["enc_d_ff"] = cfg["d_enc_ff"]
            del cfg["d_enc_ff"]
        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs):
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          prefix: prefix string
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        base_args = XVector.filter_args(**kwargs)

        valid_args = (
            "num_enc_blocks",
            "in_feats",
            "enc_d_model",
            "num_enc_heads",
            "enc_att_type",
            "enc_att_context",
            "enc_ff_type",
            "enc_d_ff",
            "enc_ff_kernel_size",
            "pos_dropout_rate",
            "att_dropout_rate",
            "in_layer_type",
            "enc_concat_after",
        )

        child_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds TransformerXVector config parameters to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser)
        parser.add_argument(
            "--num-enc-blocks",
            default=6,
            type=int,
            help=("number of tranformer blocks"),
        )

        parser.add_argument(
            "--enc-d-model", default=512, type=int, help=("encoder layer sizes")
        )

        parser.add_argument(
            "--num-enc-heads",
            default=4,
            type=int,
            help=("number of heads in self-attention layers"),
        )

        parser.add_argument(
            "--enc-att-type",
            default="scaled-dot-prod-v1",
            choices=["scaled-dot-prod-v1", "local-scaled-dot-prod-v1"],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--enc-att-context",
            default=25,
            type=int,
            help=("context size when using local attention"),
        )

        parser.add_argument(
            "--enc-ff-type",
            default="linear",
            choices=["linear", "conv1dx2", "conv1dlinear"],
            help=("type of feed forward layers in transformer block"),
        )

        parser.add_argument(
            "--enc-d-ff",
            default=2048,
            type=int,
            help=("size middle layer in feed forward block"),
        )

        parser.add_argument(
            "--enc-ff-kernel-size",
            default=3,
            type=int,
            help=("kernel size in convolutional feed forward block"),
        )

        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )

        parser.add_argument(
            "--in-layer-type",
            default="linear",
            choices=["linear", "conv2d-sub"],
            help=("type of input layer"),
        )

        parser.add_argument(
            "--enc-concat-after",
            default=False,
            action="store_true",
            help="concatenate attention input and output instead of adding",
        )

        # parser.add_argument('--in-norm', default=False, action='store_true',
        #                     help='batch normalization at the input')
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='xvector options')

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs):
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          kwargs: args dictionary

        Returns:
          args dictionary
        """
        base_args = XVector.filter_finetune_args(**kwargs)

        valid_args = (
            "pos_dropout_rate",
            "att_dropout_rate",
        )

        child_args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        """Adds TransformerXVector config parameters for finetuning to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
