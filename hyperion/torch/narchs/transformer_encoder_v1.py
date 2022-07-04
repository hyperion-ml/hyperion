"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import PosEncoder, RelPosEncoder
from ..layer_blocks import TransformerEncoderBlockV1 as EBlock
from ..layer_blocks import TransformerConv2dSubsampler as Conv2dSubsampler
from .net_arch import NetArch


class TransformerEncoderV1(NetArch):
    """Transformer encoder module.

    Attributes:
      in_feats: input features dimension
      d_model: encoder blocks feature dimension
      num_heads: number of heads
      num_blocks: number of self attn blocks
      att_type: string in ['scaled-dot-prod-att-v1', 'local-scaled-dot-prod-att-v1']
      att_context: maximum context range for local attention
      ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
      d_ff: dimension of middle layer in feed_forward block
      ff_kernel_size: kernel size for convolutional versions of ff block

      ff_dropout_rate: dropout rate for ff block
      pos_dropout_rate: dropout rate for positional encoder
      att_dropout_rate: dropout rate for attention block
      in_layer_type: input layer block type in ['linear','conv2d-sub', 'embed', None]
      rel_pos_enc: if True, use relative postional encodings, absolute encodings otherwise.
      causal_pos_enc: if True, use causal positional encodings (when rel_pos_enc=True), it assumes
                      that query q_i only attents to key k_j when j<=i
      hid_act:  hidden activations in ff and input blocks
      norm_before: if True, use layer norm before layers, otherwise after
      concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)
      padding_idx: padding idx for embed layer
      in_time_dim: time dimension in the input Tensor
      out_time_dim: dimension that we want to be time in the output tensor

    """

    def __init__(
        self,
        in_feats,
        d_model=256,
        num_heads=4,
        num_blocks=6,
        att_type="scaled-dot-prod-v1",
        att_context=25,
        ff_type="linear",
        d_ff=2048,
        ff_kernel_size=1,
        ff_dropout_rate=0.1,
        pos_dropout_rate=0.1,
        att_dropout_rate=0.0,
        in_layer_type="conv2d-sub",
        rel_pos_enc=False,
        causal_pos_enc=False,
        hid_act="relu6",
        norm_before=True,
        concat_after=False,
        padding_idx=-1,
        in_time_dim=-1,
        out_time_dim=1,
    ):

        super().__init__()
        self.in_feats = in_feats
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.att_type = att_type
        self.att_context = att_context

        self.ff_type = ff_type
        self.d_ff = d_ff
        self.ff_kernel_size = ff_kernel_size
        self.ff_dropout_rate = ff_dropout_rate
        self.rel_pos_enc = rel_pos_enc
        self.causal_pos_enc = causal_pos_enc
        self.att_dropout_rate = att_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.in_layer_type = in_layer_type
        self.norm_before = norm_before
        self.concat_after = concat_after
        self.padding_idx = padding_idx
        self.in_time_dim = in_time_dim
        self.out_time_dim = out_time_dim
        self.hid_act = hid_act

        self._make_in_layer()

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                EBlock(
                    d_model,
                    att_type,
                    num_heads,
                    ff_type,
                    d_ff,
                    ff_kernel_size,
                    ff_act=hid_act,
                    ff_dropout_rate=ff_dropout_rate,
                    att_context=att_context,
                    att_dropout_rate=att_dropout_rate,
                    rel_pos_enc=rel_pos_enc,
                    causal_pos_enc=causal_pos_enc,
                    norm_before=norm_before,
                    concat_after=concat_after,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        if self.norm_before:
            self.norm = nn.LayerNorm(d_model)

    def _make_in_layer(self):

        in_feats = self.in_feats
        d_model = self.d_model
        dropout_rate = self.ff_dropout_rate
        if self.rel_pos_enc:
            pos_enc = RelPosEncoder(d_model, self.pos_dropout_rate)
        else:
            pos_enc = PosEncoder(d_model, self.pos_dropout_rate)

        hid_act = AF.create(self.hid_act)

        if self.in_layer_type == "linear":
            self.in_layer = nn.Sequential(
                nn.Linear(in_feats, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout_rate),
                hid_act,
                pos_enc,
            )
        elif self.in_layer_type == "conv2d-sub":
            self.in_layer = Conv2dSubsampler(
                in_feats, d_model, hid_act, pos_enc, time_dim=self.in_time_dim
            )
        elif self.in_layer_type == "embed":
            self.in_layer = nn.Sequential(
                nn.Embedding(in_feats, d_model, padding_idx=self.padding_idx), pos_enc
            )
        elif isinstance(self.in_layer_type, nn.Module):
            self.in_layer = nn.Sequential(self.in_layer_type, pos_enc)
        elif self.in_layer_type is None:
            self.in_layer = pos_enc
        else:
            raise ValueError("unknown in_layer_type: " + self.in_layer_type)

    def forward(self, x, mask=None, target_shape=None, use_amp=False):
        if use_amp:
            with torch.cuda.amp.autocast():
                return self._forward(x, mask, target_shape)

        return self._forward(x, mask, target_shape)

    def _forward(self, x, mask=None, target_shape=None):
        """Forward pass function

        Args:
          x: input tensor with size=(batch, time, num_feats)
          mask: mask to indicate valid time steps for x (batch, time)

        Returns:
           Tensor with output features
           Tensor with mask
        """
        if isinstance(self.in_layer, Conv2dSubsampler):
            x, mask = self.in_layer(x, mask)
        else:
            if self.in_time_dim != 1:
                x = x.transpose(1, self.in_time_dim).contiguous()
            x = self.in_layer(x)

        if isinstance(x, tuple):
            x, pos_emb = x
            b_args = {"pos_emb": pos_emb}
        else:
            b_args = {}

        for i in range(len(self.blocks)):
            x, mask = self.blocks[i](x, mask=mask, **b_args)

        if self.norm_before:
            x = self.norm(x)

        if self.out_time_dim != 1:
            x = x.transpose(1, self.out_time_dim)

        if mask is None:
            return x

        return x, mask

    def get_config(self):
        """Gets network config
        Returns:
           dictionary with config params
        """
        config = {
            "in_feats": self.in_feats,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "att_type": self.att_type,
            "att_context": self.att_context,
            "ff_type": self.ff_type,
            "d_ff": self.d_ff,
            "ff_kernel_size": self.ff_kernel_size,
            "ff_dropout_rate": self.ff_dropout_rate,
            "att_dropout_rate": self.att_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "in_layer_type": self.in_layer_type,
            "rel_pos_enc": self.rel_pos_enc,
            "causal_pos_enc": self.causal_pos_enc,
            "hid_act": self.hid_act,
            "norm_before": self.norm_before,
            "concat_after": self.concat_after,
            "padding_idx": self.padding_idx,
            "in_time_dim": self.in_time_dim,
            "out_time_dim": self.out_time_dim,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_dropouts(self, pos_dropout_rate, att_dropout_rate, ff_dropout_rate):

        assert pos_dropout_rate == 0 or self.pos_dropout_rate > 0
        assert att_dropout_rate == 0 or self.att_dropout_rate > 0
        assert ff_dropout_rate == 0 or self.ff_dropout_rate > 0

        for module in self.modules():
            if isinstance(module, PosEncoder):
                for layer in module.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = pos_dropout_rate

            elif isinstance(module, EBlock):
                for layer in module.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = ff_dropout_rate

                for layer in module.self_attn.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = att_dropout_rate

        self.pos_dropout_rate = pos_dropout_rate
        self.att_dropout_rate = att_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate

    def in_context(self):
        return (self.att_context, self.att_context)

    def in_shape(self):
        """Input shape for network

        Returns:
           Tuple describing input shape
        """
        if self.in_time_dim == 1:
            return (None, None, self.in_feats)
        else:
            return (None, self.in_feats, None)

    def out_shape(self, in_shape=None):
        """Infers the network output shape given the input shape

        Args:
          in_shape: input shape tuple

        Returns:
          Tuple with the output shape
        """
        if in_shape is None:
            out_t = None
            batch_size = None
        else:
            assert len(in_shape) == 3
            batch_size = in_shape[0]
            in_t = in_shape[self.in_time_dim]
            if in_t is None:
                out_t = None
            else:
                if isinstance(self.in_layer, Conv2dSubsampler):
                    # out_t = in_t//4
                    out_t = ((in_t - 1) // 2 - 1) // 2
                else:
                    out_t = in_t

        if self.out_time_dim == 1:
            return (batch_size, out_t, self.d_model)
        else:
            return (batch_size, self.d_model, out_t)

    @staticmethod
    def filter_args(**kwargs):
        """Filters arguments correspondin to TransformerXVector
            from args dictionary

        Args:
          kwargs: args dictionary

        Returns:
          args dictionary
        """

        valid_args = (
            "num_blocks",
            "in_feats",
            "d_model",
            "num_heads",
            "att_type",
            "att_context",
            "ff_type",
            "d_ff",
            "ff_kernel_size",
            "ff_dropout_rate",
            "pos_dropout_rate",
            "att_dropout_rate",
            "in_layer_type",
            "hid_act",
            "rel_pos_enc",
            "causal_pos_enc",
            "concat_after",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, in_feats=False):
        """Adds Transformer config parameters to argparser

        Args:
           parser: argparse object
           prefix: prefix string to add to the argument names
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if in_feats:
            parser.add_argument(
                "--in-feats", type=int, default=80, help=("input feature dimension")
            )

        parser.add_argument(
            "--num-blocks", default=6, type=int, help=("number of tranformer blocks")
        )

        parser.add_argument(
            "--d-model", default=512, type=int, help=("encoder layer sizes")
        )

        parser.add_argument(
            "--num-heads",
            default=4,
            type=int,
            help=("number of heads in self-attention layers"),
        )

        parser.add_argument(
            "--att-type",
            default="scaled-dot-prod-v1",
            choices=["scaled-dot-prod-v1", "local-scaled-dot-prod-v1"],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--att-context",
            default=25,
            type=int,
            help=("context size when using local attention"),
        )

        parser.add_argument(
            "--ff-type",
            default="linear",
            choices=["linear", "conv1dx2", "conv1dlinear"],
            help=("type of feed forward layers in transformer block"),
        )

        parser.add_argument(
            "--d-ff",
            default=2048,
            type=int,
            help=("size middle layer in feed forward block"),
        )

        parser.add_argument(
            "--ff-kernel-size",
            default=3,
            type=int,
            help=("kernel size in convolutional feed forward block"),
        )

        try:
            parser.add_argument("--hid-act", default="relu6", help="hidden activation")
        except:
            pass

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
            "--ff-dropout-rate",
            default=0.1,
            type=float,
            help="feed-forward layer dropout",
        )

        parser.add_argument(
            "--in-layer-type",
            default="linear",
            choices=["linear", "conv2d-sub"],
            help=("type of input layer"),
        )

        parser.add_argument(
            "--rel-pos-enc",
            default=False,
            action="store_true",
            help="use relative positional encoder",
        )

        parser.add_argument(
            "--causal-pos-enc",
            default=False,
            action="store_true",
            help="relative positional encodings are zero when attending to the future",
        )

        parser.add_argument(
            "--concat-after",
            default=False,
            action="store_true",
            help="concatenate attention input and output instead of adding",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='transformer encoder options')

    add_argparse_args = add_class_args
