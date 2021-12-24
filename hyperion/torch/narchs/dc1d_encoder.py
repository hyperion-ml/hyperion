"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser
import math

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from ..layer_blocks.dc1d_blocks import DC1dEncBlock
from .net_arch import NetArch


class DC1dEncoder(NetArch):
    def __init__(
        self,
        in_feats,
        in_conv_channels=128,
        in_kernel_size=3,
        in_stride=1,
        conv_repeats=[1, 1, 1],
        conv_channels=[128, 64, 32],
        conv_kernel_sizes=3,
        conv_strides=2,
        conv_dilations=1,
        head_channels=0,
        hid_act="relu6",
        head_act=None,
        dropout_rate=0,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()
        self.in_feats = in_feats
        self.in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        num_superblocks = len(conv_repeats)
        self.conv_repeats = conv_repeats
        self.conv_channels = self._standarize_convblocks_param(
            conv_channels, num_superblocks, "conv_channels"
        )
        self.conv_kernel_sizes = self._standarize_convblocks_param(
            conv_kernel_sizes, num_superblocks, "conv_kernel_sizes"
        )
        self.conv_strides = self._standarize_convblocks_param(
            conv_strides, num_superblocks, "conv_strides"
        )
        self.conv_dilations = self._standarize_convblocks_param(
            conv_dilations, num_superblocks, "conv_dilations"
        )
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(np.min(self.conv_channels) // 2, 32)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        # stem block
        self.in_block = DC1dEncBlock(
            in_feats,
            in_conv_channels,
            in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
        )
        self._context = self.in_block.context
        self._downsample_factor = self.in_block.stride

        cur_in_channels = in_conv_channels

        # middle blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_superblocks):
            repeats_i = self.conv_repeats[i]
            channels_i = self.conv_channels[i]
            stride_i = self.conv_strides[i]
            kernel_size_i = self.conv_kernel_sizes[i]
            dilation_i = self.conv_dilations[i]
            block_i = DC1dEncBlock(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=1,
                activation=hid_act,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
            )

            self.blocks.append(block_i)
            self._context += block_i.context * self._downsample_factor
            self._downsample_factor *= block_i.stride

            for j in range(repeats_i - 1):
                block_i = DC1dEncBlock(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                )

                self.blocks.append(block_i)
                self._context += block_i.context * self._downsample_factor

            cur_in_channels = channels_i

        # head feature block
        if self.head_channels > 0:
            self.head_block = DC1dEncBlock(
                cur_in_channels,
                head_channels,
                kernel_size=1,
                stride=1,
                activation=head_act,
                use_norm=False,
                norm_before=norm_before,
            )

        self._init_weights(hid_act)

    def _init_weights(self, hid_act):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                try:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=act_name
                    )
                except:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _standarize_convblocks_param(p, num_blocks, p_name):
        if isinstance(p, int):
            p = [p] * num_blocks
        elif isinstance(p, list):
            if len(p) == 1:
                p = p * num_blocks

            assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
                p_name,
                len(p),
                num_blocks,
            )
        else:
            raise TypeError("wrong type for param {}={}".format(p_name, p))

        return p

    def _compute_out_size(self, in_size):
        out_size = int((in_size - 1) // self.in_stride + 1)

        for stride in self.conv_strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, self.in_feats, None)

    def out_shape(self, in_shape=None):

        out_channels = (
            self.head_channels if self.head_channels > 0 else self.conv_channels[-1]
        )
        if in_shape is None:
            return (None, out_channels, None)

        assert len(in_shape) == 3
        if in_shape[2] is None:
            T = None
        else:
            T = self._compute_out_size(in_shape[2])

        return (in_shape[0], out_chanels, T)

    def forward(self, x):

        x = self.in_block(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if self.head_channels > 0:
            x = self.head_block(x)

        return x

    def get_config(self):

        head_act = self.head_act
        hid_act = self.hid_act

        config = {
            "in_feats": self.in_feats,
            "in_conv_channels": self.in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "conv_repeats": self.conv_repeats,
            "conv_channels": self.conv_channels,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_strides": self.conv_strides,
            "conv_dilations": self.conv_dilations,
            "head_channels": self.head_channels,
            "dropout_rate": self.dropout_rate,
            "hid_act": hid_act,
            "head_act": head_act,
            "use_norm": self.use_norm,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs):

        if "wo_norm" in kwargs:
            kwargs["use_norm"] = not kwargs["wo_norm"]
            del kwargs["wo_norm"]

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        valid_args = (
            "in_feats",
            "in_conv_channels",
            "in_kernel_size",
            "in_stride",
            "conv_repeats",
            "conv_channels",
            "conv_kernel_sizes",
            "conv_strides",
            "conv_dilations",
            "head_channels",
            "hid_act",
            "had_act",
            "dropout_rate",
            "use_norm",
            "norm_layer",
            "norm_before",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser, prefix=None, head_channels=False, in_feats=False):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if in_feats:
            parser.add_argument(
                "--in-feats", type=int, required=True, help=("input feature dimension")
            )

        parser.add_argument(
            "--in-conv-channels",
            default=128,
            type=int,
            help=("number of output channels in input convolution"),
        )

        parser.add_argument(
            "--in-kernel-size",
            default=3,
            type=int,
            help=("kernel size of input convolution"),
        )

        parser.add_argument(
            "--in-stride", default=1, type=int, help=("stride of input convolution")
        )

        parser.add_argument(
            "--conv-repeats",
            default=[1, 1, 1],
            type=int,
            nargs="+",
            help=("conv-blocks repeats in each encoder stage"),
        )

        parser.add_argument(
            "--conv-channels",
            default=[128, 64, 32],
            type=int,
            nargs="+",
            help=("conv-blocks channels for each stage"),
        )

        parser.add_argument(
            "--conv-kernel-sizes",
            default=[3],
            nargs="+",
            type=int,
            help=("conv-blocks kernels for each encoder stage"),
        )

        parser.add_argument(
            "--conv-strides",
            default=[2],
            nargs="+",
            type=int,
            help=("conv-blocks strides for each encoder stage"),
        )

        parser.add_argument(
            "--conv-dilations",
            default=[1],
            nargs="+",
            type=int,
            help=("conv-blocks dilations for each encoder stage"),
        )

        if head_channels:
            parser.add_argument(
                "--head-channels",
                default=16,
                type=int,
                help=("channels in the last conv block of encoder"),
            )

        try:
            parser.add_argument("--hid-act", default="relu6", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--head-act", default=None, help="activation in encoder head"
        )

        try:
            parser.add_argument(
                "--dropout-rate", default=0, type=float, help="dropout probability"
            )
        except:
            pass

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
                help="type of normalization layer",
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

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='DC1d encoder options')

    add_argparse_args = add_class_args
