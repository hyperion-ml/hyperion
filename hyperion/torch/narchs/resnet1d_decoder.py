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
from ..layer_blocks import ResNet1dBasicDecBlock, ResNet1dBNDecBlock, DC1dDecBlock
from ..layer_blocks import SEResNet1dBasicDecBlock, SEResNet1dBNDecBlock
from ..layers import SubPixelConv1d, ICNR1d
from .net_arch import NetArch


class ResNet1dDecoder(NetArch):
    def __init__(
        self,
        in_channels=128,
        in_conv_channels=128,
        in_kernel_size=3,
        in_stride=1,
        resb_type="basic",
        resb_repeats=[1, 1, 1],
        resb_channels=128,
        resb_kernel_sizes=3,
        resb_strides=2,
        resb_dilations=1,
        resb_groups=1,
        head_channels=0,
        hid_act="relu6",
        head_act=None,
        dropout_rate=0,
        se_r=16,
        use_norm=True,
        norm_layer=None,
        norm_before=True,
    ):

        super().__init__()

        self.resb_type = resb_type
        bargs = {}  # block's extra arguments
        if resb_type == "basic":
            self._block = ResNet1dBasicDecBlock
        elif resb_type == "bn":
            self._block = ResNet1dBNDecBlock
        elif resb_type == "sebasic":
            self._block = SEResNet1dBasicDecBlock
            bargs["se_r"] = se_r
        elif resb_type == "sebn":
            self._block = SEResNet1dBNDecBlock
            bargs["se_r"] = se_r

        self.in_channels = in_channels
        self.in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        num_superblocks = len(resb_repeats)
        self.resb_repeats = resb_repeats
        self.resb_channels = self._standarize_resblocks_param(
            resb_channels, num_superblocks, "resb_channels"
        )
        self.resb_kernel_sizes = self._standarize_resblocks_param(
            resb_kernel_sizes, num_superblocks, "resb_kernel_sizes"
        )
        self.resb_strides = self._standarize_resblocks_param(
            resb_strides, num_superblocks, "resb_strides"
        )
        self.resb_dilations = self._standarize_resblocks_param(
            resb_dilations, num_superblocks, "resb_dilations"
        )
        self.resb_groups = resb_groups
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.se_r = se_r

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(min(resb_channels) // 2, 32)
            norm_groups = max(norm_groups, resb_groups)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        # stem block
        self.in_block = DC1dDecBlock(
            in_channels,
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
        self._upsample_factor = self.in_block.stride

        cur_in_channels = in_conv_channels

        # middle blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_superblocks):
            repeats_i = self.resb_repeats[i]
            channels_i = self.resb_channels[i]
            stride_i = self.resb_strides[i]
            kernel_size_i = self.resb_kernel_sizes[i]
            dilation_i = self.resb_dilations[i]
            block_i = self._block(
                cur_in_channels,
                channels_i,
                kernel_size_i,
                stride=stride_i,
                dilation=1,
                groups=self.resb_groups,
                activation=hid_act,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
                norm_layer=self._norm_layer,
                norm_before=norm_before,
                **bargs
            )

            self.blocks.append(block_i)
            self._context += block_i.context * self._upsample_factor
            self._upsample_factor *= block_i.upsample_factor

            for j in range(repeats_i - 1):
                block_i = self._block(
                    channels_i,
                    channels_i,
                    kernel_size_i,
                    stride=1,
                    dilation=dilation_i,
                    groups=self.resb_groups,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    use_norm=use_norm,
                    norm_layer=self._norm_layer,
                    norm_before=norm_before,
                    **bargs
                )

                self.blocks.append(block_i)
                self._context += block_i.context * self._upsample_factor

            cur_in_channels = channels_i

        # head feature block
        if self.head_channels > 0:
            self.head_block = DC1dDecBlock(
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
        if isinstance(hid_act, str):
            act_name = hid_act
        if isinstance(hid_act, dict):
            act_name = hid_act["name"]
        if act_name in ["relu6", "swish"]:
            act_name = "relu"

        init_f1 = lambda x: nn.init.kaiming_normal_(
            x, mode="fan_out", nonlinearity=act_name
        )
        init_f2 = lambda x: nn.init.kaiming_normal_(
            x, mode="fan_out", nonlinearity="relu"
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                try:
                    init_f1(m.weight)
                except:
                    init_f2(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # re-init subpixelconvs
        for m in self.modules():
            if isinstance(m, SubPixelConv1d):
                try:
                    ICNR1d(m.conv.weight, stride=m.stride, initializer=init_f1)
                except:
                    ICNR1d(m.conv.weight, stride=m.stride, initializer=init_f2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         if isinstance(hid_act, str):
        #             act_name = hid_act
        #         if isinstance(hid_act, dict):
        #             act_name = hid_act['name']
        #         if act_name == 'swish':
        #             act_name = 'relu'
        #         try:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_name)
        #         except:
        #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    @staticmethod
    def _standarize_resblocks_param(p, num_blocks, p_name):
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
        out_size = in_size * self.in_stride

        for stride in self.conv_strides:
            out_size *= stride

        return out_size

    def in_context(self):
        in_context = int(math.ceil(self._context / self._upsample_factor))
        return (in_context, in_context)

    def in_shape(self):
        return (None, self.in_channels, None)

    def out_shape(self, in_shape=None):

        out_channels = (
            self.head_channels if self.head_channels > 0 else self.resb_channels[-1]
        )
        if in_shape is None:
            return (None, out_channels, None)

        assert len(in_shape) == 3
        if in_shape[2] is None:
            T = None
        else:
            T = self._compute_out_size(in_shape[2])

        return (in_shape[0], out_channels, T)

    def _match_shape(self, x, target_shape):
        t = x.size(-1)
        target_t = target_shape[-1]
        surplus = t - target_t
        assert surplus >= 0
        if surplus > 0:
            x = torch.narrow(x, -1, surplus // 2, target_t).contiguous()

        return x

    def forward(self, x, target_shape=None):

        x = self.in_block(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if self.head_channels > 0:
            x = self.head_block(x)

        if target_shape is not None:
            x = self._match_shape(x, target_shape)

        return x

    def get_config(self):

        head_act = self.head_act
        hid_act = self.hid_act

        config = {
            "in_channels": self.in_channels,
            "in_conv_channels": self.in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "resb_type": self.resb_type,
            "resb_repeats": self.resb_repeats,
            "resb_channels": self.resb_channels,
            "resb_kernel_sizes": self.resb_kernel_sizes,
            "resb_strides": self.resb_strides,
            "resb_dilations": self.resb_dilations,
            "resb_groups": self.resb_groups,
            "head_channels": self.head_channels,
            "dropout_rate": self.dropout_rate,
            "hid_act": hid_act,
            "head_act": head_act,
            "se_r": self.se_r,
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
            "in_channels",
            "in_conv_channels",
            "in_kernel_size",
            "in_stride",
            "resb_type",
            "resb_repeats",
            "resb_channels",
            "resb_kernel_sizes",
            "resb_strides",
            "resb_dilations",
            "resb_groups",
            "head_channels",
            "se_r",
            "hid_act",
            "head_act",
            "dropout_rate",
            "use_norm",
            "norm-layer",
            "norm_before",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--in-channels", type=int, default=80, help=("input channels of decoder")
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
            "--resb-type",
            default="basic",
            choices=["basic", "bn"],
            help=("residual blocks type"),
        )

        parser.add_argument(
            "--resb-repeats",
            default=[1, 1, 1],
            type=int,
            nargs="+",
            help=("resb-blocks repeats in each encoder stage"),
        )

        parser.add_argument(
            "--resb-channels",
            default=[128, 64, 32],
            type=int,
            nargs="+",
            help=("resb-blocks channels for each stage"),
        )

        parser.add_argument(
            "--resb-kernel-sizes",
            default=[3],
            nargs="+",
            type=int,
            help=("resb-blocks kernels for each encoder stage"),
        )

        parser.add_argument(
            "--resb-strides",
            default=[2],
            nargs="+",
            type=int,
            help=("resb-blocks strides for each encoder stage"),
        )

        parser.add_argument(
            "--resb-dilations",
            default=[1],
            nargs="+",
            type=int,
            help=("resb-blocks dilations for each encoder stage"),
        )

        parser.add_argument(
            "--resb-groups",
            default=1,
            type=int,
            help=("resb-blocks groups in convolutions"),
        )

        parser.add_argument(
            "--head-channels",
            default=0,
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

        parser.add_argument(
            "--se-r",
            default=16,
            type=int,
            help=("squeeze-excitation compression ratio"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='ResNet1d decoder options')

    add_argparse_args = add_class_args
