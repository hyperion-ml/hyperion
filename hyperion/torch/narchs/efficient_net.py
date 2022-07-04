"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from jsonargparse import ArgumentParser, ActionParser, ActionYesNo

import torch
import torch.nn as nn
from torch.nn import Linear, Dropout

from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..layer_blocks import MBConvBlock, MBConvInOutBlock
from .net_arch import NetArch


class EfficientNet(NetArch):
    """EfficientNet class based on
       Tan, M., Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
       https://arxiv.org/abs/1905.11946

    Attributes:
      effnet_type: EfficientNet types as defined in the paper in
                   ['efficientnet-b0', 'efficientnet-b1', ..., ], it applies the
                   default width and depth scales defined in the paper for those networks.
      in_channels: number of input channels
      in_conv_channels: output channels of the input convolution
      in_kernel_size: kernel size of the input convolution
      in_stride: stride of the input convolution
      mbconv_repeats: number of MobileNet blocks in each super-block,
                      a super-block is group of MobileNet blocks with common parameters
      mbconv_channels: number of channels of the bottleneck in the MobileNet blocks in each super-block
      mbconv_kernel_sizes: kernel sizes of the MobileNet blocks in each super-block
      mbconv_strides: strides applied at the beginning of each super-block
      mbconv_expansions: expansion in the number channels in the inner layer of the MobileNet block
                         w.r.t. the outer (bottleneck) layers.
      head_channels: number of layers in head convolution.
      width_scale: width scale to apply to the network channels w.r.t efficientnet-b0,
                   it overrides effnet_type argument.
      depth_scale: width scale to apply to the network number of layers w.r.t efficientnet-b0,
                   it overrides effnet_type argument.
      fix_stem_head: if True, the number of channels in the head is not affected by width/depth scale,
                     if False, it is affected.
      out_units: output number of classes, if equal to 0, there is no output layer and the head layer
                 becomes the output.
      hid_act: hidden activation
      out_act: output activation with out_units > 0.
      drop_connect_rate: drop connect rate for stochastic depth
      dropout_rate: dropout rate after pooling when out_units > 0
      norm_layer: norm_layer object or str indicating type layer-norm object, if None it uses BatchNorm2d
      se_r: squeeze-excitation dimension compression
      time_se: if True squeeze-excitation embedding is obtaining by averagin only in the time dimension,
               instead of time-freq dimension or HxW dimensions
      in_feats: input feature size (number of components in dimension of 2 of input tensor), this is only
                required when time_se=True to calculcate the size of the squeeze excitation matrices.
    """

    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        "efficientnet-b8": (2.2, 3.6, 672, 0.5),
        "efficientnet-l2": (4.3, 5.3, 800, 0.5),
    }

    def __init__(
        self,
        effnet_type="efficientnet-b0",
        in_channels=1,
        in_conv_channels=32,
        in_kernel_size=3,
        in_stride=2,
        mbconv_repeats=[1, 2, 2, 3, 3, 4, 1],
        mbconv_channels=[16, 24, 40, 80, 112, 192, 320],
        mbconv_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
        mbconv_strides=[1, 2, 2, 2, 1, 2, 1],
        mbconv_expansions=[1, 6, 6, 6, 6, 6, 6],
        head_channels=1280,
        width_scale=None,
        depth_scale=None,
        fix_stem_head=False,
        out_units=0,
        hid_act="swish",
        out_act=None,
        drop_connect_rate=0.2,
        dropout_rate=0,
        norm_layer=None,
        se_r=4,
        time_se=False,
        in_feats=None,
    ):

        super().__init__()

        assert len(mbconv_repeats) == len(mbconv_channels)
        assert len(mbconv_repeats) == len(mbconv_kernel_sizes)
        assert len(mbconv_repeats) == len(mbconv_strides)
        assert len(mbconv_repeats) == len(mbconv_expansions)

        self.effnet_type = effnet_type

        self.in_channels = in_channels
        self.b0_in_conv_channels = in_conv_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride

        self.b0_mbconv_repeats = mbconv_repeats
        self.b0_mbconv_channels = mbconv_channels
        self.mbconv_kernel_sizes = mbconv_kernel_sizes
        self.mbconv_strides = mbconv_strides
        self.mbconv_expansions = mbconv_expansions

        self.b0_head_channels = head_channels
        self.out_units = out_units
        self.hid_act = hid_act

        self.drop_connect_rate = drop_connect_rate
        self.dropout_rate = dropout_rate

        self.se_r = se_r
        self.time_se = time_se
        self.in_feats = in_feats

        # set depth/width scales from net name
        self.cfg_width_scale = width_scale
        self.cfg_depth_scale = depth_scale
        if width_scale is None or depth_scale is None:
            width_scale, depth_scale = self.efficientnet_params(effnet_type)[:2]
        self.width_scale = width_scale
        self.depth_scale = depth_scale
        self.fix_stem_head = fix_stem_head

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(int(mbconv_channels[0] * width_scale) // 2, 32)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        self.in_conv_channels = self._round_channels(in_conv_channels, fix_stem_head)
        self.in_block = MBConvInOutBlock(
            in_channels,
            self.in_conv_channels,
            kernel_size=in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            norm_layer=self._norm_layer,
        )

        self._context = self.in_block.context
        self._downsample_factor = self.in_block.downsample_factor

        cur_in_channels = self.in_conv_channels
        cur_feats = None
        if self.time_se:
            cur_feats = (in_feats + in_stride - 1) // in_stride

        num_superblocks = len(self.b0_mbconv_repeats)
        self.mbconv_channels = [0] * num_superblocks
        self.mbconv_repeats = [0] * num_superblocks
        total_blocks = 0
        for i in range(num_superblocks):
            self.mbconv_channels[i] = self._round_channels(self.b0_mbconv_channels[i])
            self.mbconv_repeats[i] = self._round_repeats(self.b0_mbconv_repeats[i])
            total_blocks += self.mbconv_repeats[i]

        self.blocks = nn.ModuleList([])
        k = 0
        for i in range(num_superblocks):
            repeats_i = self.mbconv_repeats[i]
            channels_i = self.mbconv_channels[i]
            stride_i = self.mbconv_strides[i]
            kernel_size_i = self.mbconv_kernel_sizes[i]
            expansion_i = self.mbconv_expansions[i]
            drop_i = drop_connect_rate * k / (total_blocks - 1)
            block_i = MBConvBlock(
                cur_in_channels,
                channels_i,
                expansion_i,
                kernel_size_i,
                stride_i,
                hid_act,
                drop_connect_rate=drop_i,
                norm_layer=self._norm_layer,
                se_r=se_r,
                time_se=time_se,
                num_feats=cur_feats,
            )
            self.blocks.append(block_i)
            k += 1
            self._context += block_i.context * self._downsample_factor
            self._downsample_factor *= block_i.downsample_factor
            if self.time_se:
                cur_feats = (cur_feats + stride_i - 1) // stride_i

            for j in range(repeats_i - 1):
                drop_i = drop_connect_rate * k / (total_blocks - 1)
                block_i = MBConvBlock(
                    channels_i,
                    channels_i,
                    expansion_i,
                    kernel_size_i,
                    1,
                    hid_act,
                    drop_connect_rate=drop_i,
                    norm_layer=self._norm_layer,
                    se_r=se_r,
                    time_se=time_se,
                    num_feats=cur_feats,
                )
                self.blocks.append(block_i)
                k += 1
                self._context += block_i.context * self._downsample_factor

            cur_in_channels = channels_i

        # head feature block
        self.head_channels = self._round_channels(head_channels, fix_stem_head)
        self.head_block = MBConvInOutBlock(
            cur_in_channels,
            self.head_channels,
            kernel_size=1,
            stride=1,
            activation=hid_act,
            norm_layer=self._norm_layer,
        )

        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_rate)
            self.output = nn.Linear(self.head_channels, out_units)
            self.out_act = AF.create(out_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = "relu"
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=act_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def efficientnet_params(model_name):
        """Get efficientnet params based on model name."""
        return EfficientNet.params_dict[model_name]

    def _round_channels(self, channels, fix=False):
        """Calculate and round number of channels based on depth multiplier.
        It will make the number of channel multiple of 8
        """
        if fix:
            return channels
        divisor = 8  # this makes the number of channels a multiple of 8
        channels = channels * self.width_scale
        new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        if new_channels < 0.9 * channels:  # prevent rounding by more than 10%
            new_channels += divisor
        return int(new_channels)

    def _round_repeats(self, repeats):
        """Round number of block repeats based on depth multiplier."""
        return int(math.ceil(self.depth_scale * repeats))

    def _compute_out_size(self, in_size):
        out_size = int((in_size - 1) // self.in_stride + 1)

        for stride in self.mbconv_strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, self.in_channels, None, None)

    def out_shape(self, in_shape=None):
        if self.with_output:
            return (None, self.out_units)

        if in_shape is None:
            return (None, self.head_block.out_channels, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        return (in_shape[0], self.head_block.out_channels, H, W)

    def forward(self, x):

        x = self.in_block(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        x = self.head_block(x)

        if self.with_output:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def forward_hid_feats(self, x, layers=None, return_output=False):

        assert layers is not None or return_output
        if layers is None:
            layers = []

        if return_output:
            last_layer = len(self.blocks) + 1
        else:
            last_layer = max(layers)

        h = []
        x = self.in_block(x)
        if 0 in layers:
            h.append(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx + 1 in layers:
                h.append(x)
            if last_layer == idx + 1:
                return h

        x = self.head_block(x)
        if len(self.blocks) + 1 in layers:
            h.append(x)

        if return_output:
            return h, x

        return h

    def get_config(self):

        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {
            "effnet_type": self.effnet_type,
            "in_channels": self.in_channels,
            "in_conv_channels": self.b0_in_conv_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "mbconv_repeats": self.b0_mbconv_repeats,
            "mbconv_channels": self.b0_mbconv_channels,
            "mbconv_kernel_sizes": self.mbconv_kernel_sizes,
            "mbconv_strides": self.mbconv_strides,
            "mbconv_expansions": self.mbconv_expansions,
            "head_channels": self.head_channels,
            "width_scale": self.cfg_width_scale,
            "depth_scale": self.cfg_width_scale,
            "fix_stem_head": self.fix_stem_head,
            "out_units": self.out_units,
            "drop_connect_rate": self.drop_connect_rate,
            "dropout_rate": self.dropout_rate,
            "out_act": out_act,
            "hid_act": hid_act,
            "norm_layer": self.norm_layer,
            "se_r": self.se_r,
            "time_se": self.time_se,
            "in_feats": self.in_feats,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_dropouts(self, dropout_rate, drop_connect_rate):
        super().change_dropouts(dropout_rate)
        from ..layers import DropConnect2d

        for module in self.modules():
            if isinstance(module, DropConnect2d):
                module.p *= drop_connect_rate / self.drop_connect_rate

        self.drop_connect_rate = drop_connect_rate
        self.dropout_rate = dropout_rate

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "effnet_type",
            "in_channels",
            "in_conv_channels",
            "in_kernel_size",
            "in_stride",
            "mbconv_repeats",
            "mbconv_channels",
            "mbconv_kernel_sizes",
            "mbconv_strides",
            "mbconv_expansions",
            "head_channels",
            "width_scale",
            "depth_scale",
            "fix_stem_head",
            "out_units",
            "hid_act",
            "out_act",
            "norm_layer",
            "drop_connect_rate",
            "dropout_rate",
            "se_r",
            "time_se",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        net_types = list(EfficientNet.params_dict.keys())

        parser.add_argument(
            "--effnet-type",
            type=str.lower,
            default=net_types[0],
            choices=net_types,
            help=("EfficientNet type"),
        )

        parser.add_argument(
            "--in-channels", default=1, type=int, help=("number of input channels")
        )

        parser.add_argument(
            "--in-conv-channels",
            default=32,
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
            "--in-stride", default=2, type=int, help=("stride of input convolution")
        )

        parser.add_argument(
            "--mbconv-repeats",
            default=[1, 2, 2, 3, 3, 4, 1],
            type=int,
            nargs="+",
            help=("mbconv-mbconvs repeats for efficientnet-b0"),
        )

        parser.add_argument(
            "--mbconv-channels",
            default=[16, 24, 40, 80, 112, 192, 320],
            type=int,
            nargs="+",
            help=("mbconv-blocks channels for efficientnet-b0"),
        )

        parser.add_argument(
            "--mbconv-kernel-sizes",
            default=[3, 3, 5, 3, 5, 5, 3],
            nargs="+",
            type=int,
            help=("mbconv-size kernels for efficientnet-b0"),
        )

        parser.add_argument(
            "--mbconv-strides",
            default=[1, 2, 2, 2, 1, 2, 1],
            nargs="+",
            type=int,
            help=("mbconv-blocks strides for efficientnet-b0"),
        )

        parser.add_argument(
            "--mbconv-expansions",
            default=[1, 6, 6, 6, 6, 6, 6],
            nargs="+",
            type=int,
            help=("mbconv-blocks expansions for efficientnet-b0"),
        )

        parser.add_argument(
            "--head-channels",
            default=1280,
            type=int,
            help=("channels in the last conv block for efficientnet-b0"),
        )

        parser.add_argument(
            "--width-scale",
            default=None,
            type=int,
            help=(
                "width multiplicative factor wrt efficientnet-b0, if None inferred from effnet-type"
            ),
        )

        parser.add_argument(
            "--depth-scale",
            default=None,
            type=int,
            help=(
                "depth multiplicative factor wrt efficientnet-b0, if None inferred from effnet-type"
            ),
        )

        parser.add_argument(
            "--fix-stem-head",
            default=False,
            action="store_true",
            help=(
                "if True, the input and head conv blocks are not affected by the width-scale factor"
            ),
        )

        parser.add_argument(
            "--se-r",
            default=4,
            type=int,
            help=("squeeze ratio in squeeze-excitation blocks"),
        )

        parser.add_argument(
            "--time-se",
            default=False,
            action="store_true",
            help=("squeeze-excitation pooling operation in time-dimension only"),
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
                help="type of normalization layer",
            )
        except:
            pass

        try:
            parser.add_argument("--hid-act", default="swish", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--drop-connect-rate",
            default=0.2,
            type=float,
            help="layer drop probability",
        )

        try:
            parser.add_argument(
                "--dropout-rate", default=0, type=float, help="dropout probability"
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
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

        parser.add_argument(
            "--drop-connect-rate",
            default=0.2,
            type=float,
            help="layer drop probability",
        )

        try:
            parser.add_argument(
                "--dropout-rate", default=0, type=float, help="dropout probability"
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):

        valid_args = (
            "out_units",
            "override_dropouts",
            "drop_connect_rate",
            "dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args
