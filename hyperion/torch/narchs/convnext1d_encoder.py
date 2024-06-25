"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import (
    ConvNext1dBlock,
    ConvNext1dDownsampleBlock,
    ConvNext1dEndpoint,
    ConvNext1dStemBlock,
)
from ..layers import ActivationFactory as AF
from ..utils import scale_seq_lengths, seq_lengths_to_mask
from .convnext2d_encoder import ConvNextNormLayerType
from .net_arch import NetArch


class ConvNext1dShortName(str, Enum):
    ATTO = "atto"
    FEMTO = "femto"
    PICO = "pico"
    PICO2 = "pico2"
    PICO3 = "pico3"
    PICO4 = "pico4"
    PICO5 = "pico5"
    PICO6 = "pico6"
    PICO7 = "pico7"
    NANO = "nano"
    SMALL = "small"
    TINY = "tiny"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"

    @staticmethod
    def choices():
        return [o.value for o in ConvNext1dShortName]

    @staticmethod
    def to_config(short_name):
        if short_name == ConvNext1dShortName.ATTO:
            repeats = [2, 2, 6, 2]
            channels = [96, 128, 160, 320]
        elif short_name == ConvNext1dShortName.FEMTO:
            repeats = [2, 2, 6, 2]
            channels = [96, 128, 192, 384]
        elif short_name == ConvNext1dShortName.PICO:
            repeats = [2, 2, 6, 2]
            channels = [96, 128, 256, 512]
        elif short_name == ConvNext1dShortName.PICO2:
            repeats = [2, 2, 6, 2]
            channels = [128, 128, 256, 512]
        elif short_name == ConvNext1dShortName.PICO3:
            repeats = [2, 2, 6, 2]
            channels = [128, 256, 256, 512]
        elif short_name == ConvNext1dShortName.PICO4:
            repeats = [2, 2, 6, 2]
            channels = [256, 256, 256, 512]
        elif short_name == ConvNext1dShortName.PICO5:
            repeats = [2, 2, 6, 2]
            channels = [512, 512, 512, 512]
        elif short_name == ConvNext1dShortName.PICO6:
            repeats = [2, 2, 6, 2]
            channels = [512, 798, 1024, 1536]
        elif short_name == ConvNext1dShortName.PICO7:
            repeats = [2, 2, 6, 2]
            channels = [798, 1024, 1536, 2048]
        elif short_name == ConvNext1dShortName.NANO:
            repeats = [2, 2, 8, 2]
            channels = [96, 160, 320, 640]
        elif short_name == ConvNext1dShortName.TINY:
            repeats = [3, 3, 9, 3]
            channels = [96, 192, 384, 768]
        elif short_name == ConvNext1dShortName.SMALL:
            repeats = [3, 3, 27, 3]
            channels = [96, 192, 384, 768]
        elif short_name == ConvNext1dShortName.BASE:
            repeats = [3, 3, 27, 3]
            channels = [128, 256, 512, 1024]
        elif short_name == ConvNext1dShortName.LARGE:
            repeats = [3, 3, 27, 3]
            channels = [192, 384, 768, 1536]
        elif short_name == ConvNext1dShortName.XLARGE:
            repeats = [3, 3, 27, 3]
            channels = [256, 512, 1024, 2048]
        elif short_name == ConvNext1dShortName.HUGE:
            repeats = [3, 3, 27, 3]
            channels = [352, 704, 1408, 2816]
        else:
            raise ValueError(f"wrong ConvNext short name {short_name.value}")
        # if short_name == ConvNext1dShortName.ATTO:
        #     repeats = [2, 2, 6, 2]
        #     channels = [128, 196, 256, 384]
        # elif short_name == ConvNext1dShortName.FEMTO:
        #     repeats = [2, 2, 6, 2]
        #     channels = [196, 256, 384, 512]
        # elif short_name == ConvNext1dShortName.PICO:
        #     repeats = [2, 2, 6, 2]
        #     channels = [256, 384, 512, 640]
        # elif short_name == ConvNext1dShortName.NANO:
        #     repeats = [2, 2, 8, 2]
        #     channels = [256, 384, 512, 768]
        # elif short_name == ConvNext1dShortName.TINY:
        #     repeats = [3, 3, 9, 3]
        #     channels = [384, 512, 640, 768]
        # elif short_name == ConvNext1dShortName.BASE:
        #     repeats = [3, 3, 27, 3]
        #     channels = [384, 512, 768, 1024]
        # elif short_name == ConvNext1dShortName.LARGE:
        #     repeats = [3, 3, 27, 3]
        #     channels = [512, 768, 1024, 1536]
        # elif short_name == ConvNext1dShortName.HUGE:
        #     repeats = [3, 3, 27, 3]
        #     channels = [512, 1024, 1536, 2048]
        # else:
        #     raise ValueError(f"wrong ConvNext short name {short_name.value}")

        strides = [2, 2, 2]
        return repeats, channels, strides


class ConvNext1dEncoder(NetArch):
    """ConvNext1d V2 1d Encoder.

    Attributes:
        in_feats:    input channels
        in_kernel_size: kernel size of the stem layer
        in_stride:      stride of the stem layer
        short_name:     short_name of the configuration repeats and channel numbers per block
        convb_repeats:  List of repeats of convolutional layers in each superblock
        convb_channels: List of channels of convolutional layers in each superblock
        convb_kernel_sizes: List of kernel sizes of convolutional layers in each superblock
        convb_dilations: List of dilations of convolutional layers in each superblock
        downb_strides:  List of downsampling strides after each superblock
        head_channels:  number of channels in the output, if 0, no output layers
        hid_act:        hidden activation string
        head_act:       activation in the head, if head_channels > 0
        drop_path_rate: stochastic layer dropout, for stochastic network depth
        norm_layer:     type of normalization layer string
        multilayer:     if aggregates the features of the superblocks
        multilayer_concat: if True, it does aggregation by concat, otherwise by mean.
        endpoint_channels: number of channels in the endpoints when doing multilayer
        endpoint_layers:   which layers are endpoints, if None, all of them are
        endpoint_scale_layer: which layer is used to get the downsampling scale used as endpoint
    """

    def __init__(
        self,
        in_feats: int,
        in_kernel_size: int = 4,
        in_stride: int = 4,
        short_name: Optional[str] = None,
        convb_repeats: List[int] = [3, 3, 27, 3],
        convb_channels: List[int] = [384, 512, 768, 1024],
        convb_kernel_sizes: List[int] = [7],
        convb_dilations: List[int] = [1],
        downb_strides: List[int] = [2],
        head_channels: int = 0,
        hid_act: str = "gelu",
        head_act: Optional[str] = None,
        drop_path_rate: float = 0.0,
        norm_layer: ConvNextNormLayerType = ConvNextNormLayerType.LAYERNORM,
        multilayer: bool = False,
        multilayer_concat=False,
        endpoint_channels: Optional[int] = None,
        endpoint_layers: Optional[List[int]] = None,
        endpoint_scale_layer: int = -1,
    ):

        super().__init__()
        self.in_feats = in_feats
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        self.short_name = short_name
        if short_name is not None:
            convb_repeats, convb_channels, downb_strides = (
                ConvNext1dShortName.to_config(short_name)
            )

        num_superblocks = len(convb_repeats)
        self.num_superblocks = num_superblocks
        self.convb_repeats = convb_repeats
        self.convb_channels = self._standarize_resblocks_param(
            convb_channels, num_superblocks, "convb_channels"
        )
        self.convb_kernel_sizes = self._standarize_resblocks_param(
            convb_kernel_sizes, num_superblocks, "convb_kernel_sizes"
        )
        self.convb_dilations = self._standarize_resblocks_param(
            convb_dilations, num_superblocks, "convb_dilations"
        )
        self.downb_strides = self._standarize_resblocks_param(
            downb_strides, num_superblocks - 1, "downb_strides"
        )
        self.head_channels = head_channels
        self.hid_act = hid_act
        self.head_act = head_act
        self.drop_path_rate = drop_path_rate
        # self.in_feats = in_feats
        self.norm_layer = norm_layer
        if norm_layer is None or norm_layer == ConvNextNormLayerType.LAYERNORM:
            self._norm_layer = nn.LayerNorm
        else:
            raise Exception("TODO")

        # stem block
        in_block = ConvNext1dStemBlock(
            in_feats,
            self.convb_channels[0],
            kernel_size=in_kernel_size,
            stride=in_stride,
            norm_layer=self._norm_layer,
        )
        self._context = in_block.context
        self._downsample_factor = in_block.stride

        self.downsample_blocks = nn.ModuleList([in_block])
        self.convb_scales = [self._downsample_factor]
        for i in range(num_superblocks - 1):
            stride_i = self.downb_strides[i]
            if stride_i > 1 or self.convb_channels[i] != self.convb_channels[i + 1]:
                block_i = ConvNext1dDownsampleBlock(
                    self.convb_channels[i],
                    self.convb_channels[i + 1],
                    stride=stride_i,
                    norm_layer=self._norm_layer,
                )
                self._context += block_i.context * self._downsample_factor
                self._downsample_factor *= block_i.stride
            else:
                block_i = nn.Identity()

            self.downsample_blocks.append(block_i)
            self.convb_scales.append(self._downsample_factor)

        drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(convb_repeats))
        ]
        self.conv_blocks = nn.ModuleList()
        count = 0
        for i in range(num_superblocks):
            repeats_i = self.convb_repeats[i]
            channels_i = self.convb_channels[i]
            kernel_size_i = self.convb_kernel_sizes[i]
            dilation_i = self.convb_dilations[i]
            conv_block_i = nn.ModuleList()
            for j in range(repeats_i):
                block_ij = ConvNext1dBlock(
                    channels_i,
                    kernel_size=kernel_size_i,
                    dilation=dilation_i,
                    activation=hid_act,
                    norm_layer=self._norm_layer,
                    drop_path_rate=drop_rates[count],
                )
                count += 1
                conv_block_i.append(block_ij)
            self.conv_blocks.append(conv_block_i)

        if multilayer:
            if endpoint_layers is None:
                # if is None all layers are endpoints
                endpoint_layers = [i + 1 for i in range(num_superblocks)]

            if endpoint_channels is None:
                # if None, the number of endpoint channels matches the one of the endpoint level
                endpoint_channels = self.convb_channels[endpoint_scale_layer]

            # which layers are enpoints
            self.is_endpoint = [
                True if i + 1 in endpoint_layers else False
                for i in range(num_superblocks)
            ]
            # which endpoints have a projection layer ConvNext1dEndpoint
            self.has_endpoint_block = [False] * num_superblocks
            # relates endpoint layers to their ResNet1dEndpoint object
            self.endpoint_block_idx = [0] * num_superblocks
            endpoint_scale = self.convb_scales[endpoint_scale_layer]
            endpoint_blocks = nn.ModuleList([])
            cur_endpoint = 0
            in_concat_channels = 0
            for i in range(num_superblocks):
                if self.is_endpoint[i]:
                    if multilayer_concat:
                        out_channels = self.convb_channels[i]
                        if self.convb_scales[i] != endpoint_scale:
                            self.has_endpoint_block[i] = True

                        in_concat_channels += out_channels
                    else:
                        self.has_endpoint_block[i] = True
                        out_channels = endpoint_channels

                    if self.has_endpoint_block[i]:
                        endpoint_i = ConvNext1dEndpoint(
                            self.convb_channels[i],
                            out_channels,
                            in_scale=self.convb_scales[i],
                            out_scale=endpoint_scale,
                            norm_layer=self._norm_layer,
                        )
                        self.endpoint_block_idx[i] = cur_endpoint
                        endpoint_blocks.append(endpoint_i)
                        cur_endpoint += 1

            self.endpoint_blocks = endpoint_blocks
            if multilayer_concat:
                self.concat_endpoint_block = ConvNext1dEndpoint(
                    in_concat_channels,
                    endpoint_channels,
                    in_scale=1,
                    out_scale=1,
                    norm_layer=self._norm_layer,
                )
        else:
            endpoint_channels = self.convb_channels[-1]

        self.multilayer = multilayer
        self.multilayer_concat = multilayer_concat
        self.endpoint_channels = endpoint_channels
        self.endpoint_layers = endpoint_layers
        self.endpoint_scale_layer = endpoint_scale_layer

        # head feature block
        if self.head_channels > 0:
            self.head_norm = self._norm_layer(convb_channels[-1], eps=1e-6)
            self.head = nn.Linear(convb_channels[-1], head_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

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
        out_size = int((in_size - 1) // self.in_stride + 1)

        for stride in self.downb_strides:
            out_size = int((out_size - 1) // stride + 1)

        return out_size

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, self.in_feats, None)

    def out_shape(self, in_shape=None):
        out_channels = (
            self.head_channels if self.head_channels > 0 else self.endpoint_channels
        )
        if in_shape is None:
            return (None, out_channels, None)

        assert len(in_shape) == 3
        if in_shape[2] is None:
            T = None
        else:
            T = self._compute_out_size(in_shape[2])

        return (in_shape[0], out_channels, T)

    @staticmethod
    def _update_mask(
        x: torch.Tensor, x_lengths: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ):
        if x_lengths is None:
            return None

        if x_mask is not None and x.size(-1) == x_mask.size(-1):
            return x_mask

        return seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=2)

    @staticmethod
    def _match_lens(endpoints):
        lens = [e.shape[-1] for e in endpoints]
        min_len = min(lens)
        for i in range(len(endpoints)):
            if lens[i] > min_len:
                t_start = (lens[i] - min_len) // 2
                t_end = t_start + min_len
                endpoints[i] = endpoints[i][:, :, t_start:t_end]

        return endpoints

    def _merge_endpoints(self, endpoints):
        endpoints = self._match_lens(endpoints)
        if self.multilayer_concat:
            try:
                x = torch.cat(endpoints, dim=1)
            except:
                for k in range(len(endpoints)):
                    logging.error(
                        f"cat shape error ep={k},  shape{endpoints[k].size()}"
                    )

            x = self.concat_endpoint_block(x)
        else:
            x = torch.mean(torch.stack(endpoints), 0)

        return x

    def forward(self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None):

        x_mask = None
        endpoints = []
        for i in range(self.num_superblocks):
            max_length = x.size(-1)
            x = self.downsample_blocks[i](x)
            x_lengths = scale_seq_lengths(
                x_lengths, max_out_length=x.size(-1), max_in_length=max_length
            )
            x_mask = self._update_mask(x, x_lengths, x_mask)
            for j in range(self.convb_repeats[i]):
                x = self.conv_blocks[i][j](x, x_mask=x_mask)

            if self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)

                endpoints.append(endpoint_i)

        if self.multilayer:
            x = self._merge_endpoints(endpoints)

        x = x.contiguous()

        if self.head_channels > 0:
            x = self.head_norm(torch.mean(x, dim=2))
            x = self.head(x)

        return x

    def get_config(self):

        head_act = self.head_act
        hid_act = self.hid_act
        config = {
            "in_feats": self.in_feats,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "short_name": self.short_name,
            "convb_repeats": self.convb_repeats,
            "convb_channels": self.convb_channels,
            "convb_kernel_sizes": self.convb_kernel_sizes,
            "convb_dilations": self.convb_dilations,
            "downb_strides": self.downb_strides,
            "head_channels": self.head_channels,
            "hid_act": hid_act,
            "head_act": head_act,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "multilayer": self.multilayer,
            "multilayer_concat": self.multilayer_concat,
            "endpoint_channels": self.endpoint_channels,
            "endpoint_layers": self.endpoint_layers,
            "endpoint_scale_layer": self.endpoint_scale_layer,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, override_dropouts, drop_path_rate):
        if override_dropouts:
            logging.info("chaning convnext1d dropouts")
            self.change_dropouts(drop_path_rate)

    def change_dropouts(self, drop_path_rate):
        from ..layers import DropPath1d

        for module in self.modules():
            if isinstance(module, DropPath1d):
                module.p *= drop_path_rate / self.drop_path_rate

        self.drop_path_rate = drop_path_rate

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(ConvNext1dEncoder.__init__, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--in-feats", type=int, help=("input channel dimension"))

        parser.add_argument(
            "--in-kernel-size",
            default=4,
            type=int,
            help=("kernel size of input convolution"),
        )

        parser.add_argument(
            "--in-stride", default=4, type=int, help=("stride of input convolution")
        )

        parser.add_argument(
            "--short-name",
            default=None,
            choices=ConvNext1dShortName.choices(),
            help="short_name of the configuration repeats and channel numbers per block",
        )

        parser.add_argument(
            "--convb-repeats",
            default=[3, 3, 27, 3],
            type=int,
            nargs="+",
            help=("conv-blocks repeats in each encoder stage"),
        )

        parser.add_argument(
            "--convb-channels",
            default=[384, 512, 768, 1024],
            type=int,
            nargs="+",
            help=("conv-blocks channels for each stage"),
        )

        parser.add_argument(
            "--convb-kernel-sizes",
            default=[7],
            type=int,
            nargs="+",
            help=("conv-blocks kernels for each stage"),
        )

        parser.add_argument(
            "--convb-dilations",
            default=[1],
            type=int,
            nargs="+",
            help=("conv-blocks dilations for each stage"),
        )
        parser.add_argument(
            "--downb-strides",
            default=[2],
            nargs="+",
            type=int,
            help=("resb-blocks strides for each encoder stage"),
        )

        if "head_channels" not in skip:
            parser.add_argument(
                "--head-channels",
                default=0,
                type=int,
                help=("channels in the last conv block of encoder"),
            )

        try:
            parser.add_argument("--hid-act", default="gelu", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--head-act", default=None, help="activation in encoder head"
        )

        parser.add_argument(
            "--drop-path-rate",
            default=0,
            type=float,
            help="stochastic depth drop probability",
        )

        parser.add_argument(
            "--norm-layer",
            default=ConvNextNormLayerType.LAYERNORM.value,
            choices=ConvNextNormLayerType.choices(),
            help="type of normalization layer",
        )

        parser.add_argument(
            "--multilayer",
            default=False,
            action=ActionYesNo,
            help="use multilayer feature aggregation (mfa)",
        )

        parser.add_argument(
            "--multilayer-concat",
            default=False,
            action=ActionYesNo,
            help="use concatenation for mfa",
        )

        parser.add_argument(
            "--endpoint-channels",
            default=None,
            type=int,
            help=("num. endpoint channels when using mfa"),
        )

        parser.add_argument(
            "--endpoint-layers",
            default=None,
            nargs="+",
            type=int,
            help=(
                "layers to aggreagate in mfa, "
                "if None, all residual blocks are aggregated"
            ),
        )

        parser.add_argument(
            "--endpoint-scale-layer",
            default=-1,
            type=int,
            help=("layer number which indicates the time scale in mfa"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):

        valid_args = (
            "override_dropouts",
            "drop_path_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser, prefix=None, skip=set([])):
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
            parser.add_argument(
                "--drop-path-rate",
                default=0,
                type=float,
                help="layer drop probability",
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    # class ConvNext1dEncoder(NetArch):
    #     def __init__(
    #         self,
    #         in_feats,
    #         in_conv_channels=128,
    #         in_kernel_size=3,
    #         in_stride=1,
    #         resb_type="basic",
    #         resb_repeats=[1, 1, 1],
    #         resb_channels=128,
    #         resb_kernel_sizes=3,
    #         resb_strides=2,
    #         resb_dilations=1,
    #         resb_groups=1,
    #         head_channels=0,
    #         hid_act="relu",
    #         head_act=None,
    #         dropout_rate=0,
    #         drop_connect_rate=0,
    #         se_r=16,
    #         res2net_width_factor=1,
    #         res2net_scale=4,
    #         multilayer=False,
    #         multilayer_concat=False,
    #         endpoint_channels=None,
    #         endpoint_layers=None,
    #         endpoint_scale_layer=-1,
    #         use_norm=True,
    #         norm_layer=None,
    #         norm_before=True,
    #         upsampling_mode="nearest",
    #     ):

    #         super().__init__()

    #         self.resb_type = resb_type
    #         bargs = {}  # block's extra arguments
    #         if resb_type == "basic":
    #             self._block = ResNet1dBasicBlock
    #         elif resb_type == "bn":
    #             self._block = ResNet1dBNBlock
    #         elif resb_type == "sebasic":
    #             self._block = SEResNet1dBasicBlock
    #             bargs["se_r"] = se_r
    #         elif resb_type == "sebn":
    #             self._block = SEResNet1dBNBlock
    #             bargs["se_r"] = se_r
    #         elif resb_type in ["res2basic", "seres2basic", "res2bn", "seres2bn"]:
    #             bargs["width_factor"] = res2net_width_factor
    #             bargs["scale"] = res2net_scale
    #             if resb_type in ["seres2basic", "seres2bn"]:
    #                 bargs["se_r"] = se_r
    #             if resb_type in ["res2basic", "seres2basic"]:
    #                 self._block = Res2Net1dBasicBlock
    #             else:
    #                 self._block = Res2Net1dBNBlock

    #         self.in_feats = in_feats
    #         self.in_conv_channels = in_conv_channels
    #         self.in_kernel_size = in_kernel_size
    #         self.in_stride = in_stride
    #         num_superblocks = len(resb_repeats)
    #         self.resb_repeats = resb_repeats
    #         self.resb_channels = self._standarize_resblocks_param(
    #             resb_channels, num_superblocks, "resb_channels"
    #         )
    #         self.resb_kernel_sizes = self._standarize_resblocks_param(
    #             resb_kernel_sizes, num_superblocks, "resb_kernel_sizes"
    #         )
    #         self.resb_strides = self._standarize_resblocks_param(
    #             resb_strides, num_superblocks, "resb_strides"
    #         )
    #         self.resb_dilations = self._standarize_resblocks_param(
    #             resb_dilations, num_superblocks, "resb_dilations"
    #         )
    #         self.resb_groups = resb_groups
    #         self.head_channels = head_channels
    #         self.hid_act = hid_act
    #         self.head_act = head_act
    #         self.dropout_rate = dropout_rate
    #         self.drop_connect_rate = drop_connect_rate
    #         self.use_norm = use_norm
    #         self.norm_before = norm_before
    #         self.se_r = se_r
    #         self.res2net_width_factor = res2net_width_factor
    #         self.res2net_scale = res2net_scale
    #         self.norm_layer = norm_layer
    #         norm_groups = None
    #         if norm_layer == "group-norm":
    #             norm_groups = min(np.min(resb_channels) // 2, 32)
    #             norm_groups = max(norm_groups, resb_groups)
    #         self._norm_layer = NLF.create(norm_layer, norm_groups)

    #         # stem block
    #         self.in_block = DC1dEncBlock(
    #             in_feats,
    #             in_conv_channels,
    #             in_kernel_size,
    #             stride=in_stride,
    #             activation=hid_act,
    #             dropout_rate=dropout_rate,
    #             use_norm=use_norm,
    #             norm_layer=self._norm_layer,
    #             norm_before=norm_before,
    #         )
    #         self._context = self.in_block.context
    #         self._downsample_factor = self.in_block.stride

    #         cur_in_channels = in_conv_channels
    #         total_blocks = np.sum(self.resb_repeats)

    #         # middle blocks
    #         self.blocks = nn.ModuleList([])
    #         k = 0
    #         self.resb_scales = []
    #         for i in range(num_superblocks):
    #             blocks_i = nn.ModuleList([])
    #             repeats_i = self.resb_repeats[i]
    #             channels_i = self.resb_channels[i]
    #             stride_i = self.resb_strides[i]
    #             kernel_size_i = self.resb_kernel_sizes[i]
    #             dilation_i = self.resb_dilations[i]
    #             # if there is downsampling the dilation of the first block
    #             # is set to 1
    #             dilation_i1 = dilation_i if stride_i == 1 else 1
    #             drop_i = drop_connect_rate * k / (total_blocks - 1)
    #             block_i1 = self._block(
    #                 cur_in_channels,
    #                 channels_i,
    #                 kernel_size_i,
    #                 stride=stride_i,
    #                 dilation=dilation_i1,
    #                 groups=self.resb_groups,
    #                 activation=hid_act,
    #                 dropout_rate=dropout_rate,
    #                 drop_connect_rate=drop_i,
    #                 use_norm=use_norm,
    #                 norm_layer=self._norm_layer,
    #                 norm_before=norm_before,
    #                 **bargs,
    #             )

    #             blocks_i.append(block_i1)
    #             k += 1
    #             self._context += block_i1.context * self._downsample_factor
    #             self._downsample_factor *= block_i1.downsample_factor
    #             self.resb_scales.append(self._downsample_factor)

    #             for j in range(repeats_i - 1):
    #                 drop_i = drop_connect_rate * k / (total_blocks - 1)
    #                 block_ij = self._block(
    #                     channels_i,
    #                     channels_i,
    #                     kernel_size_i,
    #                     stride=1,
    #                     dilation=dilation_i,
    #                     groups=self.resb_groups,
    #                     activation=hid_act,
    #                     dropout_rate=dropout_rate,
    #                     drop_connect_rate=drop_i,
    #                     use_norm=use_norm,
    #                     norm_layer=self._norm_layer,
    #                     norm_before=norm_before,
    #                     **bargs,
    #                 )
    #                 blocks_i.append(block_ij)
    #                 k += 1
    #                 self._context += block_ij.context * self._downsample_factor
    #             self.blocks.append(blocks_i)

    #             cur_in_channels = channels_i

    #         if multilayer:
    #             if endpoint_layers is None:
    #                 # if is None all layers are endpoints
    #                 endpoint_layers = [i + 1 for i in range(num_superblocks)]

    #             if endpoint_channels is None:
    #                 # if None, the number of endpoint channels matches the one of the endpoint level
    #                 endpoint_channels = self.resb_channels[endpoint_scale_layer]

    #             # which layers are enpoints
    #             self.is_endpoint = [
    #                 True if i + 1 in endpoint_layers else False
    #                 for i in range(num_superblocks)
    #             ]
    #             # which endpoints have a projection layer ResNet1dEndpoint
    #             self.has_endpoint_block = [False] * num_superblocks
    #             # relates endpoint layers to their ResNet1dEndpoint object
    #             self.endpoint_block_idx = [0] * num_superblocks
    #             endpoint_scale = self.resb_scales[endpoint_scale_layer]
    #             endpoint_blocks = nn.ModuleList([])
    #             cur_endpoint = 0
    #             in_concat_channels = 0
    #             for i in range(num_superblocks):
    #                 if self.is_endpoint[i]:
    #                     if multilayer_concat:
    #                         out_channels = self.resb_channels[i]
    #                         if self.resb_scales[i] != endpoint_scale:
    #                             self.has_endpoint_block[i] = True

    #                         # if self.resb_channels[i] != endpoint_channels:
    #                         #     out_channels = endpoint_channels
    #                         #     self.has_endpoint_block[i] = True

    #                         in_concat_channels += out_channels
    #                     else:
    #                         self.has_endpoint_block[i] = True
    #                         out_channels = endpoint_channels

    #                     if self.has_endpoint_block[i]:
    #                         endpoint_i = ResNet1dEndpoint(
    #                             self.resb_channels[i],
    #                             out_channels,
    #                             in_scale=self.resb_scales[i],
    #                             scale=endpoint_scale,
    #                             activation=hid_act,
    #                             upsampling_mode=upsampling_mode,
    #                             norm_layer=self._norm_layer,
    #                             norm_before=norm_before,
    #                         )
    #                         self.endpoint_block_idx[i] = cur_endpoint
    #                         endpoint_blocks.append(endpoint_i)
    #                         cur_endpoint += 1

    #             self.endpoint_blocks = endpoint_blocks
    #             if multilayer_concat:
    #                 self.concat_endpoint_block = ResNet1dEndpoint(
    #                     in_concat_channels,
    #                     endpoint_channels,
    #                     in_scale=1,
    #                     scale=1,
    #                     activation=hid_act,
    #                     norm_layer=self._norm_layer,
    #                     norm_before=norm_before,
    #                 )
    #         else:
    #             endpoint_channels = self.resb_channels[-1]

    #         self.multilayer = multilayer
    #         self.multilayer_concat = multilayer_concat
    #         self.endpoint_channels = endpoint_channels
    #         self.endpoint_layers = endpoint_layers
    #         self.endpoint_scale_layer = endpoint_scale_layer
    #         self.upsampling_mode = upsampling_mode

    #         # head feature block
    #         if self.head_channels > 0:
    #             self.head_block = DC1dEncBlock(
    #                 cur_in_channels,
    #                 head_channels,
    #                 kernel_size=1,
    #                 stride=1,
    #                 activation=head_act,
    #                 use_norm=False,
    #                 norm_before=norm_before,
    #             )

    #         self._init_weights(hid_act)

    #     def _init_weights(self, hid_act):
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv1d):
    #                 if isinstance(hid_act, str):
    #                     act_name = hid_act
    #                 if isinstance(hid_act, dict):
    #                     act_name = hid_act["name"]
    #                 if act_name == "swish":
    #                     act_name = "relu"
    #                 try:
    #                     nn.init.kaiming_normal_(
    #                         m.weight, mode="fan_out", nonlinearity=act_name
    #                     )
    #                 except:
    #                     nn.init.kaiming_normal_(
    #                         m.weight, mode="fan_out", nonlinearity="relu"
    #                     )
    #             elif isinstance(m, nn.BatchNorm1d):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)

    #     @staticmethod
    #     def _standarize_resblocks_param(p, num_blocks, p_name):
    #         if isinstance(p, int):
    #             p = [p] * num_blocks
    #         elif isinstance(p, list):
    #             if len(p) == 1:
    #                 p = p * num_blocks

    #             assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
    #                 p_name,
    #                 len(p),
    #                 num_blocks,
    #             )
    #         else:
    #             raise TypeError("wrong type for param {}={}".format(p_name, p))

    #         return p

    #     def _compute_out_size(self, in_size):
    #         out_size = int((in_size - 1) // self.in_stride + 1)

    #         if self.multilayer:
    #             strides = self.resb_strides[self.endpoint_scale_layer]
    #         else:
    #             strides = self.resb_strides

    #         for stride in strides:
    #             out_size = int((out_size - 1) // stride + 1)

    #         return out_size

    #     def in_context(self):
    #         return (self._context, self._context)

    #     def in_shape(self):
    #         return (None, self.in_feats, None)

    #     def out_shape(self, in_shape=None):
    #         out_channels = (
    #             self.head_channels if self.head_channels > 0 else self.endpoint_channels
    #         )
    #         if in_shape is None:
    #             return (None, out_channels, None)

    #         assert len(in_shape) == 3
    #         if in_shape[2] is None:
    #             T = None
    #         else:
    #             T = self._compute_out_size(in_shape[2])

    #         return (in_shape[0], out_channels, T)

    #     @staticmethod
    #     def _match_lens(endpoints):
    #         lens = [e.shape[-1] for e in endpoints]
    #         min_len = min(lens)
    #         for i in range(len(endpoints)):
    #             if lens[i] > min_len:
    #                 t_start = (lens[i] - min_len) // 2
    #                 t_end = t_start + min_len
    #                 endpoints[i] = endpoints[i][:, :, t_start:t_end]

    #         return endpoints

    #     @staticmethod
    #     def _update_mask(x, x_lengths, x_mask=None):
    #         if x_lengths is None:
    #             return None

    #         if x_mask is not None and x.size(-1) == x_mask.size(-1):
    #             return x_mask

    #         return seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=2)

    #     def forward(self, x, x_lengths=None):
    #         """forward function

    #         Args:
    #            x: input tensor of size=(batch, C, time)
    #            x_lengths:  it contains the lengths of the sequences.
    #         Returns:
    #            Tensor with output logits of size=(batch, out_units) if out_units>0,
    #            otherwise, it returns tensor of represeantions of size=(batch, Cout, out_time)

    #         """

    #         x_mask = self._update_mask(x, x_lengths)
    #         x = self.in_block(x, x_mask=x_mask)
    #         endpoints = []

    #         for i, superblock in enumerate(self.blocks):
    #             for j, block in enumerate(superblock):
    #                 x_mask = self._update_mask(x, x_lengths, x_mask)
    #                 x = block(x, x_mask=x_mask)

    #             if self.multilayer and self.is_endpoint[i]:
    #                 endpoint_i = x
    #                 if self.has_endpoint_block[i]:
    #                     idx = self.endpoint_block_idx[i]
    #                     endpoint_i = self.endpoint_blocks[idx](endpoint_i)

    #                 endpoints.append(endpoint_i)

    #         if self.multilayer:
    #             endpoints = self._match_lens(endpoints)
    #             if self.multilayer_concat:
    #                 try:
    #                     x = torch.cat(endpoints, dim=1)
    #                 except:
    #                     for k in range(len(endpoints)):
    #                         print("epcat ", k, endpoints[k].shape, flush=True)

    #                 x = self.concat_endpoint_block(x)
    #             else:
    #                 x = torch.mean(torch.stack(endpoints), 0)

    #         if self.head_channels > 0:
    #             x_mask = self._update_mask(x, x_lengths, x_mask)
    #             x = self.head_block(x)

    #         return x

    #     def forward_hid_feats(self, x, x_lengths=None, layers=None, return_output=False):

    #         assert layers is not None or return_output
    #         if layers is None:
    #             layers = []

    #         if return_output:
    #             last_layer = len(self.blocks) + 1
    #         else:
    #             last_layer = max(layers)

    #         h = []
    #         x = self.in_block(x)
    #         if 0 in layers:
    #             h.append(x)

    #         endpoints = []
    #         for i, superblock in enumerate(self.blocks):
    #             for j, block in enumerate(superblock):
    #                 x = block(x)

    #             if i + 1 in layers:
    #                 h.append(x)

    #             if return_output and self.multilayer and self.is_endpoint[i]:
    #                 endpoint_i = x
    #                 if self.has_endpoint_block[i]:
    #                     idx = self.endpoint_block_idx[i]
    #                     endpoint_i = self.endpoint_blocks[idx](endpoint_i)
    #                 endpoints.append(endpoint_i)

    #             if last_layer == i + 1:
    #                 break

    #         if not return_output:
    #             return h

    #         if self.multilayer:
    #             if self.multilayer_concat:
    #                 x = torch.cat(endpoints, dim=1)
    #                 x = self.concat_endpoint_block(x)
    #             else:
    #                 x = torch.mean(torch.stack(endpoints), 0)

    #         if self.head_channels > 0:
    #             x = self.head_block(x)

    #         return h, x

    #     def get_config(self):

    #         head_act = self.head_act
    #         hid_act = self.hid_act

    #         config = {
    #             "in_feats": self.in_feats,
    #             "in_conv_channels": self.in_conv_channels,
    #             "in_kernel_size": self.in_kernel_size,
    #             "in_stride": self.in_stride,
    #             "resb_type": self.resb_type,
    #             "resb_repeats": self.resb_repeats,
    #             "resb_channels": self.resb_channels,
    #             "resb_kernel_sizes": self.resb_kernel_sizes,
    #             "resb_strides": self.resb_strides,
    #             "resb_dilations": self.resb_dilations,
    #             "resb_groups": self.resb_groups,
    #             "head_channels": self.head_channels,
    #             "dropout_rate": self.dropout_rate,
    #             "drop_connect_rate": self.drop_connect_rate,
    #             "hid_act": hid_act,
    #             "head_act": head_act,
    #             "se_r": self.se_r,
    #             "res2net_width_factor": self.res2net_width_factor,
    #             "res2net_scale": self.res2net_scale,
    #             "use_norm": self.use_norm,
    #             "norm_layer": self.norm_layer,
    #             "norm_before": self.norm_before,
    #             "multilayer": self.multilayer,
    #             "multilayer_concat": self.multilayer_concat,
    #             "endpoint_channels": self.endpoint_channels,
    #             "endpoint_layers": self.endpoint_layers,
    #             "endpoint_scale_layer": self.endpoint_scale_layer,
    #             "upsampling_mode": self.upsampling_mode,
    #         }

    #         base_config = super().get_config()
    #         return dict(list(base_config.items()) + list(config.items()))

    #     def change_config(self, override_dropouts, dropout_rate, drop_connect_rate):
    #         if override_dropouts:
    #             logging.info("chaning resnet1d dropouts")
    #             self.change_dropouts(dropout_rate, drop_connect_rate)

    #     def change_dropouts(self, dropout_rate, drop_connect_rate):
    #         super().change_dropouts(dropout_rate)
    #         from ..layers import DropConnect1d

    #         for module in self.modules():
    #             if isinstance(module, DropConnect1d):
    #                 module.p *= drop_connect_rate / self.drop_connect_rate

    #         self.drop_connect_rate = drop_connect_rate
    #         self.dropout_rate = dropout_rate

    #     @staticmethod
    #     def filter_args(**kwargs):
    #         return filter_func_args(ResNet1dEncoder.__init__, kwargs)
    #         # valid_args = (
    #         #     "in_feats",
    #         #     "in_conv_channels",
    #         #     "in_kernel_size",
    #         #     "in_stride",
    #         #     "resb_type",
    #         #     "resb_repeats",
    #         #     "resb_channels",
    #         #     "resb_kernel_sizes",
    #         #     "resb_strides",
    #         #     "resb_dilations",
    #         #     "resb_groups",
    #         #     "head_channels",
    #         #     "se_r",
    #         #     "res2net_width_factor",
    #         #     "res2net_scale",
    #         #     "hid_act",
    #         #     "head_act",
    #         #     "dropout_rate",
    #         #     "drop_connect_rate",
    #         #     "use_norm",
    #         #     "norm_layer",
    #         #     "norm_before",
    #         #     "multilayer",
    #         #     "multilayer_concat",
    #         #     "endpoint_channels",
    #         #     "endpoint_layers",
    #         #     "endpoint_scale_layer",
    #         #     "upsampling_mode",
    #         # )

    #         # args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    #         # return args

    #     @staticmethod
    #     def add_class_args(parser, prefix=None, skip=set(["in_feats"])):
    #         if prefix is not None:
    #             outer_parser = parser
    #             parser = ArgumentParser(prog="")

    #         if "in_feats" not in skip:
    #             parser.add_argument(
    #                 "--in-feats", type=int, required=True, help=("input feature dimension")
    #             )

    #         parser.add_argument(
    #             "--in-conv-channels",
    #             default=128,
    #             type=int,
    #             help=("number of output channels in input convolution"),
    #         )

    #         parser.add_argument(
    #             "--in-kernel-size",
    #             default=3,
    #             type=int,
    #             help=("kernel size of input convolution"),
    #         )

    #         parser.add_argument(
    #             "--in-stride", default=1, type=int, help=("stride of input convolution")
    #         )

    #         parser.add_argument(
    #             "--resb-type",
    #             default="basic",
    #             choices=[
    #                 "basic",
    #                 "bn",
    #                 "sebasic",
    #                 "sebn",
    #                 "res2basic",
    #                 "res2bn",
    #                 "seres2basic",
    #                 "seres2bn",
    #             ],
    #             help=("residual blocks type"),
    #         )

    #         parser.add_argument(
    #             "--resb-repeats",
    #             default=[1, 1, 1],
    #             type=int,
    #             nargs="+",
    #             help=("resb-blocks repeats in each encoder stage"),
    #         )

    #         parser.add_argument(
    #             "--resb-channels",
    #             default=[128, 64, 32],
    #             type=int,
    #             nargs="+",
    #             help=("resb-blocks channels for each stage"),
    #         )

    #         parser.add_argument(
    #             "--resb-kernel-sizes",
    #             default=[3],
    #             nargs="+",
    #             type=int,
    #             help=("resb-blocks kernels for each encoder stage"),
    #         )

    #         parser.add_argument(
    #             "--resb-strides",
    #             default=[2],
    #             nargs="+",
    #             type=int,
    #             help=("resb-blocks strides for each encoder stage"),
    #         )

    #         parser.add_argument(
    #             "--resb-dilations",
    #             default=[1],
    #             nargs="+",
    #             type=int,
    #             help=("resb-blocks dilations for each encoder stage"),
    #         )

    #         parser.add_argument(
    #             "--resb-groups",
    #             default=1,
    #             type=int,
    #             help=("resb-blocks groups in convolutions"),
    #         )

    #         if "head_channels" not in skip:
    #             parser.add_argument(
    #                 "--head-channels",
    #                 default=0,
    #                 type=int,
    #                 help=("channels in the last conv block of encoder"),
    #             )

    #         try:
    #             parser.add_argument("--hid-act", default="relu", help="hidden activation")
    #         except:
    #             pass

    #         parser.add_argument(
    #             "--head-act", default=None, help="activation in encoder head"
    #         )

    #         try:
    #             parser.add_argument(
    #                 "--dropout-rate", default=0, type=float, help="dropout probability"
    #             )
    #         except:
    #             pass

    #         try:
    #             parser.add_argument(
    #                 "--drop-connect-rate",
    #                 default=0,
    #                 type=float,
    #                 help="layer drop probability",
    #             )
    #         except:
    #             pass

    #         try:
    #             parser.add_argument(
    #                 "--norm-layer",
    #                 default=None,
    #                 choices=[
    #                     "batch-norm",
    #                     "group-norm",
    #                     "instance-norm",
    #                     "instance-norm-affine",
    #                     "layer-norm",
    #                 ],
    #                 help="type of normalization layer",
    #             )
    #         except:
    #             pass

    #         # parser.add_argument(
    #         #     "--wo-norm",
    #         #     default=False,
    #         #     action="store_true",
    #         #     help="without batch normalization",
    #         # )

    #         # parser.add_argument(
    #         #     "--norm-after",
    #         #     default=False,
    #         #     action="store_true",
    #         #     help="batch normalizaton after activation",
    #         # )
    #         parser.add_argument(
    #             "--use-norm",
    #             default=True,
    #             action=ActionYesNo,
    #             help="without batch normalization",
    #         )

    #         parser.add_argument(
    #             "--norm-before",
    #             default=True,
    #             action=ActionYesNo,
    #             help="batch normalizaton before activation",
    #         )

    #         parser.add_argument(
    #             "--se-r",
    #             default=16,
    #             type=int,
    #             help=("squeeze-excitation compression ratio"),
    #         )

    #         parser.add_argument(
    #             "--res2net-width-factor",
    #             default=1,
    #             type=float,
    #             help=(
    #                 "scaling factor for channels in middle layer "
    #                 "of res2net bottleneck blocks"
    #             ),
    #         )

    #         parser.add_argument(
    #             "--res2net-scale",
    #             default=1,
    #             type=int,
    #             help=("res2net scaling parameter "),
    #         )

    #         parser.add_argument(
    #             "--multilayer",
    #             default=False,
    #             action=ActionYesNo,
    #             help="use multilayer feature aggregation (mfa)",
    #         )

    #         parser.add_argument(
    #             "--multilayer-concat",
    #             default=False,
    #             action=ActionYesNo,
    #             help="use concatenation for mfa",
    #         )

    #         parser.add_argument(
    #             "--endpoint-channels",
    #             default=None,
    #             type=int,
    #             help=("num. endpoint channels when using mfa"),
    #         )

    #         parser.add_argument(
    #             "--endpoint-layers",
    #             default=None,
    #             nargs="+",
    #             type=int,
    #             help=(
    #                 "layers to aggreagate in mfa, "
    #                 "if None, all residual blocks are aggregated"
    #             ),
    #         )

    #         parser.add_argument(
    #             "--endpoint-scale-layer",
    #             default=-1,
    #             type=int,
    #             help=("layer number which indicates the time scale in mfa"),
    #         )

    #         parser.add_argument(
    #             "--upsampling-mode",
    #             choices=["nearest", "bilinear", "subpixel"],
    #             default="nearest",
    #             help=("upsampling method when upsampling feature maps for mfa"),
    #         )

    #         if prefix is not None:
    #             outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    #     add_argparse_args = add_class_args

    # @staticmethod
    # def filter_finetune_args(**kwargs):

    #     valid_args = (
    #         "override_dropouts",
    #         "drop_connect_rate",
    #         "dropout_rate",
    #     )
    #     args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
    #     return args

    # @staticmethod
    # def add_finetune_args(parser, prefix=None, skip=set([])):
    #     if prefix is not None:
    #         outer_parser = parser
    #         parser = ArgumentParser(prog="")

    #     try:
    #         parser.add_argument(
    #             "--override-dropouts",
    #             default=False,
    #             action=ActionYesNo,
    #             help=(
    #                 "whether to use the dropout probabilities passed in the "
    #                 "arguments instead of the defaults in the pretrained model."
    #             ),
    #         )
    #     except:
    #         pass

    #     try:
    #         parser.add_argument(
    #             "--dropout-rate", default=0, type=float, help="dropout probability"
    #         )
    #     except:
    #         pass

    #     try:
    #         parser.add_argument(
    #             "--drop-connect-rate",
    #             default=0,
    #             type=float,
    #             help="layer drop probability",
    #         )
    #     except:
    #         pass

    #     if prefix is not None:
    #         outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
