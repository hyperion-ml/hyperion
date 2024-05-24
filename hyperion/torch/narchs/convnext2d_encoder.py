"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import (
    ConvNext2dBlock,
    ConvNext2dDownsampleBlock,
    ConvNext2dEndpoint,
    ConvNext2dStemBlock,
)
from ..layers import ActivationFactory as AF
from ..utils import scale_seq_lengths, seq_lengths_to_mask
from .net_arch import NetArch


class ConvNext2dShortName(str, Enum):
    ATTO = "atto"
    FEMTO = "femto"
    PICO = "pico"
    NANO = "nano"
    TINY = "tiny"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"

    @staticmethod
    def choices():
        return [o.value for o in ConvNext2dShortName]

    @staticmethod
    def to_config(short_name):
        if short_name == ConvNext2dShortName.ATTO:
            repeats = [2, 2, 6, 2]
            channels = [40, 80, 160, 320]
        elif short_name == ConvNext2dShortName.FEMTO:
            repeats = [2, 2, 6, 2]
            channels = [48, 96, 192, 384]
        elif short_name == ConvNext2dShortName.PICO:
            repeats = [2, 2, 6, 2]
            channels = [64, 128, 256, 512]
        elif short_name == ConvNext2dShortName.NANO:
            repeats = [2, 2, 8, 2]
            channels = [80, 160, 320, 640]
        elif short_name == ConvNext2dShortName.TINY:
            repeats = [3, 3, 9, 3]
            channels = [96, 192, 384, 768]
        elif short_name == ConvNext2dShortName.BASE:
            repeats = [3, 3, 27, 3]
            channels = [128, 256, 512, 1024]
        elif short_name == ConvNext2dShortName.LARGE:
            repeats = [3, 3, 27, 3]
            channels = [192, 384, 768, 1536]
        elif short_name == ConvNext2dShortName.HUGE:
            repeats = [3, 3, 27, 3]
            channels = [352, 704, 1408, 2816]
        else:
            raise ValueError(f"wrong ConvNext short name {short_name.value}")

        strides = [2, 2, 2]
        return repeats, channels, strides


class ConvNextNormLayerType(str, Enum):
    LAYERNORM = "layer-norm"
    RMSNORM = "rms-norm"

    @staticmethod
    def choices():
        return [o.value for o in ConvNextNormLayerType]


class ConvNext2dEncoder(NetArch):
    """ConvNext2d V2 2d Encoder.

    Attributes:
        in_channels:    input channels
        in_kernel_size: kernel size of the stem layer
        in_stride:      stride of the stem layer
        short_name:     short_name of the configuration repeats and channel numbers per block
        convb_repeats:  List of repeats of convolutional layers in each superblock
        convb_channels: List of channels of convolutional layers in each superblock
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
        in_channels: int = 1,
        in_kernel_size: int = 4,
        in_stride: int = 4,
        short_name: Optional[str] = None,
        convb_repeats: List[int] = [3, 3, 27, 3],
        convb_channels: List[int] = [128, 256, 512, 1024],
        downb_strides: int = 2,
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
        self.in_channels = in_channels
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        self.short_name = short_name
        if short_name is not None:
            convb_repeats, convb_channels, downb_strides = (
                ConvNext2dShortName.to_config(short_name)
            )

        num_superblocks = len(convb_repeats)
        self.num_superblocks = num_superblocks
        self.convb_repeats = convb_repeats
        self.convb_channels = self._standarize_resblocks_param(
            convb_channels, num_superblocks, "convb_channels"
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
        in_block = ConvNext2dStemBlock(
            in_channels,
            self.convb_channels[0],
            kernel_size=in_kernel_size,
            stride=in_stride,
            norm_layer=self._norm_layer,
        )
        self._context = in_block.context
        self._downsample_factor = in_block.stride

        self.downsample_blocks = nn.ModuleList([in_block])
        self.resb_scales = [self._downsample_factor]
        for i in range(num_superblocks - 1):
            block_i = ConvNext2dDownsampleBlock(
                self.convb_channels[i],
                self.convb_channels[i + 1],
                stride=self.downb_strides[i],
                norm_layer=self._norm_layer,
            )
            self.downsample_blocks.append(block_i)
            self._context += block_i.context * self._downsample_factor
            self._downsample_factor *= block_i.stride
            self.resb_scales = [self._downsample_factor]

        drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(convb_repeats))
        ]
        self.conv_blocks = nn.ModuleList()
        count = 0
        for i in range(num_superblocks):
            repeats_i = self.convb_repeats[i]
            channels_i = self.convb_channels[i]
            conv_block_i = nn.ModuleList()
            for j in range(repeats_i):
                block_ij = ConvNext2dBlock(
                    channels_i,
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
            # which endpoints have a projection layer ConvNext2dEndpoint
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
                        endpoint_i = ConvNext2dEndpoint(
                            self.convb_channels[i],
                            out_channels,
                            in_scale=self.resb_scales[i],
                            scale=endpoint_scale,
                            norm_layer=self._norm_layer,
                        )
                        self.endpoint_block_idx[i] = cur_endpoint
                        endpoint_blocks.append(endpoint_i)
                        cur_endpoint += 1

            self.endpoint_blocks = endpoint_blocks
            if multilayer_concat:
                self.concat_endpoint_block = ConvNext2dEndpoint(
                    in_concat_channels,
                    endpoint_channels,
                    in_scale=1,
                    scale=1,
                    activation=hid_act,
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
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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
        return (None, self.in_channels, None, None)

    def out_shape(self, in_shape=None):

        out_channels = (
            self.head_channels if self.head_channels > 0 else self.convb_channels[-1]
        )
        if in_shape is None:
            return (None, out_channels, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        return (in_shape[0], out_channels, H, W)

    @staticmethod
    def _update_mask(
        x: torch.Tensor, x_lengths: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ):
        if x_lengths is None:
            return None

        if x_mask is not None and x.size(-1) == x_mask.size(-1):
            return x_mask

        return seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=3)

    @staticmethod
    def _match_lens(endpoints):
        lens = [e.shape[-1] for e in endpoints]
        min_len = min(lens)
        for i in range(len(endpoints)):
            if lens[i] > min_len:
                t_start = (lens[i] - min_len) // 2
                t_end = t_start + min_len
                endpoints[i] = endpoints[i][:, :, :, t_start:t_end]

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
            x = self.head_norm(torch.mean(x, dim=(2, 3)))
            x = self.head(x)

        return x

    def get_config(self):

        head_act = self.head_act
        hid_act = self.hid_act
        config = {
            "in_channels": self.in_channels,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "short_name": self.short_name,
            "convb_repeats": self.convb_repeats,
            "convb_channels": self.convb_channels,
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
            logging.info("chaning convnext2d dropouts")
            self.change_dropouts(drop_path_rate)

    def change_dropouts(self, drop_path_rate):
        from ..layers import DropPath2d

        for module in self.modules():
            if isinstance(module, DropPath2d):
                module.p *= drop_path_rate / self.drop_path_rate

        self.drop_path_rate = drop_path_rate

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(ConvNext2dEncoder.__init__, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--in-channels", type=int, default=1, help=("input channel dimension")
        )

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
            choices=ConvNext2dShortName.choices(),
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
            default=[128, 256, 512, 1024],
            type=int,
            nargs="+",
            help=("conv-blocks channels for each stage"),
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

        # parser.add_argument(
        #     "--time-se",
        #     default=False,
        #     action="store_true",
        #     help=("squeeze-excitation pooling is done in time dimension only"),
        # )

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
