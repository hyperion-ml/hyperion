"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import ConvNext1dDownsampleBlock, ConvNext1dEndpoint
from ..layer_blocks.transformer_v2 import (
    TransformerEncoderV2StemType,
    TransformerV2AttType,
    TransformerV2FeedForwardType,
    TransformerV2NormLayerType,
    TransformerV2SelfAttBlock,
)
from ..layers import RotaryPosEncoder
from ..layers.attention_v2 import ScaledDotProdAttV2
from ..utils import scale_seq_lengths, seq_lengths_to_mask
from .net_arch import NetArch


class TransformerEncoderV2ShortName(str, Enum):
    ATTO = "atto"
    FEMTO = "femto"
    PICO = "pico"
    NANO = "nano"
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    BASE_GQA = "base_gqa"
    LARGE = "large"
    LARGE_GQA = "large_gqa"
    XLARGE = "xlarge"
    HUGE = "huge"

    @staticmethod
    def choices():
        return [o.value for o in TransformerEncoderV2ShortName]

    @staticmethod
    def to_config(short_name):
        strides = None
        ff_dim_multiplier = 4
        num_kv_heads = None
        ff_multiple_of = 256
        if short_name == TransformerEncoderV2ShortName.ATTO:
            repeats = 3 * [2]
            channels = 3 * [96]
            num_heads = 6
        elif short_name == TransformerEncoderV2ShortName.FEMTO:
            repeats = 3 * [2]
            channels = 3 * [128]
            num_heads = 4
        elif short_name == TransformerEncoderV2ShortName.PICO:
            repeats = 3 * [2]
            channels = 4 * [192]
            num_heads = 6
        elif short_name == TransformerEncoderV2ShortName.NANO:
            repeats = 4 * [2]
            channels = 4 * [256]
            num_heads = 4
        elif short_name == TransformerEncoderV2ShortName.TINY:
            repeats = 5 * [2]
            channels = 5 * [384]
            num_heads = 6
        elif short_name == TransformerEncoderV2ShortName.SMALL:
            repeats = 4 * [3]
            channels = 4 * [512]
            num_heads = 8
        elif short_name == TransformerEncoderV2ShortName.BASE:
            repeats = 4 * [3]
            channels = 4 * [768]
            num_heads = 12
        elif short_name == TransformerEncoderV2ShortName.BASE_GQA:
            repeats = 4 * [3]
            channels = 4 * [768]
            num_heads = 12
            num_kv_heads = 6
        elif short_name == TransformerEncoderV2ShortName.LARGE:
            repeats = 6 * [4]
            channels = 6 * [1024]
            num_heads = 24
            ff_dim_multiplier = 3.5
        elif short_name == TransformerEncoderV2ShortName.LARGE_GQA:
            repeats = 6 * [4]
            channels = [1024]
            num_heads = 24
            num_kv_heads = 8
            ff_dim_multiplier = 3.5
        else:
            raise ValueError(f"wrong ConvNext short name {short_name.value}")

        return (
            repeats,
            channels,
            num_heads,
            num_kv_heads,
            ff_dim_multiplier,
            ff_multiple_of,
            strides,
        )


class TransformerEncoderV2(NetArch):
    """ConvNext1d V2 1d Encoder.

    Attributes:
        in_feats: input features dimension
        stem_type: Types of stem block in [conv1d, conv2d]
        stem_hidden_channels: hidden channels of the stem's conv layers
        stem_kernel_sizes: kernels of the stem's conv layers
        stem_strides: strides of the stem's conv layers
        stem_act: activation of the stem layers
        stem_dropout_rate: dropout rate at the stem output
        short_name: short_name of the configuration for the transformer size
        att_type: type of attention layer in [sdp, torch_sdp, flash_sdp]
        encb_repeats: transformer block repeats in each encoder stage
        hidden_dims: transformer block hidden features in each encoder stage
        num_heads: num. of attention heads
        num_kv_heads: num. of key, value attention heads when using GQA
        att_dropout_rate: attention dropout rate
        att_bias: use bias in Linear layers of attention blocks
        ff_type: type of feed forward layer in [mlp, convnext]
        ff_dim_multiplier: number that multiplies the hidden dimension to get the inv. bottleneck dimension
        ff_multiple_of: the inv bottleneck dim has to be a multiple of this
        ff_kernel_sizes: kernels sizes when using convnext feed forward layer
        ff_dilations: ilations when using convnext feedforward layers
        ff_act: activation of feedforward layers
        ff_bias: use bias in Linear layers of feed forward blocks
        downb_strides: strides to be downsample feature maps before each encoder stage
        rope_theta: ROPE base theta
        rope_scale_freqs: scale ROPE frequencies when seq lenght is larger than the maximmum length of the original training sequences
        rope_update_max_seq_length: update the invernal ROPE variable that keeps track of the max seq length seen on training
        rope_original_max_seq_length: sets manually the max seq length seen in training for ROPE
        rope_scaling_factor: ROPE scaling factors
        rope_low_freq_factor: ROPE frequencies are not scaled for wavelengths < max_seq_length / self.low_freq_factor
        rope_high_freq_factor: ROPE frequencies are scaled by scaling for wavelengths > max_seq_length / self.high_freq_factor
        out_feats: features for ouptut projection, if None, no output proj is done
        drop_path_rate: drop path rate
        norm_layer: type of norm layer in [layer-norm, rms-norm]
        norm_eps: eps for layer norms
        use_cache: use cache for previous key, value states
        is_causal: attention mask is causal
        multilayer: use multilayer feature aggregation (mfa)
        multilayer_concat: use concatenation for mfa
        endpoint_channels: num. endpoint channels when using mfa
        endpoint_layers: layers to aggreagate in mfa, if None, all residual blocks are aggregated
        endpoint_scale_layer: layer number which indicates the time scale in mfa
        model_parallel: train with model parallel using fairscale tools

    """

    def __init__(
        self,
        in_feats: int,
        stem_type: TransformerEncoderV2StemType = TransformerEncoderV2StemType.CONV2D,
        stem_hidden_channels: List[int] = [64, 128],
        stem_kernel_sizes: List[int] = [5, 3],
        stem_strides: List[int] = [1, 2],
        stem_act: str = "silu",
        stem_dropout_rate: float = 0.1,
        short_name: Optional[str] = None,
        att_type: TransformerV2AttType = TransformerV2AttType.SDP,
        encb_repeats: List[int] = 4 * [3],
        hidden_dims: List[int] = 4 * [768],
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        ff_type: TransformerV2FeedForwardType = TransformerV2FeedForwardType.MLP,
        ff_dim_multiplier: int = 4,
        ff_multiple_of: int = 256,
        ff_kernel_sizes: List[int] = [7],
        ff_dilations: List[int] = [1],
        ff_act: str = "silu",
        ff_bias: bool = False,
        downb_strides: List[int] = [2],
        rope_theta: float = 50000,
        rope_scale_freqs: bool = True,
        rope_update_max_seq_length: bool = True,
        rope_original_max_seq_length: Optional[int] = None,
        rope_scaling_factor: float = 8,
        rope_low_freq_factor: float = 1,
        rope_high_freq_factor: float = 4,
        out_feats: Optional[int] = None,
        drop_path_rate: float = 0.0,
        norm_layer: TransformerV2NormLayerType = TransformerV2NormLayerType.LAYERNORM,
        norm_eps: float = 1e-5,
        use_cache: bool = False,
        is_causal: bool = False,
        multilayer: bool = False,
        multilayer_concat: bool = False,
        endpoint_channels: Optional[int] = None,
        endpoint_layers: Optional[List[int]] = None,
        endpoint_scale_layer: int = -1,
        model_parallel: bool = False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.stem_type = stem_type

        num_stem_layers = len(stem_hidden_channels)
        self.stem_hidden_channels = stem_hidden_channels
        assert num_stem_layers == len(stem_kernel_sizes)
        assert num_stem_layers == len(stem_strides)
        self.stem_kernel_sizes = stem_kernel_sizes
        self.stem_strides = stem_strides
        self.stem_act = stem_act
        self.stem_dropout_rate = stem_dropout_rate

        self.short_name = short_name
        if short_name is not None:
            (
                encb_repeats,
                hidden_dims,
                num_heads,
                num_kv_heads,
                ff_dim_multiplier,
                ff_multiple_of,
                downb_strides,
            ) = TransformerEncoderV2ShortName.to_config(short_name)

        num_superblocks = len(encb_repeats)
        self.num_superblocks = num_superblocks
        self.encb_repeats = encb_repeats
        self.hidden_dims = self._standarize_resblocks_param(
            hidden_dims, num_superblocks, "hidden_dims"
        )
        self.ff_kernel_sizes = self._standarize_resblocks_param(
            ff_kernel_sizes, num_superblocks, "ff_kernel_sizes"
        )
        self.ff_dilations = self._standarize_resblocks_param(
            ff_dilations, num_superblocks, "ff_dilations"
        )
        self.downb_strides = self._standarize_resblocks_param(
            downb_strides, num_superblocks - 1, "downb_strides"
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.att_type = att_type
        self.att_dropout_rate = att_dropout_rate
        self.att_bias = att_bias

        self.ff_type = ff_type
        self.ff_dim_multiplier = ff_dim_multiplier
        self.ff_multiple_of = ff_multiple_of
        self.ff_act = ff_act
        self.ff_bias = ff_bias

        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.norm_eps = norm_eps
        self._norm_layer = TransformerV2NormLayerType.to_class(norm_layer)

        self.use_cache = use_cache
        self.is_causal = is_causal

        self.rope_theta = rope_theta
        self.rope_scale_freqs = rope_scale_freqs
        self.rope_update_max_seq_length = rope_update_max_seq_length
        self.rope_original_max_seq_length = rope_original_max_seq_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope = RotaryPosEncoder(
            theta=rope_theta,
            scale_freqs=rope_scale_freqs,
            pdate_max_seq_length=rope_update_max_seq_length,
            original_max_seq_length=rope_original_max_seq_length,
            scaling_factor=rope_scaling_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
        )

        # stem block
        stem_class = TransformerEncoderV2StemType.to_config(self.stem_type)
        stem_block = stem_class(
            in_feats,
            self.hidden_dims[0],
            self.stem_hidden_channels,
            self.stem_kernel_sizes,
            self.stem_strides,
            activation=self.stem_act,
            norm_layer=self._norm_layer,
            norm_eps=self.norm_eps,
            dropout_rate=stem_dropout_rate,
        )

        self._context = stem_block.context
        self._downsample_factor = stem_block.downsample_factor
        self.stem_block = stem_block

        # downsample blocks
        self.downsample_blocks = nn.ModuleList([nn.Identity()])
        self.convb_scales = [self._downsample_factor]
        for i in range(num_superblocks - 1):
            stride_i = self.downb_strides[i]
            if stride_i > 1 or self.hidden_dims[i] != self.hidden_dims[i + 1]:
                block_i = ConvNext1dDownsampleBlock(
                    self.hidden_dims[i],
                    self.hidden_dims[i + 1],
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
            x.item() for x in torch.linspace(0, drop_path_rate, sum(encb_repeats))
        ]
        self.trans_blocks = nn.ModuleList()
        count = 0
        for i in range(num_superblocks):
            repeats_i = self.encb_repeats[i]
            hidden_dim_i = self.hidden_dims[i]
            ff_kernel_size_i = self.ff_kernel_sizes[i]
            ff_dilation_i = self.ff_dilations[i]
            trans_block_i = nn.ModuleList()
            for j in range(repeats_i):
                block_ij = TransformerV2SelfAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim_i,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    att_dropout_rate=att_dropout_rate,
                    ff_intermediate_feats=hidden_dim_i * self.ff_dim_multiplier,
                    ff_kernel_size=ff_kernel_size_i,
                    ff_dilation=ff_dilation_i,
                    ff_activation=self.ff_act,
                    ff_bias=self.ff_bias,
                    ff_multiple_of=self.ff_multiple_of,
                    att_dropout_rate=self.att_dropout_rate,
                    att_bias=self.att_bias,
                    rope=self.rope,
                    is_causal=self.is_causal,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    use_cache=self.use_cache,
                    # max_batch_size=self.max_batch_size,
                    # max_seq_length=max_seq_length,
                    drop_path_rate=drop_rates[count],
                    model_parallel=model_parallel,
                )
                count += 1
                trans_block_i.append(block_ij)
            self.trans_blocks.append(trans_block_i)

        # code for multilayer aggregation
        if multilayer:
            if endpoint_layers is None:
                # if is None all layers are endpoints
                endpoint_layers = [i + 1 for i in range(num_superblocks)]

            if endpoint_channels is None:
                # if None, the number of endpoint channels matches the one of the endpoint level
                endpoint_channels = self.hidden_dims[endpoint_scale_layer]

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
                        out_channels = self.hidden_dims[i]
                        if self.convb_scales[i] != endpoint_scale:
                            self.has_endpoint_block[i] = True

                        in_concat_channels += out_channels
                    else:
                        self.has_endpoint_block[i] = True
                        out_channels = endpoint_channels

                    if self.has_endpoint_block[i]:
                        endpoint_i = ConvNext1dEndpoint(
                            self.hidden_dims[i],
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
            endpoint_channels = self.hidden_dims[-1]

        self.multilayer = multilayer
        self.multilayer_concat = multilayer_concat
        self.endpoint_channels = endpoint_channels
        self.endpoint_layers = endpoint_layers
        self.endpoint_scale_layer = endpoint_scale_layer

        # head feature block
        self.out_norm = self._norm_layer(hidden_dims[-1], eps=norm_eps)
        if out_feats is not None and out_feats > 0:
            self.out_feats = out_feats
            if model_parallel:
                self.out_proj = ColumnParallelLinear(
                    hidden_dims[-1],
                    out_feats,
                    bias=False,
                )
            else:
                self.out_proj = nn.Linear(hidden_dims[-1], out_feats, bias=False)
        else:
            self.out_feats = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

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
        out_size = in_size
        for stride in self.stem_strides:
            out_size = int((out_size + stride - 1) // stride)

        for stride in self.downb_strides:
            out_size = int((out_size + stride - 1) // stride)

        return out_size

    def in_context(self):
        return (self._context, self._context)

    def in_shape(self):
        return (None, None, self.in_feats)

    def out_shape(self, in_shape=None):
        out_channels = self.out_feats if self.out_feats > 0 else self.endpoint_channels
        if in_shape is None:
            return (None, None, out_channels)

        assert len(in_shape) == 3
        if in_shape[2] is None:
            T = None
        else:
            T = self._compute_out_size(in_shape[2])

        return (in_shape[0], T, out_channels)

    @staticmethod
    def _update_mask(
        x: torch.Tensor, x_lengths: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ):
        if x_lengths is None:
            return None

        if x_mask is not None and x.size(-1) == x_mask.size(-1):
            return x_mask

        return seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=1)

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

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ):

        x_mask = None
        max_length = x.size(-1)
        _, x, x_lengths = self.stem_block(i)
        endpoints = []

        for i in range(self.num_superblocks):
            # downsample if needed and recalculate lengths
            # max_length = x.size(-1)
            x = self.downsample_blocks[i](x)
            stride_i = self.downb_strides[i]
            if stride_i > 1:
                # x_lengths = scale_seq_lengths(
                #     x_lengths, max_out_length=x.size(-1), max_in_length=max_length
                # )
                # x_mask = self._update_mask(x, x_lengths, x_mask)
                start_pos = start_pos // stride_i

            for j in range(self.encb_repeats[i]):
                x = self.trans_blocks[i][j](x, x_mask=x_mask, start_pos=start_pos)

            if self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)

                endpoints.append(endpoint_i)

        if self.multilayer:
            x = self._merge_endpoints(endpoints)

        x = x.contiguous()
        x = self.out_norm(x)

        if self.out_feats is not None:
            x = self.out_proj(x)

        x_lengths = scale_seq_lengths(
            x_lengths, max_out_length=x.size(-1), max_in_length=max_length
        )

        return x, x_lengths

    def get_config(self):

        config = {
            "in_feats": self.in_feats,
            "stem_type": self.stem_type,
            "stem_hidden_channels": self.stem_hidden_channels,
            "stem_kernel_sizes": self.stem_kernel_sizes,
            "stem_strides": self.stem_strides,
            "stem_act": self.stem_act,
            "stem_dropout_rate": self.stem_dropout_rate,
            "short_name": self.short_name,
            "att_type": self.att_type,
            "encb_repeats": self.encb_repeats,
            "hidden_dims": self.hidden_dims,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "att_dropout_rate": self.att_dropout_rate,
            "att_bias": self.att_bias,
            "ff_type": self.ff_type,
            "ff_dim_multiplier": self.ff_dim_multiplier,
            "ff_multiple_of": self.ff_multiple_of,
            "ff_kernel_sizes": self.ff_kernel_sizes,
            "ff_dilations": self.ff_dilations,
            "ff_act": self.ff_act,
            "ff_bias": self.ff_bias,
            "downb_strides": self.downb_strides,
            "rope_theta": self.rope_theta,
            "rope_scale_freqs": self.rope_scale_freqs,
            "rope_update_max_seq_length": self.rope_update_max_seq_length,
            "rope_original_max_seq_length": self.rope_original_max_seq_length,
            "rope_scaling_factor": self.rope_scaling_factor,
            "rope_low_freq_factor": self.rope_low_freq_factor,
            "rope_high_freq_factor": self.rope_high_freq_factor,
            "out_feats": self.out_feats,
            "norm_eps": self.norm_eps,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "use_cache": self.use_cache,
            "is_causal": self.is_causal,
            "multilayer": self.multilayer,
            "multilayer_concat": self.multilayer_concat,
            "endpoint_channels": self.endpoint_channels,
            "endpoint_layers": self.endpoint_layers,
            "endpoint_scale_layer": self.endpoint_scale_layer,
            "model_parallel": self.model_parallel,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, override_dropouts: bool, drop_path_rate: float, att_dropout_rate: float
    ):
        if override_dropouts:
            logging.info("chaning convnext1d dropouts")
            self.change_dropouts(drop_path_rate, att_dropout_rate)

    def change_dropouts(self, drop_path_rate: float, att_dropout_rate: float):
        from ..layers import DropPath1d

        for module in self.modules():
            if isinstance(module, DropPath1d):
                module.p *= drop_path_rate / self.drop_path_rate

            if isinstance(module, ScaledDotProdAttV2):
                module.dropout_rate = att_dropout_rate

        self.drop_path_rate = drop_path_rate

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(TransformerEncoderV2.__init__, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--in-feats", default=int, type=int, help="input features dimension"
        )
        parser.add_argument(
            "--stem-type",
            default=TransformerEncoderV2StemType.CONV2D.value,
            choices=TransformerEncoderV2StemType.choices(),
            help="Types of stem block in [conv1d, conv2d]",
        )
        parser.add_argument(
            "--stem-hidden-channels",
            default=[64, 128],
            type=int,
            nargs="+",
            help="hidden channels of the stem's conv layers",
        )
        parser.add_argument(
            "--stem-kernel-sizes",
            default=[5, 3],
            type=int,
            nargs="+",
            help="kernels of the stem's conv layers",
        )
        parser.add_argument(
            "--stem-strides",
            default=[1, 2],
            type=int,
            nargs="+",
            help="strides of the stem's conv layers",
        )
        parser.add_argument(
            "--stem-act", default="silu", help="activation of the stem layers"
        )
        parser.add_argument(
            "--stem-dropout-rate",
            default=0.1,
            type=float,
            help="dropout rate at the stem output",
        )
        parser.add_argument(
            "--short-name",
            default=None,
            choices=TransformerEncoderV2ShortName.choices(),
            help="short_name of the configuration for the transformer size",
        )
        parser.add_argument(
            "--att-type",
            default=TransformerV2AttType.SDP.value,
            choices=TransformerV2AttType.choices(),
            help="type of attention layer in [sdp, torch_sdp, flash_sdp]",
        )
        parser.add_argument(
            "--encb-repeats",
            default=4 * [3],
            type=int,
            nargs="+",
            help="transformer block repeats in each encoder stage",
        )
        parser.add_argument(
            "--hidden-dims",
            default=4 * [768],
            type=int,
            nargs="+",
            help="transformer block hidden features in each encoder stage",
        )
        parser.add_argument(
            "--num-heads", default=12, type=int, help="num of attention heads"
        )
        parser.add_argument(
            "--num-kv-heads",
            default=None,
            type=int,
            help="num. of key, value attention heads when using GQA",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0.0, type=float, help="attention dropout rate"
        )
        parser.add_argument(
            "--att-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of attention blocks",
        )
        parser.add_argument(
            "--ff-type",
            default=TransformerV2FeedForwardType.MLP.value,
            type=TransformerV2FeedForwardType.choices(),
            help="type of feed forward layer in [mlp, convnext]",
        )
        parser.add_argument(
            "--ff-dim-multiplier",
            default=4,
            type=int,
            help="number that multiplies the hidden dimension to get the inv. bottleneck dimension",
        )
        parser.add_argument(
            "--ff-multiple-of",
            default=256,
            type=int,
            help="the inv bottleneck dim has to be a multiple of this",
        )
        parser.add_argument(
            "--ff-kernel-sizes",
            default=[7],
            type=int,
            nargs="+",
            help="kernels sizes when using convnext feed forward layer",
        )
        parser.add_argument(
            "--ff-dilations",
            default=[1],
            type=int,
            nargs="+",
            help="dilations when using convnext feedforward layers",
        )
        parser.add_argument(
            "--ff-act", default="silu", help="activation of feedforward layers"
        )
        parser.add_argument(
            "--ff-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of feed forward blocks",
        )
        parser.add_argument(
            "--downb-strides",
            default=None,
            type=int,
            nargs="+",
            help="strides to be downsample feature maps before each encoder stage",
        )
        parser.add_argument(
            "--rope-theta", default=50000, type=float, help="ROPE base theta"
        )
        parser.add_argument(
            "--rope-scale-freqs",
            default=True,
            action=ActionYesNo,
            help="scale ROPE frequencies when seq lenght is larger than the maximmum length of the original training sequences",
        )
        parser.add_argument(
            "--rope-update-max-seq-length",
            default=True,
            action=ActionYesNo,
            help="update the invernal ROPE variable that keeps track of the max seq length seen on training",
        )
        parser.add_argument(
            "--rope-original-max-seq-length",
            default=None,
            type=int,
            help="sets manually the max seq length seen in training for ROPE",
        )
        parser.add_argument(
            "--rope-scaling-factor", default=8, type=float, help="ROPE scaling factors"
        )
        parser.add_argument(
            "--rope-low-freq-factor",
            default=1,
            type=float,
            help="ROPE frequencies are not scaled for wavelengths < max_seq_length / self.low_freq_factor",
        )
        parser.add_argument(
            "--rope-high-freq-factor",
            default=4,
            type=float,
            help="ROPE frequencies are scaled by scaling for wavelengths > max_seq_length / self.high_freq_factor",
        )
        parser.add_argument(
            "--out-feats",
            default=None,
            type=int,
            help="features for ouptut projection, if None, no output proj is done",
        )
        parser.add_argument(
            "--drop-path-rate", default=0.0, type=float, help="drop path rate"
        )
        parser.add_argument(
            "--norm-layer",
            default=TransformerV2NormLayerType.LAYERNORM.value,
            type=int,
            help="type of norm layer in [layer-norm, rms-norm]",
        )
        parser.add_argument(
            "--norm-eps", default=1e-5, type=int, help="eps for layer norms"
        )
        parser.add_argument(
            "--use-cache",
            default=False,
            action=ActionYesNo,
            help="use cache for previous key, value states",
        )
        parser.add_argument(
            "--is-causal",
            default=False,
            action=ActionYesNo,
            help="attention mask is causal",
        )
        parser.add_argument("--multilayer", default=False, action=ActionYesNo, help="")
        parser.add_argument(
            "--model-parallel",
            default=False,
            action=ActionYesNo,
            help="train with model parallel using fairscale tools",
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
        return filter_func_args(TransformerEncoderV2.change_config, kwargs)

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

        try:
            parser.add_argument(
                "--att-dropout-rate",
                default=0,
                type=float,
                help="attention layers dropout rate",
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


# def _prepare_4d_causal_attention_mask_with_cache_position(
#     attention_mask: torch.Tensor,
#     sequence_length: int,
#     target_length: int,
#     dtype: torch.dtype,
#     device: torch.device,
#     min_dtype: float,
#     cache_position: torch.Tensor,
#     batch_size: int,
# ):
#     """
#     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
#     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

#     Args:
#         attention_mask (`torch.Tensor`):
#             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
#         sequence_length (`int`):
#             The sequence length being processed.
#         target_length (`int`):
#             The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
#         dtype (`torch.dtype`):
#             The dtype to use for the 4D attention mask.
#         device (`torch.device`):
#             The device to plcae the 4D attention mask on.
#         min_dtype (`float`):
#             The minimum value representable with the dtype `dtype`.
#         cache_position (`torch.Tensor`):
#             Indices depicting the position of the input sequence tokens in the sequence.
#         batch_size (`torch.Tensor`):
#             Batch size.
#     """
#     if attention_mask is not None and attention_mask.dim() == 4:
#         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
#         causal_mask = attention_mask
#     else:
#         causal_mask = torch.full(
#             (sequence_length, target_length),
#             fill_value=min_dtype,
#             dtype=dtype,
#             device=device,
#         )
#         if sequence_length != 1:
#             causal_mask = torch.triu(causal_mask, diagonal=1)
#         causal_mask *= torch.arange(
#             target_length, device=device
#         ) > cache_position.reshape(-1, 1)
#         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
#         if attention_mask is not None:
#             causal_mask = (
#                 causal_mask.clone()
#             )  # copy to contiguous memory for in-place edit
#             mask_length = attention_mask.shape[-1]
#             padding_mask = (
#                 causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
#             )
#             padding_mask = padding_mask == 0
#             causal_mask[:, :, :, :mask_length] = causal_mask[
#                 :, :, :, :mask_length
#             ].masked_fill(padding_mask, min_dtype)

#     return causal_mask


# class Transformer(nn.Module):
#     def __init__(self, params: ModelArgs):
#         super().__init__()
#         self.params = params
#         self.vocab_size = params.vocab_size
#         self.n_layers = params.n_layers

#         self.tok_embeddings = VocabParallelEmbedding(
#             params.vocab_size, params.dim, init_method=lambda x: x
#         )

#         self.layers = torch.nn.ModuleList()
#         for layer_id in range(params.n_layers):
#             self.layers.append(TransformerBlock(layer_id, params))

#         self.norm = RMSNorm(params.dim, eps=params.norm_eps)
#         self.output = ColumnParallelLinear(
#             params.dim, params.vocab_size, bias=False, init_method=lambda x: x
#         )

#         self.freqs_cis = precompute_freqs_cis(
#             params.dim // params.n_heads,
#             params.max_seq_len * 2,
#             params.rope_theta,
#         )

#     def forward(self, tokens: torch.Tensor, start_pos: int):
#         _bsz, seqlen = tokens.shape
#         h = self.tok_embeddings(tokens)
#         self.freqs_cis = self.freqs_cis.to(h.device)
#         freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

#         mask = None
#         if seqlen > 1:
#             mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

#             mask = torch.triu(mask, diagonal=1)

#             # When performing key-value caching, we compute the attention scores
#             # only for the new sequence. Thus, the matrix of scores is of size
#             # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
#             # j > cache_len + i, since row i corresponds to token cache_len + i.
#             mask = torch.hstack(
#                 [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
#             ).type_as(h)

#         for layer in self.layers:
#             h = layer(h, start_pos, freqs_cis, mask)
#         h = self.norm(h)
#         output = self.output(h).float()
#         return output


# class LlamaPreTrainedModel(PreTrainedModel):
#     config_class = LlamaConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["LlamaDecoderLayer"]
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_cache_class = True
#     _supports_quantized_cache = True
#     _supports_static_cache = True

#     def _init_weights(self, module):
#         std = self.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()


# class LlamaModel(LlamaPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

#     Args:
#         config: LlamaConfig
#     """

#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(
#             config.vocab_size, config.hidden_size, self.padding_idx
#         )
#         self.layers = nn.ModuleList(
#             [
#                 LlamaDecoderLayer(config, layer_idx)
#                 for layer_idx in range(config.num_hidden_layers)
#             ]
#         )
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = LlamaRotaryEmbedding(config=config)
#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError(
#                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#             )

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         return_legacy_cache = False
#         if (
#             use_cache and not isinstance(past_key_values, Cache) and not self.training
#         ):  # kept for BC (non `Cache` `past_key_values` inputs)
#             return_legacy_cache = True
#             past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#             logger.warning_once(
#                 "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
#                 "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
#             )

#         if cache_position is None:
#             past_seen_tokens = (
#                 past_key_values.get_seq_length() if past_key_values is not None else 0
#             )
#             cache_position = torch.arange(
#                 past_seen_tokens,
#                 past_seen_tokens + inputs_embeds.shape[1],
#                 device=inputs_embeds.device,
#             )
#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         causal_mask = self._update_causal_mask(
#             attention_mask,
#             inputs_embeds,
#             cache_position,
#             past_key_values,
#             output_attentions,
#         )
#         hidden_states = inputs_embeds

#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                     position_embeddings,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                     position_embeddings=position_embeddings,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if return_legacy_cache:
#             next_cache = next_cache.to_legacy_cache()

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
#                 if v is not None
#             )
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

#     def _update_causal_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_tensor: torch.Tensor,
#         cache_position: torch.Tensor,
#         past_key_values: Cache,
#         output_attentions: bool,
#     ):
#         if self.config._attn_implementation == "flash_attention_2":
#             if attention_mask is not None and 0.0 in attention_mask:
#                 return attention_mask
#             return None

#         # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
#         # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
#         # to infer the attention mask.
#         past_seen_tokens = (
#             past_key_values.get_seq_length() if past_key_values is not None else 0
#         )
#         using_static_cache = isinstance(past_key_values, StaticCache)

#         # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
#         if (
#             self.config._attn_implementation == "sdpa"
#             and not using_static_cache
#             and not output_attentions
#         ):
#             if AttentionMaskConverter._ignore_causal_mask_sdpa(
#                 attention_mask,
#                 inputs_embeds=input_tensor,
#                 past_key_values_length=past_seen_tokens,
#                 is_training=self.training,
#             ):
#                 return None

#         dtype, device = input_tensor.dtype, input_tensor.device
#         min_dtype = torch.finfo(dtype).min
#         sequence_length = input_tensor.shape[1]
#         if using_static_cache:
#             target_length = past_key_values.get_max_length()
#         else:
#             target_length = (
#                 attention_mask.shape[-1]
#                 if isinstance(attention_mask, torch.Tensor)
#                 else past_seen_tokens + sequence_length + 1
#             )

#         # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
#         causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#             attention_mask,
#             sequence_length=sequence_length,
#             target_length=target_length,
#             dtype=dtype,
#             device=device,
#             min_dtype=min_dtype,
#             cache_position=cache_position,
#             batch_size=input_tensor.shape[0],
#         )

#         if (
#             self.config._attn_implementation == "sdpa"
#             and attention_mask is not None
#             and attention_mask.device.type == "cuda"
#             and not output_attentions
#         ):
#             # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
#             # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
#             # Details: https://github.com/pytorch/pytorch/issues/110213
#             causal_mask = AttentionMaskConverter._unmask_unattended(
#                 causal_mask, min_dtype
#             )

#         return causal_mask
