"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.hyper_dataclass import HyperDataClass
from ...utils.misc import filter_func_args
from ..layer_blocks.transformer_v2 import (
    SDPBackendType,
    TransformerEncoderV2StemType,
    TransformerEncoderV2StreamingConv1dStemBlock,
    TransformerV2AttType,
    TransformerV2ConvDownsampleBlock,
    TransformerV2ConvEndpoint,
    TransformerV2FeedForwardType,
    TransformerV2NormLayerType,
    TransformerV2SelfAttBlock,
    TransformerV2StreamingConvDownsampleBlock,
)
from ..layers import RotaryPosEncoder
from ..layers.attention_v2 import ScaledDotProdAttV2, TorchScaledDotProdAttV2
from ..layers.tensor_parallel import ColumnParallelLinear
from ..utils import seq_lengths_to_mask
from .net_arch import NetArch


@dataclass
class TransformerBlockState(HyperDataClass):
    """Cache container for an individual transformer block.

    Attributes:
        self_att: Dictionary with cached key/value tensors produced by the self-attention layer.
        cross_att: Optional dictionary with cached key/value tensors for the cross-attention branch.
    """

    self_att: Optional[Dict[str, torch.Tensor]] = None
    cross_att: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class TransformerEncoderState(HyperDataClass):
    """Aggregated cache state for `TransformerEncoderV2`.

    Attributes:
        block_states: List of per-block cache states mirroring the encoder layout.
        stem_state: Per-convolution states for a causal 1-D stem, or None.
        downsample_states: Per-stage causal downsampling states; identity stages use None.
    """

    block_states: List[TransformerBlockState] = field(default_factory=list)
    stem_state: Optional[List[Dict[str, Any]]] = None
    downsample_states: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of block cache entries.

        Returns:
            int: Number of cached transformer blocks.
        """
        return len(self.block_states)


class TransformerEncoderV2ShortName(str, Enum):
    """Preset short names for common `TransformerEncoderV2` configurations.

    Attributes:
        ATTO: Smallest preset.
        FEMTO: Very small preset.
        PICO: Small preset.
        NANO: Small preset with more stages.
        TINY: Compact preset.
        SMALL: Mid-sized preset.
        BASE: Base-size preset.
        BASE_GQA: Base-size preset using grouped-query attention.
        LARGE: Large preset.
        LARGE_GQA: Large preset using grouped-query attention.
        XLARGE: Extra-large preset.
        HUGE: Largest preset in this file.
    """

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
    XLARGE_GQA = "xlarge_gqa"
    HUGE = "huge"
    HUGE_GQA = "huge_gqa"

    @staticmethod
    def choices() -> List[str]:
        """Return every available short-name preset.

        Returns:
            List[str]: String values accepted by the configuration parser.
        """
        return [o.value for o in TransformerEncoderV2ShortName]

    @staticmethod
    def to_config(
        short_name: "TransformerEncoderV2ShortName",
    ) -> Tuple[
        List[int],
        List[int],
        int,
        Optional[int],
        float,
        int,
        List[int],
    ]:
        """Map a short-name preset to canonical transformer hyper-parameters.

        Args:
            short_name: Preset name to translate.

        Returns:
            Tuple[List[int], List[int], int, Optional[int], float, int, List[int]]:
            Transformer stage repeats, hidden dimensions, number of heads,
            optional key/value heads, feed-forward multiplier, feed-forward
            rounding multiple, and downsample strides.
        """
        strides = [1]
        ff_dim_multiplier = 4
        num_kv_heads = None
        ff_multiple_of = 256
        if short_name == TransformerEncoderV2ShortName.ATTO:
            repeats = 2 * [2]
            channels = 2 * [384]
            num_heads = 6
        elif short_name == TransformerEncoderV2ShortName.FEMTO:
            repeats = 3 * [2]
            channels = 3 * [384]
            num_heads = 6
        elif short_name == TransformerEncoderV2ShortName.PICO:
            repeats = 2 * [2]
            channels = 2 * [512]
            num_heads = 8
        elif short_name == TransformerEncoderV2ShortName.NANO:
            repeats = 3 * [2]
            channels = 3 * [512]
            num_heads = 8
        elif short_name == TransformerEncoderV2ShortName.TINY:
            repeats = 4 * [2]
            channels = 4 * [512]
            num_heads = 8
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
            num_kv_heads = 4
        elif short_name == TransformerEncoderV2ShortName.LARGE:
            repeats = 6 * [4]
            channels = 6 * [1024]
            num_heads = 16
            ff_dim_multiplier = 3.5
        elif short_name == TransformerEncoderV2ShortName.LARGE_GQA:
            repeats = 6 * [4]
            channels = 6 * [1024]
            num_heads = 16
            num_kv_heads = 4
            ff_dim_multiplier = 3.5
        elif short_name == TransformerEncoderV2ShortName.XLARGE:
            repeats = 8 * [4]
            channels = 8 * [1280]
            num_heads = 20
            ff_dim_multiplier = 3.5
        elif short_name == TransformerEncoderV2ShortName.XLARGE_GQA:
            repeats = 8 * [4]
            channels = 8 * [1280]
            num_heads = 20
            num_kv_heads = 5
            ff_dim_multiplier = 3
        elif short_name == TransformerEncoderV2ShortName.HUGE:
            repeats = 8 * [5]
            channels = 8 * [1536]
            num_heads = 24
            ff_dim_multiplier = 2.7
        elif short_name == TransformerEncoderV2ShortName.HUGE_GQA:
            repeats = 8 * [5]
            channels = 8 * [1536]
            num_heads = 24
            num_kv_heads = 6
            ff_dim_multiplier = 2.7
        else:
            raise ValueError(f"wrong TransformerEncoderV2 short name {short_name}")

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
    """Hyperion Transformer encoder loosely inspired by LLaMA-3.

    Attributes:
        in_feats (int): Dimensionality of the input features.
        stem_type (TransformerEncoderV2StemType): Stem block variant used to downsample inputs.
        stem_hidden_channels (List[int]): Channel widths for the stem convolutions.
        stem_kernel_sizes (List[int]): Kernel sizes for the stem convolutions.
        stem_strides (List[int]): Strides applied by the stem convolutions.
        stem_act (str): Activation function used in the stem block.
        stem_dropout_rate (float): Dropout applied after the stem block.
        short_name (Optional[str]): Optional preset identifier overriding several hyper-parameters.
        att_type (TransformerV2AttType): Attention kernel implementation.
        encb_repeats (List[int]): Number of transformer layers per encoder stage.
        hidden_dims (List[int]): Transformer hidden sizes per stage.
        num_heads (int): Number of attention heads for the main stream.
        num_kv_heads (Optional[int]): Number of key/value heads when using grouped-query attention.
        att_dropout_rate (float): Dropout applied to attention weights.
        att_bias (bool): Whether attention projections include biases.
        local_attention_sliding_window (Optional[int]): Local attention window in stage tokens; None is unrestricted.
        global_attention_sliding_window (Optional[int]): Global attention window in stage tokens; None is unrestricted.
        num_kv_shared_layers (List[int]): Number of shared suffix layers in each superblock.
        kv_source_layers (List[Optional[int]]): Flat source indices for consumers; None for independent layers.
        layer_types (List[str]): Derived local/global schedule in execution order.
        local_rope (nn.ModuleList): Local positional encoders with independent stage caches.
        global_rope (nn.ModuleList): Global positional encoders with independent stage caches.
        local_head_dim (Optional[int]): Local head width; None derives the width at each stage.
        global_k_eq_v (bool): Reuse the raw key projection for values in global layers only.
        global_head_dim (Optional[int]): Global head width; None derives the width at each stage.
        local_to_global_ratio (int): Local layers per global layer across stages; 0 means all global. The final layer is global.
        local_rope_theta (float): Local RoPE frequency base.
        global_rope_theta (float): Global RoPE frequency base.
        local_rope_partial_rotary_factor (float): Fraction of local head dimensions rotated using full-head frequency spacing.
        global_rope_partial_rotary_factor (float): Fraction of global head dimensions rotated using full-head frequency spacing.
        local_rope_scale_freqs (bool): Apply wavelength-based frequency scaling to local RoPE.
        global_rope_scale_freqs (bool): Apply wavelength-based frequency scaling to global RoPE.
        enable_v_norm (bool): Enable per-head value RMSNorm without learned scaling.
        enable_qk_norm (bool): Enable per-head Q/K RMSNorm before RoPE and unit attention scaling.
        ff_type (TransformerV2FeedForwardType): Feed-forward module implementation.
        ff_dim_multiplier (float): Factor multiplying hidden_dim to obtain the dense feed-forward width, including g4moe.
        ff_multiple_of (int): Rounds the feed-forward width up to this multiple.
        ff_num_experts: Total routed experts when ff_type is g4moe.
        ff_top_k_experts: Experts selected per token when ff_type is g4moe.
        ff_moe_intermediate_dim: Intermediate width per routed expert before rounding by ff_multiple_of.
        ff_kernel_sizes (List[int]): Kernel sizes used by convolutional feed-forward modules.
        ff_dilations (List[int]): Dilations used by convolutional feed-forward modules.
        ff_act (str): Activation function used inside feed-forward blocks.
        ff_bias (bool): Whether feed-forward projections include biases.
        downb_strides (List[int]): Strides applied by the inter-stage downsampling blocks.
        rope_update_max_seq_length (bool): Whether to update the cached RoPE maximum sequence length.
        rope_original_max_seq_length (Optional[int]): Original RoPE context length override; None uses the positional encoder default.
        rope_scaling_factor (float): Global scaling applied to RoPE frequencies.
        rope_low_freq_factor (float): Long-wavelength threshold factor for full frequency scaling.
        rope_high_freq_factor (float): Short-wavelength threshold factor for unchanged frequencies.
        out_feats (Optional[int]): Output projection size; when ``None`` the projection is skipped.
        drop_path_rate (float): Stochastic depth rate across transformer layers.
        norm_layer (TransformerV2NormLayerType): Normalization layer family used throughout the encoder.
        norm_eps (float): Epsilon passed to normalization layers.
        pre_post_norm: Add branch post-norms before residual addition using norm_layer.
        is_causal (bool): Enables causal attention and streaming 1-D convolutions;
            Conv2D stems and ConvNeXt feed-forward blocks are unsupported.
        flash_attention_version: Process-wide native Torch Flash Attention version (2, 3, or 4).
        sdp_backend (SDPBackendType): Preferred backend for PyTorch scaled dot-product attention.
        multilayer (bool): Whether to enable multi-layer feature aggregation (MFA).
        multilayer_concat (bool): Whether MFA concatenates features instead of summing them.
        endpoint_channels (Optional[int]): Target channel size for MFA endpoints.
        endpoint_layers (Optional[List[int]]): Zero-based indices of encoder stages used as MFA endpoints.
        endpoint_scale_layer (int): Stage index defining the temporal scale for MFA.
        model_parallel (bool): Enables built-in tensor-parallel projections.
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
        att_type: TransformerV2AttType = TransformerV2AttType.TORCH_SDP,
        encb_repeats: List[int] = 4 * [3],
        hidden_dims: List[int] = 4 * [768],
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        enable_qk_norm: bool = False,
        enable_v_norm: bool = False,
        local_attention_sliding_window: Optional[int] = None,
        global_attention_sliding_window: Optional[int] = None,
        local_to_global_ratio: int = 0,
        local_head_dim: Optional[int] = None,
        global_head_dim: Optional[int] = None,
        global_k_eq_v: bool = False,
        num_kv_shared_layers: Union[int, List[int]] = 0,
        ff_type: TransformerV2FeedForwardType = TransformerV2FeedForwardType.MLP,
        ff_dim_multiplier: float = 4,
        ff_multiple_of: int = 256,
        ff_num_experts: Optional[int] = None,
        ff_top_k_experts: Optional[int] = None,
        ff_moe_intermediate_dim: Optional[int] = None,
        ff_kernel_sizes: List[int] = [7],
        ff_dilations: List[int] = [1],
        ff_act: str = "silu",
        ff_bias: bool = False,
        downb_strides: List[int] = [1],
        local_rope_theta: float = 10000.0,
        global_rope_theta: float = 1000000.0,
        local_rope_partial_rotary_factor: float = 1.0,
        global_rope_partial_rotary_factor: float = 1.0,
        local_rope_scale_freqs: bool = True,
        global_rope_scale_freqs: bool = True,
        rope_update_max_seq_length: bool = True,
        rope_original_max_seq_length: Optional[int] = None,
        rope_scaling_factor: float = 8,
        rope_low_freq_factor: float = 1,
        rope_high_freq_factor: float = 4,
        out_feats: Optional[int] = None,
        drop_path_rate: float = 0.0,
        norm_layer: TransformerV2NormLayerType = TransformerV2NormLayerType.LAYERNORM,
        norm_eps: float = 1e-5,
        pre_post_norm: bool = False,
        is_causal: bool = False,
        sdp_backend: SDPBackendType = SDPBackendType.default(),
        flash_attention_version: int = 2,
        multilayer: bool = False,
        multilayer_concat: bool = False,
        endpoint_channels: Optional[int] = None,
        endpoint_layers: Optional[List[int]] = None,
        endpoint_scale_layer: int = -1,
        model_parallel: bool = False,
    ) -> None:
        """Instantiate a Transformer encoder with optional multi-scale aggregation.

        Args:
            in_feats (int): Dimensionality of the incoming feature frames.
            stem_type (TransformerEncoderV2StemType, optional): Stem implementation to use. Defaults to ``CONV2D``.
            stem_hidden_channels (List[int], optional): Channel widths for the stem convolutions. Defaults to ``[64, 128]``.
            stem_kernel_sizes (List[int], optional): Kernel sizes for the stem convolutions. Defaults to ``[5, 3]``.
            stem_strides (List[int], optional): Strides for the stem convolutions. Defaults to ``[1, 2]``.
            stem_act (str, optional): Activation applied inside the stem block. Defaults to ``"silu"``.
            stem_dropout_rate (float, optional): Dropout probability applied after the stem. Defaults to ``0.1``.
            short_name (Optional[str], optional): Optional preset identifier overriding key hyper-parameters. Defaults to ``None``.
            att_type (TransformerV2AttType, optional): Attention implementation to instantiate. Defaults to ``TORCH_SDP``.
            encb_repeats (List[int], optional): Transformer layer counts per encoder stage. Defaults to ``[3, 3, 3, 3]``.
            hidden_dims (List[int], optional): Hidden sizes per encoder stage. Defaults to ``[768, 768, 768, 768]``.
            num_heads (int, optional): Number of attention heads. Defaults to ``12``.
            num_kv_heads (Optional[int], optional): Number of key/value heads for grouped-query attention. Defaults to ``None``.
            att_dropout_rate (float, optional): Attention dropout probability. Defaults to ``0.0``.
            att_bias (bool, optional): Whether attention projections include biases. Defaults to ``False``.
            local_attention_sliding_window (Optional[int], optional): Local attention window in stage tokens; None is unrestricted. Defaults to ``None``.
            global_attention_sliding_window (Optional[int], optional): Global attention window in stage tokens; None is unrestricted. Defaults to ``None``.
            local_head_dim (Optional[int], optional): Local head width; None derives hidden_dims[i] / num_heads. Defaults to ``None``.
            num_kv_shared_layers: Shared suffix length per superblock; an integer applies
                to every superblock, or a list supplies one count per superblock. Defaults to 0.
                Each shared attention type needs an earlier non-shared source in its superblock.
            global_k_eq_v (bool, optional): Reuse the raw key projection for V in global layers; independent of value normalization. Defaults to ``False``.
            global_head_dim (Optional[int], optional): Global head width; None derives hidden_dims[i] / num_heads. Defaults to ``None``.
            local_to_global_ratio (int, optional): Local layers per global layer across stages; 0 means all global. The final layer is global. Defaults to ``0``.
            local_rope_theta (float, optional): Local RoPE frequency base. Defaults to ``10000.0``.
            global_rope_theta (float, optional): Global RoPE frequency base. Defaults to ``1000000.0``.
            local_rope_partial_rotary_factor (float, optional): Fraction of local head dimensions rotated using full-head frequency spacing. Defaults to ``1.0``.
            global_rope_partial_rotary_factor (float, optional): Fraction of global head dimensions rotated using full-head frequency spacing. Defaults to ``1.0``.
            local_rope_scale_freqs (bool, optional): Apply wavelength-based frequency scaling to local RoPE. Defaults to ``True``.
            global_rope_scale_freqs (bool, optional): Apply wavelength-based frequency scaling to global RoPE. Defaults to ``True``.
            enable_v_norm (bool, optional): Enable per-head value RMSNorm without learned scaling. Defaults to ``False``.
            enable_qk_norm (bool, optional): Enable per-head Q/K RMSNorm and unit attention scaling. Defaults to ``False``.
            ff_type (TransformerV2FeedForwardType, optional): Feed-forward module implementation. Defaults to ``MLP``.
            ff_dim_multiplier (float, optional): Scales ``hidden_dim`` to obtain the feed-forward width. Defaults to ``4``.
            ff_multiple_of (int, optional): Rounds the feed-forward width to a multiple. Defaults to ``256``.
            ff_num_experts: Positive number of routed experts, required for g4moe.
            ff_top_k_experts: Experts selected per token in [1, ff_num_experts], required for g4moe.
            ff_moe_intermediate_dim: Positive expert width before rounding by ff_multiple_of, required for g4moe.
            ff_kernel_sizes (List[int], optional): Kernel sizes for convolutional feed-forward blocks. Defaults to ``[7]``.
            ff_dilations (List[int], optional): Dilations for convolutional feed-forward blocks. Defaults to ``[1]``.
            ff_act (str, optional): Activation applied inside feed-forward modules. Defaults to ``"silu"``.
            ff_bias (bool, optional): Whether feed-forward projections include biases. Defaults to ``False``.
            downb_strides (List[int], optional): Strides for inter-stage downsampling blocks. Defaults to ``[1]``.
            rope_update_max_seq_length (bool, optional): Whether to update the cached RoPE maximum sequence length. Defaults to ``True``.
            rope_original_max_seq_length (Optional[int], optional): Original RoPE context length override; None uses the positional encoder default. Defaults to ``None``.
            rope_scaling_factor (float, optional): Global scaling factor for RoPE. Defaults to ``8``.
            rope_low_freq_factor (float, optional): Wavelength threshold exempt from RoPE scaling. Defaults to ``1``.
            rope_high_freq_factor (float, optional): Wavelength threshold subject to full RoPE scaling. Defaults to ``4``.
            out_feats (Optional[int], optional): Output projection size; if ``None`` the projection is skipped. Defaults to ``None``.
            drop_path_rate (float, optional): Stochastic depth rate across transformer layers. Defaults to ``0.0``.
            norm_layer (TransformerV2NormLayerType, optional): Normalization layer family. Defaults to ``LAYERNORM``.
            norm_eps (float, optional): Epsilon used in normalization layers. Defaults to ``1e-5``.
            pre_post_norm: Add branch post-norms before residual addition using norm_layer. Defaults to False.
            is_causal (bool, optional): Whether to use causal attention and 1-D convolutions. Conv2D stems and
                ConvNeXt feed-forward blocks are rejected. Defaults to ``False``.
            flash_attention_version: Process-wide native Torch Flash Attention version; defaults to 2.
            sdp_backend (SDPBackendType, optional): Preferred PyTorch scaled dot-product backend. Defaults to ``SDPBackendType.default()``.
            multilayer (bool, optional): Enables multi-layer feature aggregation (MFA). Defaults to ``False``.
            multilayer_concat (bool, optional): If ``True``, MFA concatenates endpoints before projection. Defaults to ``False``.
            endpoint_channels (Optional[int], optional): Target channel width for MFA endpoints. Defaults to ``None``.
            endpoint_layers (Optional[List[int]], optional): Zero-based stage indices exported as MFA endpoints. Defaults to ``None``.
            endpoint_scale_layer (int, optional): Stage index defining the temporal resolution for MFA. Defaults to ``-1``.
            model_parallel (bool, optional): Enables FairScale tensor model-parallel linear layers. Defaults to ``False``.
        """
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
        self.enable_qk_norm = enable_qk_norm
        self.enable_v_norm = enable_v_norm

        self.ff_type = ff_type
        self.ff_dim_multiplier = ff_dim_multiplier
        self.ff_multiple_of = ff_multiple_of
        self.ff_num_experts = ff_num_experts
        self.ff_top_k_experts = ff_top_k_experts
        self.ff_moe_intermediate_dim = ff_moe_intermediate_dim
        self.ff_act = ff_act
        self.ff_bias = ff_bias

        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.norm_eps = norm_eps
        self.pre_post_norm = pre_post_norm
        self._norm_layer = TransformerV2NormLayerType.to_class(norm_layer)

        self.is_causal = is_causal
        if is_causal and stem_type != TransformerEncoderV2StemType.CONV1D:
            raise ValueError("Causal TransformerEncoderV2 requires a conv1d stem")
        if is_causal and ff_type == TransformerV2FeedForwardType.CONVNEXT:
            raise ValueError("ConvNeXt feed-forward blocks do not support causal mode")
        self.local_attention_sliding_window = local_attention_sliding_window
        self.global_attention_sliding_window = global_attention_sliding_window
        self.local_to_global_ratio = local_to_global_ratio
        self.local_head_dim = local_head_dim
        self.global_head_dim = global_head_dim
        self.global_k_eq_v = global_k_eq_v
        self.sdp_backend = sdp_backend
        self.flash_attention_version = flash_attention_version
        if self.att_type == TransformerV2AttType.TORCH_SDP:
            TorchScaledDotProdAttV2.set_flash_attention_version(flash_attention_version)

        self.local_rope_theta = local_rope_theta
        self.global_rope_theta = global_rope_theta
        self.local_rope_partial_rotary_factor = local_rope_partial_rotary_factor
        self.global_rope_partial_rotary_factor = global_rope_partial_rotary_factor
        self.local_rope_scale_freqs = local_rope_scale_freqs
        self.global_rope_scale_freqs = global_rope_scale_freqs
        self.rope_update_max_seq_length = rope_update_max_seq_length
        self.rope_original_max_seq_length = rope_original_max_seq_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        rope_kwargs: Dict[str, Any] = dict(
            update_max_seq_length=rope_update_max_seq_length,
            scaling_factor=rope_scaling_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
        )
        if rope_original_max_seq_length is not None:
            rope_kwargs["original_max_seq_length"] = rope_original_max_seq_length
        if (
            isinstance(local_to_global_ratio, bool)
            or not isinstance(local_to_global_ratio, int)
            or local_to_global_ratio < 0
        ):
            raise ValueError("local_to_global_ratio must be a nonnegative integer")
        for name, window in (
            ("local_attention_sliding_window", local_attention_sliding_window),
            ("global_attention_sliding_window", global_attention_sliding_window),
        ):
            if window is not None and (
                isinstance(window, bool) or not isinstance(window, int) or window <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            local_to_global_ratio > 0
            and local_attention_sliding_window is not None
            and global_attention_sliding_window is not None
            and global_attention_sliding_window <= local_attention_sliding_window
        ):
            raise ValueError(
                "global_attention_sliding_window must exceed the local window"
            )
        total_layers = sum(self.encb_repeats)
        self.layer_types = [
            (
                "global"
                if local_to_global_ratio == 0
                or (idx + 1) % (local_to_global_ratio + 1) == 0
                or idx == total_layers - 1
                else "local"
            )
            for idx in range(total_layers)
        ]
        counts = (
            [num_kv_shared_layers] * num_superblocks
            if isinstance(num_kv_shared_layers, int)
            else num_kv_shared_layers
        )
        if not isinstance(counts, list) or len(counts) != num_superblocks:
            raise ValueError(
                "num_kv_shared_layers must be an integer or one count per superblock"
            )
        self.num_kv_shared_layers = list(counts)
        self.kv_source_layers: List[Optional[int]] = []
        layer_idx = 0
        for stage_idx, (repeats, shared_count) in enumerate(
            zip(self.encb_repeats, counts)
        ):
            if (
                isinstance(shared_count, bool)
                or not isinstance(shared_count, int)
                or shared_count < 0
                or shared_count >= repeats
            ):
                raise ValueError(
                    f"num_kv_shared_layers[{stage_idx}] must be in [0, {repeats - 1}]"
                )
            sources: Dict[str, int] = {}
            for j in range(repeats):
                kind = self.layer_types[layer_idx]
                if j < repeats - shared_count:
                    sources[kind] = layer_idx
                    self.kv_source_layers.append(None)
                else:
                    if kind not in sources:
                        raise ValueError(
                            f"Shared {kind} layer {layer_idx} has no non-shared {kind} "
                            f"source in superblock {stage_idx}; reduce num_kv_shared_layers"
                        )
                    self.kv_source_layers.append(sources[kind])
                layer_idx += 1
        self._kv_export_layers = {
            idx for idx in self.kv_source_layers if idx is not None
        }

        windows = {
            "local": local_attention_sliding_window,
            "global": global_attention_sliding_window,
        }
        if self.att_type != TransformerV2AttType.HF_FLASH_SDP and any(
            windows[kind] is not None for kind in self.layer_types
        ):
            raise ValueError("Finite attention windows require att_type='hf_flash_sdp'")
        # Separate caches per stage avoid sharing dimensions and tracked lengths
        # between stages with different widths or temporal resolutions.
        self.local_rope = nn.ModuleList(
            [
                RotaryPosEncoder(
                    theta=local_rope_theta,
                    scale_freqs=local_rope_scale_freqs,
                    partial_rotary_factor=local_rope_partial_rotary_factor,
                    **rope_kwargs,
                )
                for _ in range(num_superblocks)
            ]
        )
        self.global_rope = nn.ModuleList(
            [
                RotaryPosEncoder(
                    theta=global_rope_theta,
                    scale_freqs=global_rope_scale_freqs,
                    partial_rotary_factor=global_rope_partial_rotary_factor,
                    **rope_kwargs,
                )
                for _ in range(num_superblocks)
            ]
        )

        # stem block
        stem_class = (
            TransformerEncoderV2StreamingConv1dStemBlock
            if self.is_causal
            else TransformerEncoderV2StemType.to_class(self.stem_type)
        )
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
                downsample_class = (
                    TransformerV2StreamingConvDownsampleBlock
                    if self.is_causal
                    else TransformerV2ConvDownsampleBlock
                )
                block_i = downsample_class(
                    self.hidden_dims[i],
                    self.hidden_dims[i + 1],
                    stride=stride_i,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
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
                layer_type = self.layer_types[count]
                block_ij = TransformerV2SelfAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim_i,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    ff_intermediate_feats=int(hidden_dim_i * self.ff_dim_multiplier),
                    ff_kernel_size=ff_kernel_size_i,
                    ff_dilation=ff_dilation_i,
                    ff_activation=self.ff_act,
                    ff_bias=self.ff_bias,
                    ff_multiple_of=self.ff_multiple_of,
                    ff_num_experts=self.ff_num_experts,
                    ff_top_k_experts=self.ff_top_k_experts,
                    ff_moe_intermediate_dim=self.ff_moe_intermediate_dim,
                    att_dropout_rate=self.att_dropout_rate,
                    att_bias=self.att_bias,
                    enable_qk_norm=self.enable_qk_norm,
                    enable_v_norm=self.enable_v_norm,
                    k_eq_v=self.global_k_eq_v and layer_type == "global",
                    shared_kv=self.kv_source_layers[count] is not None,
                    head_dim=(
                        self.local_head_dim
                        if layer_type == "local"
                        else self.global_head_dim
                    ),
                    rope=(
                        self.local_rope[i]
                        if layer_type == "local"
                        else self.global_rope[i]
                    ),
                    is_causal=self.is_causal,
                    att_sliding_window=windows[layer_type],
                    sdp_backend=self.sdp_backend,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    pre_post_norm=self.pre_post_norm,
                    drop_path_rate=drop_rates[count],
                    model_parallel=model_parallel,
                )
                count += 1
                trans_block_i.append(block_ij)
            self.trans_blocks.append(trans_block_i)

        # code for multilayer aggregation
        if multilayer:
            if (
                endpoint_scale_layer < -num_superblocks
                or endpoint_scale_layer >= num_superblocks
            ):
                raise ValueError(
                    "endpoint_scale_layer contains an invalid stage index "
                    f"{endpoint_scale_layer}; valid range is [{-num_superblocks}, {num_superblocks - 1}]"
                )

            if endpoint_layers is None:
                # if None, all encoder stages are endpoints
                endpoint_layers = list(range(num_superblocks))
            else:
                endpoint_layers = list(dict.fromkeys(endpoint_layers))
                if len(endpoint_layers) == 0:
                    raise ValueError(
                        "endpoint_layers must contain at least one stage when multilayer=True"
                    )

                invalid_layers = [
                    layer
                    for layer in endpoint_layers
                    if layer < 0 or layer >= num_superblocks
                ]
                if invalid_layers:
                    raise ValueError(
                        "endpoint_layers contains invalid 0-based stage indices "
                        f"{invalid_layers}; valid range is [0, {num_superblocks - 1}]"
                    )

            if endpoint_channels is None:
                # if None, the number of endpoint channels matches the one of the endpoint level
                endpoint_channels = self.hidden_dims[endpoint_scale_layer]

            # which layers are endpoints
            self.is_endpoint = [i in endpoint_layers for i in range(num_superblocks)]
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
                    if self.is_causal and self.convb_scales[i] != endpoint_scale:
                        raise ValueError(
                            "Causal multilayer endpoints must use the same temporal scale; "
                            "endpoint resampling does not support streaming"
                        )
                    if multilayer_concat:
                        out_channels = self.hidden_dims[i]
                        if self.convb_scales[i] != endpoint_scale:
                            self.has_endpoint_block[i] = True

                        in_concat_channels += out_channels
                    else:
                        self.has_endpoint_block[i] = True
                        out_channels = endpoint_channels

                    if self.has_endpoint_block[i]:
                        endpoint_i = TransformerV2ConvEndpoint(
                            self.hidden_dims[i],
                            out_channels,
                            in_scale=self.convb_scales[i],
                            out_scale=endpoint_scale,
                            norm_layer=self._norm_layer,
                            norm_eps=self.norm_eps,
                        )
                        self.endpoint_block_idx[i] = cur_endpoint
                        endpoint_blocks.append(endpoint_i)
                        cur_endpoint += 1

            self.endpoint_blocks = endpoint_blocks
            if multilayer_concat:
                self.concat_endpoint_block = TransformerV2ConvEndpoint(
                    in_concat_channels,
                    endpoint_channels,
                    in_scale=1,
                    out_scale=1,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                )
        else:
            endpoint_channels = self.hidden_dims[-1]

        self.multilayer = multilayer
        self.multilayer_concat = multilayer_concat
        self.endpoint_channels = endpoint_channels
        self.endpoint_layers = endpoint_layers
        self.endpoint_scale_layer = endpoint_scale_layer
        self.model_parallel = model_parallel

        # head feature block
        self.out_norm = self._norm_layer(endpoint_channels, eps=norm_eps)
        if out_feats is not None and out_feats > 0:
            self.out_feats = out_feats
            if model_parallel:
                self.out_proj = ColumnParallelLinear(
                    endpoint_channels,
                    out_feats,
                    bias=False,
                )
            else:
                self.out_proj = nn.Linear(endpoint_channels, out_feats, bias=False)
        else:
            self.out_feats = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize learnable weights with truncated normal values."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    @staticmethod
    def _standarize_resblocks_param(
        p: Union[int, List[int]], num_blocks: int, p_name: str
    ) -> List[int]:
        """Normalize a residual-block parameter to one value per block.

        Args:
            p: Scalar or list-valued parameter.
            num_blocks: Required number of values.
            p_name: Parameter name used in error messages.

        Returns:
            List[int]: Parameter expanded to ``num_blocks`` entries.
        """
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

    def _compute_out_size(self, in_size: int) -> int:
        """Compute the encoder time length after all convolutions.

        Args:
            in_size: Input time length.

        Returns:
            int: Output time length after stem and downsampling blocks.
        """
        out_size = in_size
        for stride in self.stem_strides:
            out_size = int((out_size + stride - 1) // stride)

        for stride in self.downb_strides:
            out_size = int((out_size + stride - 1) // stride)

        return out_size

    def in_context(self) -> Tuple[int, int]:
        """Return the encoder receptive-field context.

        Returns:
            Tuple[int, int]: Left/right convolution context in frames; causal encoders
            have zero right context.
        """
        return (self._context, 0 if self.is_causal else self._context)

    def in_shape(self) -> Tuple[Optional[int], Optional[int], int]:
        """Return the expected input tensor shape.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Expected shape
            ``(batch, time, features)``.
        """
        return (None, None, self.in_feats)

    def out_shape(
        self, in_shape: Optional[Tuple[Optional[int], Optional[int], int]] = None
    ) -> Tuple[Optional[int], Optional[int], int]:
        """Return the output tensor shape for a given input shape.

        Args:
            in_shape: Optional input shape ``(batch, time, features)``.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Output shape.
        """
        out_channels = (
            self.out_feats if self.out_feats is not None else self.endpoint_channels
        )
        if in_shape is None:
            return (None, None, out_channels)

        assert len(in_shape) == 3
        if in_shape[1] is None:
            T = None
        else:
            T = self._compute_out_size(in_shape[1])

        return (in_shape[0], T, out_channels)

    def _make_attention_mask(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor],
        start_pos: int,
        cache: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Combine chunk padding with the visible cache window and causality.

        Args:
            x: Current stage features shaped (batch, query_time, features).
            x_lengths: Valid lengths in the current stage chunk, shaped (batch,).
            start_pos: Absolute query/chunk start in this stage's time base.
            cache: Independent source cache before the current update, or None.

        Returns:
            Boolean (batch, key_time) padding mask for HF Flash/non-causal
            attention, or (batch, 1, query_time, key_time) combined mask for
            causal manual/Torch attention. None permits an unmasked backend
            call, including Torch's uncached causal shortcut.

        Historical keys are assumed valid for active sequences: a partially
        valid chunk must be final, finished sequences must not resume, and
        outputs with zero valid query length must be ignored.
        """
        query_length = x.size(1)
        padding_mask = seq_lengths_to_mask(x_lengths, query_length, time_dim=1)
        end_pos = start_pos + query_length
        key_offset = start_pos
        key_length = query_length
        if cache is not None:
            old_offset = int(cache["cache_offset"])
            old_end = old_offset + int(cache["cache_length"])
            # Attention sees the old history plus the entire current chunk;
            # cache capacity only limits what is retained for the next call.
            if int(cache["cache_length"]) > 0:
                key_offset = min(old_offset, start_pos)
                key_length = max(old_end, end_pos) - key_offset

        key_positions = torch.arange(key_length, device=x.device) + key_offset
        key_valid = None
        if padding_mask is not None:
            key_valid = torch.ones(
                x.size(0), key_length, dtype=torch.bool, device=x.device
            )
            current = (key_positions >= start_pos) & (key_positions < end_pos)
            chunk_indices = key_positions[current] - start_pos
            key_valid[:, current] = padding_mask[:, chunk_indices]

        if not self.is_causal or self.att_type == TransformerV2AttType.HF_FLASH_SDP:
            return key_valid
        if (
            self.att_type == TransformerV2AttType.TORCH_SDP
            and cache is None
            and padding_mask is None
        ):
            return None
        query_positions = torch.arange(query_length, device=x.device) + start_pos
        causal = key_positions[None, :] <= query_positions[:, None]
        causal = causal[None, None, :, :]
        return causal if key_valid is None else causal & key_valid[:, None, None, :]

    @staticmethod
    def _match_lens(endpoints: List[torch.Tensor]) -> List[torch.Tensor]:
        """Center-crop endpoint tensors so they share a common time length.

        Args:
            endpoints: Endpoint tensors to align.

        Returns:
            List[torch.Tensor]: Cropped endpoint tensors with equal time length.
        """
        lens = [e.shape[1] for e in endpoints]
        min_len = min(lens)
        for i in range(len(endpoints)):
            if lens[i] > min_len:
                t_start = (lens[i] - min_len) // 2
                t_end = t_start + min_len
                endpoints[i] = endpoints[i][:, t_start:t_end, :]

        return endpoints

    def _merge_endpoints(self, endpoints: List[torch.Tensor]) -> torch.Tensor:
        """Merge multi-layer endpoints into a single representation.

        Args:
            endpoints: Endpoint tensors to combine.

        Returns:
            torch.Tensor: Aggregated endpoint tensor.
        """
        endpoints = self._match_lens(endpoints)
        if self.multilayer_concat:
            try:
                x = torch.cat(endpoints, dim=2)
            except Exception:
                for k in range(len(endpoints)):
                    logging.error(
                        f"cat shape error ep={k},  shape{endpoints[k].size()}"
                    )
                raise

            x = self.concat_endpoint_block(x)
        else:
            x = torch.mean(torch.stack(endpoints), 0)

        return x

    def init_state(
        self,
        batch_size: int,
        max_cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TransformerEncoderState:
        """Initialize the per-block caches used for streaming inference.

        Args:
            batch_size (int): Batch size for convolution streams and initial empty
                attention cache tensors.
            max_cache_length (int): Maximum retained history in the first stage's time base.
                Later stages scale this limit by their downsampling strides. Each layer's
                cache is additionally capped at its finite attention sliding window.
            device (Optional[torch.device]): Device on which to allocate the caches.
            dtype (Optional[torch.dtype]): Tensor dtype for the caches.

        Returns:
            TransformerEncoderState: Attention caches and, for causal encoders,
                stem and downsampling convolution states. Shared layers have
                self_att=None; source layers own their K/V cache.
        """

        block_states: List[TransformerBlockState] = []
        current_cache_length = max_cache_length

        for i, blocks in enumerate(self.trans_blocks):
            if i > 0:
                stride = self.downb_strides[i - 1]
                if stride > 1:
                    current_cache_length = max(
                        1, int(math.ceil(current_cache_length / stride))
                    )
            for block in blocks:
                block_state = (
                    None
                    if block.attention.shared_kv
                    else block.init_state(
                        batch_size=batch_size,
                        max_cache_length=current_cache_length,
                        device=device,
                        dtype=dtype,
                    )
                )
                block_states.append(TransformerBlockState(self_att=block_state))

        stem_state = None
        downsample_states = []
        if self.is_causal:
            stem_state = self.stem_block.init_state(
                batch_size, device=device, dtype=dtype
            )
            downsample_states = [
                (
                    block.init_state(batch_size, device=device, dtype=dtype)
                    if isinstance(block, TransformerV2StreamingConvDownsampleBlock)
                    else None
                )
                for block in self.downsample_blocks
            ]
        return TransformerEncoderState(
            block_states=block_states,
            stem_state=stem_state,
            downsample_states=downsample_states,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        state: Optional[TransformerEncoderState] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], TransformerEncoderState],
    ]:
        """Encode input features through the transformer stack with optional cache updates.

        Args:
            x (torch.Tensor): Input tensor shaped `(batch, time, features)`.
            x_lengths (Optional[torch.Tensor]): Valid lengths for each sequence.
            start_pos (int, optional): Global starting position in the input-frame time base;
                it is rescaled as the sequence is downsampled before cache writes.
            state (Optional[TransformerEncoderState]): Optional cache state returned by :meth:`init_state`.
                Causal encoders use convolution stream methods when state is supplied;
                without state they use full-sequence causal forwards. Streaming is for
                inference in eval mode. start_pos must count consumed input frames.

        Returns:
            Either `(output, output_lengths)` when `state` is ``None`` or
            `(output, output_lengths, new_state)` when cache updates are requested.
        """

        stem_state = None
        downsample_states = []
        if state is not None and self.is_causal:
            if (
                state.stem_state is None
                or len(state.downsample_states) != self.num_superblocks
            ):
                raise ValueError(
                    "Causal streaming requires stem and downsampling states from init_state"
                )
            _, x, x_lengths, stem_state = self.stem_block.stream(
                x, state.stem_state, x_lengths
            )
        else:
            _, x, x_lengths = self.stem_block(x, x_lengths)
        for stride in self.stem_strides:
            if stride > 1:
                start_pos = (
                    (start_pos + stride - 1) // stride
                    if self.is_causal
                    else start_pos // stride
                )
        endpoints = []
        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-stem-avg=%f", torch.mean(x))

        updated_states: List[TransformerBlockState] = []
        state_blocks = (
            state.block_states if state is not None else [None] * sum(self.encb_repeats)
        )
        block_idx = 0
        if state is not None and len(state_blocks) != sum(self.encb_repeats):
            raise ValueError("Cache state must contain one entry per transformer layer")

        for i in range(self.num_superblocks):
            prepared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            downsample_block = self.downsample_blocks[i]
            downsample_state = None
            if not isinstance(downsample_block, nn.Identity):
                if self.is_causal and state is not None:
                    downsample_state = state.downsample_states[i]
                    if downsample_state is None:
                        raise ValueError("Missing causal downsampling state")
                    x, x_lengths, downsample_state = downsample_block.stream(
                        x, downsample_state, x_lengths
                    )
                else:
                    x, x_lengths = downsample_block(x, x_lengths)
            if self.is_causal:
                downsample_states.append(downsample_state)
            if i > 0:
                stride_i = self.downb_strides[i - 1]
                if stride_i > 1:
                    start_pos = (
                        (start_pos + stride_i - 1) // stride_i
                        if self.is_causal
                        else start_pos // stride_i
                    )

            # Local/global caches may retain different histories. Build each
            # type's mask before source updates, reusing masks for equal histories.
            attention_masks: Dict[str, Optional[torch.Tensor]] = {}
            masks_by_history: Dict[
                Optional[Tuple[int, int]], Optional[torch.Tensor]
            ] = {}
            for j, block in enumerate(self.trans_blocks[i]):
                layer_type = self.layer_types[block_idx + j]
                if block.attention.shared_kv or layer_type in attention_masks:
                    continue
                cache = (
                    state_blocks[block_idx + j].self_att if state is not None else None
                )
                history = (
                    (int(cache["cache_offset"]), int(cache["cache_length"]))
                    if cache is not None
                    else None
                )
                if history not in masks_by_history:
                    masks_by_history[history] = self._make_attention_mask(
                        x, x_lengths, start_pos, cache
                    )
                attention_masks[layer_type] = masks_by_history[history]

            for j in range(self.encb_repeats[i]):
                current_state = (
                    state_blocks[block_idx].self_att if state is not None else None
                )
                block = self.trans_blocks[i][j]
                # A strided convolution may emit no frames for a short chunk.
                # Preserve attention caches until this stage emits its next frame.
                if x.size(1) == 0:
                    if state is not None:
                        updated_states.append(
                            TransformerBlockState(self_att=current_state)
                        )
                    block_idx += 1
                    continue
                source_idx = self.kv_source_layers[block_idx]
                if source_idx is not None and current_state is not None:
                    raise ValueError(
                        "Shared layers must have self_att=None in cache state"
                    )
                export_kv = block_idx in self._kv_export_layers
                att_out = block(
                    x,
                    x_mask=attention_masks[self.layer_types[block_idx]],
                    start_pos=start_pos,
                    state=current_state,
                    shared_kv=None if source_idx is None else prepared_kv[source_idx],
                    return_kv=export_kv,
                )

                if export_kv:
                    x, kv, new_block_state = att_out
                    prepared_kv[block_idx] = kv
                elif state is not None and source_idx is None:
                    x, new_block_state = att_out
                else:
                    x = att_out
                    new_block_state = None
                if state is not None:
                    updated_states.append(
                        TransformerBlockState(self_att=new_block_state)
                    )

                block_idx += 1
                if not torch.all(torch.isfinite(x)):
                    logging.warning(
                        "non-finite x-enc-%d-%d-avg=%f", i, j, torch.mean(x)
                    )

            if self.multilayer and self.is_endpoint[i]:
                endpoint_i = x
                if self.has_endpoint_block[i]:
                    idx = self.endpoint_block_idx[i]
                    endpoint_i = self.endpoint_blocks[idx](endpoint_i)

                endpoints.append(endpoint_i)

        if self.multilayer:
            x = self._merge_endpoints(endpoints)

        x = x.contiguous()
        if self.out_feats is not None:
            x = self.out_proj(self.out_norm(x))
        else:
            x = self.out_norm(x).type_as(x)

        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-out-%d-%d-avg=%f", i, j, torch.mean(x))
        if state is not None:
            return (
                x,
                x_lengths,
                TransformerEncoderState(
                    block_states=updated_states,
                    stem_state=stem_state,
                    downsample_states=downsample_states,
                ),
            )
        return x, x_lengths

    def requires_ddp_find_unused_parameters(self) -> bool:
        """Return whether expert routing requires DDP unused-parameter detection.

        Returns:
            ``True`` when the feed-forward blocks use G4MoE.
        """
        return self.ff_type == TransformerV2FeedForwardType.G4MoE

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: Whether to omit class metadata from the base config.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstructing the model.
        """

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
            "enable_qk_norm": self.enable_qk_norm,
            "enable_v_norm": self.enable_v_norm,
            "ff_type": self.ff_type,
            "ff_dim_multiplier": self.ff_dim_multiplier,
            "ff_multiple_of": self.ff_multiple_of,
            "ff_num_experts": self.ff_num_experts,
            "ff_top_k_experts": self.ff_top_k_experts,
            "ff_moe_intermediate_dim": self.ff_moe_intermediate_dim,
            "ff_kernel_sizes": self.ff_kernel_sizes,
            "ff_dilations": self.ff_dilations,
            "ff_act": self.ff_act,
            "ff_bias": self.ff_bias,
            "downb_strides": self.downb_strides,
            "local_attention_sliding_window": self.local_attention_sliding_window,
            "global_attention_sliding_window": self.global_attention_sliding_window,
            "local_to_global_ratio": self.local_to_global_ratio,
            "local_head_dim": self.local_head_dim,
            "global_head_dim": self.global_head_dim,
            "global_k_eq_v": self.global_k_eq_v,
            "num_kv_shared_layers": self.num_kv_shared_layers,
            "local_rope_theta": self.local_rope_theta,
            "global_rope_theta": self.global_rope_theta,
            "local_rope_partial_rotary_factor": self.local_rope_partial_rotary_factor,
            "global_rope_partial_rotary_factor": self.global_rope_partial_rotary_factor,
            "local_rope_scale_freqs": self.local_rope_scale_freqs,
            "global_rope_scale_freqs": self.global_rope_scale_freqs,
            "rope_update_max_seq_length": self.rope_update_max_seq_length,
            "rope_original_max_seq_length": self.rope_original_max_seq_length,
            "rope_scaling_factor": self.rope_scaling_factor,
            "rope_low_freq_factor": self.rope_low_freq_factor,
            "rope_high_freq_factor": self.rope_high_freq_factor,
            "out_feats": self.out_feats,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "norm_eps": self.norm_eps,
            "pre_post_norm": self.pre_post_norm,
            "is_causal": self.is_causal,
            "sdp_backend": self.sdp_backend,
            "flash_attention_version": self.flash_attention_version,
            "multilayer": self.multilayer,
            "multilayer_concat": self.multilayer_concat,
            "endpoint_channels": self.endpoint_channels,
            "endpoint_layers": self.endpoint_layers,
            "endpoint_scale_layer": self.endpoint_scale_layer,
            "model_parallel": self.model_parallel,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, override_dropouts: bool, drop_path_rate: float, att_dropout_rate: float
    ) -> None:
        """Update configurable dropout values in-place when requested.

        Args:
            override_dropouts: Whether the provided dropout values should be applied.
            drop_path_rate: New stochastic depth rate.
            att_dropout_rate: New attention dropout rate.
        """
        if override_dropouts:
            logging.info("changing transformer dropouts")
            self.change_dropouts(drop_path_rate, att_dropout_rate)

    def change_dropouts(self, drop_path_rate: float, att_dropout_rate: float) -> None:
        """Update stochastic-depth and attention dropout probabilities.

        Args:
            drop_path_rate: New stochastic depth rate.
            att_dropout_rate: New attention dropout rate.
        """
        from ..layers import DropPath1d

        drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.encb_repeats))
        ]
        count = 0
        for stage_blocks in self.trans_blocks:
            for block in stage_blocks:
                module_drop_rate = drop_rates[count]
                if block.drop_path is None:
                    if module_drop_rate > 0.0:
                        block.drop_path = DropPath1d(module_drop_rate)
                        block.drop_path.train(self.training)
                else:
                    block.drop_path.p = module_drop_rate
                count += 1

        for module in self.modules():
            if isinstance(module, ScaledDotProdAttV2):
                module.dropout_rate = att_dropout_rate

        self.drop_path_rate = drop_path_rate
        self.att_dropout_rate = att_dropout_rate

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by the constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by ``__init__``.
        """
        return filter_func_args(TransformerEncoderV2.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ) -> None:
        """Register constructor arguments on an argument parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used to namespace the arguments.
            skip: Argument names that should not be added.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        original_add_argument = parser.add_argument

        def add_argument(*args: Any, **kwargs: Any) -> Any:
            if args:
                arg_name = args[0]
                if isinstance(arg_name, str):
                    skip_name = arg_name.lstrip("-").replace("-", "_")
                    if skip_name in skip:
                        return None
            return original_add_argument(*args, **kwargs)

        parser.add_argument = add_argument  # type: ignore[method-assign]
        try:
            parser.add_argument(
                "--in-feats", default=80, type=int, help="input features dimension"
            )
            parser.add_argument(
                "--stem-type",
                default=TransformerEncoderV2StemType.CONV2D.value,
                choices=TransformerEncoderV2StemType.choices(),
                help="Types of stem block in [conv1d, conv2d]",
            )
            parser.add_argument(
                "--stem-hidden-channels",
                default=[512, 512],
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
                default=TransformerV2AttType.TORCH_SDP.value,
                choices=TransformerV2AttType.choices(),
                help="type of attention layer in [sdp, torch_sdp, hf_flash_sdp]",
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
                "--att-dropout-rate",
                default=0.0,
                type=float,
                help="attention dropout rate",
            )
            parser.add_argument(
                "--att-bias",
                default=False,
                action=ActionYesNo,
                help="use bias in Linear layers of attention blocks",
            )
            parser.add_argument(
                "--num-kv-shared-layers",
                default=0,
                type=Union[int, List[int]],
                help="shared suffix layers per superblock; integer for all stages or a list of counts (default: 0)",
            )
            parser.add_argument(
                "--enable-qk-norm",
                default=False,
                action=ActionYesNo,
                help="enable per-head Q/K RMSNorm and unit attention scaling",
            )
            parser.add_argument(
                "--enable-v-norm",
                default=False,
                action=ActionYesNo,
                help="enable per-head value RMSNorm without learned scaling",
            )
            parser.add_argument(
                "--ff-type",
                default=TransformerV2FeedForwardType.MLP.value,
                choices=TransformerV2FeedForwardType.choices(),
                help="type of feed forward layer in [mlp, convnext, g4moe]",
            )
            parser.add_argument(
                "--ff-dim-multiplier",
                default=4,
                type=float,
                help="hidden dimension multiplier for the dense feed-forward width (dense branch for g4moe)",
            )
            parser.add_argument(
                "--ff-multiple-of",
                default=256,
                type=int,
                help="round dense and expert intermediate widths up to this multiple",
            )
            parser.add_argument(
                "--ff-num-experts",
                default=None,
                type=int,
                help="positive number of routed experts; required for g4moe",
            )
            parser.add_argument(
                "--ff-top-k-experts",
                default=None,
                type=int,
                help="experts selected per token in [1, ff_num_experts]; required for g4moe",
            )
            parser.add_argument(
                "--ff-moe-intermediate-dim",
                default=None,
                type=int,
                help="positive expert width before rounding by ff_multiple_of; required for g4moe",
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
                "--ff-act",
                default="silu",
                help="gated feed-forward activation (use gelu-tanh to match Gemma 4)",
            )
            parser.add_argument(
                "--ff-bias",
                default=False,
                action=ActionYesNo,
                help="use bias in Linear layers of feed forward blocks",
            )
            parser.add_argument(
                "--downb-strides",
                default=[1],
                type=int,
                nargs="+",
                help="strides to be downsample feature maps before each encoder stage",
            )

            parser.add_argument(
                "--local-attention-sliding-window",
                default=None,
                type=int,
                help="Local attention window in stage tokens; None is unrestricted.",
            )
            parser.add_argument(
                "--global-attention-sliding-window",
                default=None,
                type=int,
                help="Global attention window in stage tokens; None is unrestricted.",
            )
            parser.add_argument(
                "--local-to-global-ratio",
                default=0,
                type=int,
                help="Local layers per global layer across stages; 0 means all global. The final layer is global.",
            )
            parser.add_argument(
                "--global-k-eq-v",
                default=False,
                action=ActionYesNo,
                help="reuse the raw key projection as values in global layers only",
            )
            parser.add_argument(
                "--local-head-dim",
                default=None,
                type=int,
                help="local attention head width; None derives hidden_dims[i] / num_heads",
            )
            parser.add_argument(
                "--global-head-dim",
                default=None,
                type=int,
                help="global attention head width; None derives hidden_dims[i] / num_heads",
            )
            parser.add_argument(
                "--local-rope-theta",
                default=10000.0,
                type=float,
                help="Local RoPE frequency base.",
            )
            parser.add_argument(
                "--global-rope-theta",
                default=1000000.0,
                type=float,
                help="Global RoPE frequency base.",
            )
            parser.add_argument(
                "--local-rope-partial-rotary-factor",
                default=1.0,
                type=float,
                help="Fraction of local head dimensions rotated using full-head frequency spacing.",
            )
            parser.add_argument(
                "--global-rope-partial-rotary-factor",
                default=1.0,
                type=float,
                help="Fraction of global head dimensions rotated using full-head frequency spacing.",
            )
            parser.add_argument(
                "--local-rope-scale-freqs",
                default=True,
                action=ActionYesNo,
                help="Apply wavelength-based frequency scaling to local RoPE.",
            )
            parser.add_argument(
                "--global-rope-scale-freqs",
                default=True,
                action=ActionYesNo,
                help="Apply wavelength-based frequency scaling to global RoPE.",
            )
            parser.add_argument(
                "--rope-update-max-seq-length",
                default=True,
                action=ActionYesNo,
                help="grow each stage/type RoPE scaling reference length during training",
            )
            parser.add_argument(
                "--rope-original-max-seq-length",
                default=None,
                type=int,
                help="original RoPE context length override; None uses the positional encoder default",
            )
            parser.add_argument(
                "--rope-scaling-factor",
                default=8,
                type=float,
                help="ROPE scaling factors",
            )
            parser.add_argument(
                "--rope-low-freq-factor",
                default=1,
                type=float,
                help="low-frequency threshold: wavelengths above reference length / low_freq_factor are fully scaled",
            )
            parser.add_argument(
                "--rope-high-freq-factor",
                default=4,
                type=float,
                help="high-frequency threshold: wavelengths below reference length / high_freq_factor are unchanged",
            )
            parser.add_argument(
                "--out-feats",
                default=None,
                type=int,
                help="features for output projection, if None, no output proj is done",
            )
            parser.add_argument(
                "--drop-path-rate", default=0.0, type=float, help="drop path rate"
            )
            parser.add_argument(
                "--norm-layer",
                default=TransformerV2NormLayerType.LAYERNORM.value,
                choices=TransformerV2NormLayerType.choices(),
                help="branch norm type in [layer-norm, rms-norm]; independent of pre_post_norm",
            )
            parser.add_argument(
                "--pre-post-norm",
                default=False,
                action=ActionYesNo,
                help="add branch post-norms before residual addition using norm_layer (default: pre-norm only)",
            )
            parser.add_argument(
                "--norm-eps",
                default=1e-5,
                type=float,
                help="epsilon for branch, Q/K/V, and MoE router normalization",
            )
            parser.add_argument(
                "--is-causal",
                default=False,
                action=ActionYesNo,
                help="use causal attention and streaming 1-D convolutions; rejects conv2d stems and ConvNeXt",
            )

            parser.add_argument(
                "--flash-attention-version",
                default=2,
                type=int,
                choices=[2, 3, 4],
                help="process-wide native Torch Flash Attention version; FA3/FA4 require newer PyTorch and kernel support",
            )
            parser.add_argument(
                "--sdp-backend",
                default=SDPBackendType.default().value,
                choices=SDPBackendType.choices(),
                help="backend to use for native torch scaled dot product attention",
            )
            parser.add_argument(
                "--model-parallel",
                default=False,
                action=ActionYesNo,
                help="use tensor-parallel projections with an externally initialized process group",
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
                    "0-based encoder stage indices to aggregate in mfa; "
                    "if None, all encoder stages are aggregated"
                ),
            )
            parser.add_argument(
                "--endpoint-scale-layer",
                default=-1,
                type=int,
                help=(
                    "encoder stage index that indicates the MFA time scale; "
                    "supports Python-style negative indexing"
                ),
            )
        finally:
            parser.add_argument = original_add_argument  # type: ignore[method-assign]

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments accepted by fine-tuning helpers.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by ``change_config``.
        """
        return filter_func_args(TransformerEncoderV2.change_config, kwargs)

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set([])
    ) -> None:
        """Register fine-tuning arguments on an argument parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used to namespace the arguments.
            skip: Argument names that should not be added.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        original_add_argument = parser.add_argument

        def add_argument(*args: Any, **kwargs: Any) -> Any:
            if args:
                arg_name = args[0]
                if isinstance(arg_name, str):
                    skip_name = arg_name.lstrip("-").replace("-", "_")
                    if skip_name in skip:
                        return None
            return original_add_argument(*args, **kwargs)

        parser.add_argument = add_argument  # type: ignore[method-assign]
        try:
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
            except Exception:
                pass

            try:
                parser.add_argument(
                    "--drop-path-rate",
                    default=0,
                    type=float,
                    help="layer drop probability",
                )
            except Exception:
                pass

            try:
                parser.add_argument(
                    "--att-dropout-rate",
                    default=0,
                    type=float,
                    help="attention layers dropout rate",
                )
            except Exception:
                pass
        finally:
            parser.add_argument = original_add_argument  # type: ignore[method-assign]

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
