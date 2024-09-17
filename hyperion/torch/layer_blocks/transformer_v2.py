"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import List, Optional, Tuple, Type, Union

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from ..layers import ActivationFactory as AF
from ..layers import DropPath1d, GRN1d, RMSNorm
from ..layers.attention_v2 import (
    FlashScaledDotProdAttV2,
    ScaledDotProdAttV2,
    TorchScaledDotProdAttV2,
)
from ..utils import scale_seq_lengths, seq_lengths_to_mask


class TransformerEncoderV2StemType(str, Enum):
    CONV1D = "conv1d"
    CONV2D = "conv2d"

    @staticmethod
    def choices():
        return [o.value for o in TransformerEncoderV2StemType]

    @staticmethod
    def to_class(value):
        # stem block
        if value == TransformerEncoderV2StemType.CONV1D:
            stem_class = TransfomerV2Conv1dStemBlock
        elif value == TransformerEncoderV2StemType.CONV2D:
            stem_class = TransfomerV2Conv2dStemBlock
        else:
            raise ValueError(f"invalid {value=}")

        return stem_class


class TransformerV2NormLayerType(str, Enum):
    LAYERNORM = "layer-norm"
    RMSNORM = "rms-norm"

    @staticmethod
    def choices():
        return [o.value for o in TransformerV2NormLayerType]

    @staticmethod
    def to_class(value):
        if value is None or value == TransformerV2NormLayerType.LAYERNORM:
            return nn.LayerNorm
        elif value == TransformerV2NormLayerType.RMSNORM:
            return RMSNorm
        else:
            raise ValueError(f"invalid {value=}")


class TransformerV2AttType(str, Enum):
    SDP = "sdp"
    TORCH_SDP = "torch_sdp"
    FLASH_SDP = "flash_sdp"

    @staticmethod
    def choices():
        return [o.value for o in TransformerV2AttType]

    @staticmethod
    def to_class(value):
        if value == TransformerV2AttType.SDP:
            return ScaledDotProdAttV2
        elif value == TransformerV2AttType.TORCH_SDP:
            return TorchScaledDotProdAttV2
        elif value == TransformerV2AttType.FLASH_SDP:
            return FlashScaledDotProdAttV2
        else:
            raise ValueError(f"invalid {value=}")


class TransformerV2FeedForwardType(str, Enum):
    MLP = "mlp"
    CONVNEXT = "convnext"

    @staticmethod
    def choices():
        return [o.value for o in TransformerV2FeedForwardType]

    @staticmethod
    def to_class(value):
        if value == TransformerV2FeedForwardType.MLP:
            return TransformerV2MLPBlock
        elif value == TransformerV2FeedForwardType.CONVNEXT:
            return TransformerV2ConvNextBlock
        else:
            raise ValueError(f"invalid {value=}")


class Conv2dStemLayer(nn.Module):
    """Conv2d layer for 2d stem

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      activation: activation function string
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
      bias: convolution has bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str,
        norm_layer: Type[nn.Module],
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels, eps=norm_eps)
        self.act = AF.create(activation)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(self.norm(x.permute(0, 2, 3, 1)))
        return x.permute(0, 3, 1, 2)  # .contiguous()


class Conv1dStemLayer(nn.Module):
    """Conv1d layer for 1d stem

    Args:
      in_channels: input channels
      out_channels: output channels
      kernel_size: kernel size of the convolution
      stride: stride of the convolution
      activation: activation function string
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
      bias: convolution has bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str,
        norm_layer: Type[nn.Module],
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels, eps=norm_eps)
        self.act = AF.create(activation)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(self.norm(x.permute(0, 2, 1)))
        return x.permute(0, 2, 1)  # .contiguous()


class TransfomerV2Conv2dStemBlock(nn.Module):
    """ConvNext-v2 2d input block

    Args:
      in_channels: input channels
      out_channels: output channels
      hidden_channels: channels of the convolutions
      kernel_sizes: kernel sizes of the convolutions
      strides: stride of the convolution
      activation: activation function string
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
      norm_eps: epsilon for norm layer
      dropout_rate: dropout probility
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_channels: List[int] = [128],
        kernel_sizes: List[int] = [4],
        strides: List[int] = [2],
        activation: str = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        norm_eps: float = 1e-5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        if norm_layer == RMSNorm:
            conv_bias = True
        else:
            conv_bias = False

        conv_i = Conv2dStemLayer(
            in_feats,
            hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            activation=activation,
            norm_layer=norm_layer,
            bias=conv_bias,
            norm_eps=norm_eps,
        )
        conv_layers = [conv_i]
        feat_dim = in_feats
        feat_dim = (feat_dim + strides[0] - 1) // strides[0]

        self.context = conv_i.context
        self.dowsample_factor = strides[0]
        for i in range(len(hidden_channels) - 1):
            conv_i = Conv2dStemLayer(
                hidden_channels[i],
                hidden_channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                activation=activation,
                norm_layer=norm_layer,
                bias=conv_bias,
                norm_eps=norm_eps,
            )
            conv_layers.append(conv_i)
            feat_dim = feat_dim = (feat_dim + strides[i] - 1) // strides[i]
            self.context += conv_i.context * self.downsample_factor
            self.dowsample_factor *= strides[i]

        self.conv_layers = nn.Sequential(conv_layers)
        self.norm_layer = norm_layer(feat_dim * hidden_channels[-1], eps=norm_eps)
        self.projection = nn.Linear(feat_dim * hidden_channels[-1], out_feats)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        bs, t_in, c = x.size()
        x = x.view(bs, 1, t_in, c).permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        bs, c, f, t_out = x.size()
        x = x.permute(0, 3, 2, 1).reshape(bs, t_out, -1)
        x = self.norm_layer(x)
        if x_lengths is not None:
            x_lengths = scale_seq_lengths(x_lengths, t_out, t_in)
            x_mask = ~seq_lengths_to_mask(x_lengths, t_out)
            x = x.masked_fill(x_mask, 0.0)

        x_proj = self.projection(x)
        x_proj = self.dropout(x_proj)
        if x_lengths is not None:
            x_proj = x.masked_fill(x_mask, 0.0)

        return x, x_proj, x_lengths


class TransfomerV2Conv1dStemBlock(nn.Module):
    """ConvNext-v2 2d input block

    Args:
      in_channels: input channels
      out_channels: output channels
      hidden_channels: channels of the convolutions
      kernel_sizes: kernel sizes of the convolutions
      strides: stride of the convolution
      activation: activation function string
      norm_layer: normalization layer constructor, if None, LayerNorm is used.
      norm_eps: epsilon for norm layer
      dropout_rate: dropout probility
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_channels: List[int] = [128],
        kernel_sizes: List[int] = [4],
        strides: List[int] = [2],
        activation: str = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        norm_eps: float = 1e-5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        if norm_layer == RMSNorm:
            conv_bias = True
        else:
            conv_bias = False

        conv_i = Conv1dStemLayer(
            in_feats,
            hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            activation=activation,
            norm_layer=norm_layer,
            bias=conv_bias,
            norm_eps=norm_eps,
        )
        conv_layers = [conv_i]

        self.context = conv_i.context
        self.dowsample_factor = strides[0]
        for i in range(len(hidden_channels) - 1):
            conv_i = Conv1dStemLayer(
                hidden_channels[i],
                hidden_channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                activation=activation,
                norm_layer=norm_layer,
                bias=conv_bias,
                norm_eps=norm_eps,
            )
            conv_layers.append(conv_i)
            self.context += conv_i.context * self.downsample_factor
            self.dowsample_factor *= strides[i]

        self.conv_layers = nn.Sequential(conv_layers)
        self.norm_layer = norm_layer(hidden_channels[-1], eps=norm_eps)
        self.projection = nn.Linear(hidden_channels[-1], out_feats)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        bs, t_in, c = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm_layer(x)
        if x_lengths is not None:
            x_lengths = scale_seq_lengths(x_lengths, x.size(1), t_in)
            x_mask = ~seq_lengths_to_mask(x_lengths, x.size(1))
            x = x.masked_fill(x_mask, 0.0)

        x_proj = self.projection(x)
        x_proj = self.dropout(x_proj)
        if x_lengths is not None:
            x_proj = x.masked_fill(x_mask, 0.0)

        return x, x_proj, x_lengths


class TransformerV2MLPBlock(nn.Module):
    """MLP Block with 1d convolutions to use as
    a replacement for feed forward layer in transformer

    Args:
        num_channels (int): Number of input channels.
        kernel_size: kernel size
        dilation: dilation factor of convolution
        activation: activation function name or object
        norm_layer: normalization layer constructor, if None, LayerNorm is used.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: Union[str, nn.Module] = "silu",
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        model_parallel: bool = False,
        **kwargs,
    ):
        super().__init__()
        # mimics LLama 3 readjustemnt of intermediate_dim
        intermediate_dim = ff_multiple_of * (
            (intermediate_dim + ff_multiple_of - 1) // ff_multiple_of
        )

        if model_parallel:
            self.gate_proj = ColumnParallelLinear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
                gather_output=False,
            )
            self.up_proj = RowParallelLinear(
                hidden_dim,
                intermediate_dim,
                bias=False,
                input_is_parallel=True,
            )
            self.down_proj = ColumnParallelLinear(
                intermediate_dim,
                hidden_dim,
                bias=False,
                gather_output=False,
            )
        else:
            self.gate_proj = nn.Linear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
            )
            self.up_proj = nn.Linear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
            )
            self.down_proj = nn.Linear(
                intermediate_dim,
                hidden_dim,
                bias=ff_bias,
            )

        self.act = AF.create(activation)
        self.context = 1

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerV2ConvNextBlock(nn.Module):
    """ConvNeXtV2 Block with 1d convolutions to use as
    a replacement for feed forward layer in transformer

    Args:
        num_channels (int): Number of input channels.
        kernel_size: kernel size
        dilation: dilation factor of convolution
        activation: activation function name or object
        norm_layer: normalization layer constructor, if None, LayerNorm is used.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        activation: Union[str, nn.Module] = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        model_parallel: bool = False,
    ):
        super().__init__()
        assert model_parallel == False
        # mimics LLama 3 readjustemnt of intermediate_dim
        intermediate_dim = ff_multiple_of * (
            (intermediate_dim + ff_multiple_of - 1) // ff_multiple_of
        )

        padding = dilation * (kernel_size - 1) // 2
        self.dwconv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_dim,
        )  # depthwise conv
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.norm = norm_layer(hidden_dim, eps=1e-6)
        self.gate_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=ff_bias,
        )
        self.up_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=ff_bias,
        )
        self.down_proj = nn.Linear(
            intermediate_dim,
            hidden_dim,
            bias=ff_bias,
        )
        self.act = AF.create(activation)
        self.grn = GRN1d(intermediate_dim, channels_last=True)
        self.context = padding

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        input = x
        x = x.permute(0, 2, 1).contiguous()  # (N, T, C) -> (N, C, T)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.grn(x, x_mask)
        x = self.down_proj(x)
        # x = input + self.drop_path(x)
        return x


class TransformerV2SelfAttBlock(nn.Module):
    def __init__(
        self,
        att_type: TransformerV2AttType,
        ff_type: TransformerV2FeedForwardType,
        num_feats: int,
        num_heads: int,
        num_kv_heads: int,
        ff_intermediate_feats: int,
        ff_kernel_size: int,
        ff_dilation: int,
        ff_activation: Union[str, nn.Module] = "silu",
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        rope=None,
        is_causal: bool = False,
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-5,
        use_cache: bool = False,
        internal_cache: bool = True,
        max_batch_size: int = 0,
        max_seq_length: int = 0,
        model_parallel: bool = False,
    ):
        super().__init__()
        att_class = TransformerV2AttType.to_model_class(att_type)
        ff_class = TransformerV2FeedForwardType.to_model_class(ff_type)
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.att_norm = norm_layer(num_feats, norm_eps)
        self.ff_norm = norm_layer(num_feats, norm_eps)

        self.attention = att_class(
            num_feats=num_feats,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=att_dropout_rate,
            use_cache=use_cache,
            internal_cache=internal_cache,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            att_bias=att_bias,
            rope=rope,
            is_causal=is_causal,
            model_parallel=model_parallel,
        )
        self.feed_foward = ff_class(
            num_feats,
            ff_intermediate_feats,
            activation=ff_activation,
            kernel_size=ff_kernel_size,
            dilation=ff_dilation,
            ff_bias=ff_bias,
            ff_multiple_of=ff_multiple_of,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath1d(drop_path_rate) if drop_path_rate > 0.0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int,
    ):
        x_norm = self.att_norm(x)
        h = x + self.attention(x_norm, x_norm, x_norm, mask, start_pos, start_pos)
        out = h + self.feed_forward(self.ff_norm(h))
        if self.drop_path is not None and self.training:
            out = x + self.drop_path(out - x)
        return out


# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim=None,
#         max_position_embeddings=2048,
#         base=10000,
#         device=None,
#         scaling_factor=1.0,
#         rope_type="default",
#         config: Optional[LlamaConfig] = None,
#     ):
#         super().__init__()
#         # TODO (joao): remove the `if` below, only used for BC
#         self.rope_kwargs = {}
#         if config is None:
#             logger.warning_once(
#                 "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
#                 "`config` argument. All other arguments will be removed in v4.45"
#             )
#             self.rope_kwargs = {
#                 "rope_type": rope_type,
#                 "factor": scaling_factor,
#                 "dim": dim,
#                 "base": base,
#                 "max_position_embeddings": max_position_embeddings,
#             }
#             self.rope_type = rope_type
#             self.max_seq_len_cached = max_position_embeddings
#             self.original_max_seq_len = max_position_embeddings
#         else:
#             # BC: "rope_type" was originally "type"
#             if config.rope_scaling is not None:
#                 self.rope_type = config.rope_scaling.get(
#                     "rope_type", config.rope_scaling.get("type")
#                 )
#             else:
#                 self.rope_type = "default"
#             self.max_seq_len_cached = config.max_position_embeddings
#             self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(
#             self.config, device, **self.rope_kwargs
#         )
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(
#                 self.config, device, seq_len=seq_len, **self.rope_kwargs
#             )
#             self.register_buffer(
#                 "inv_freq", inv_freq, persistent=False
#             )  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len

#         if (
#             seq_len < self.original_max_seq_len
#             and self.max_seq_len_cached > self.original_max_seq_len
#         ):  # reset
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len

#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_ids, device=x.device)

#         # Core RoPE block
#         inv_freq_expanded = (
#             self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         )
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = (
#             device_type
#             if isinstance(device_type, str) and device_type != "mps"
#             else "cpu"
#         )
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (
#                 inv_freq_expanded.float() @ position_ids_expanded.float()
#             ).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()

#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(*shape)


# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# # meta
# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
#     bs, slen, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     return (
#         x[:, :, :, None, :]
#         .expand(bs, slen, n_kv_heads, n_rep, head_dim)
#         .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
#     )

# # hf
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(
#         batch, num_key_value_heads, n_rep, slen, head_dim
#     )
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         model_parallel_size = fs_init.get_model_parallel_world_size()
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.wq = ColumnParallelLinear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wk = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wv = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wo = RowParallelLinear(
#             args.n_heads * self.head_dim,
#             args.dim,
#             bias=False,
#             input_is_parallel=True,
#             init_method=lambda x: x,
#         )

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         bsz, seqlen, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         self.cache_k = self.cache_k.to(xq)
#         self.cache_v = self.cache_v.to(xq)

#         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         keys = self.cache_k[:bsz, : start_pos + seqlen]
#         values = self.cache_v[:bsz, : start_pos + seqlen]

#         # repeat k/v heads if n_kv_heads < n_heads
#         keys = repeat_kv(
#             keys, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
#         values = repeat_kv(
#             values, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         values = values.transpose(
#             1, 2
#         )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)


# class LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         if layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.v_proj = nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.o_proj = nn.Linear(
#             self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
#         )

#         # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
#         self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         if self.config.pretraining_tp > 1:
#             key_value_slicing = (
#                 self.num_key_value_heads * self.head_dim
#             ) // self.config.pretraining_tp
#             query_slices = self.q_proj.weight.split(
#                 (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#             )
#             key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#             value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

#             query_states = [
#                 F.linear(hidden_states, query_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             query_states = torch.cat(query_states, dim=-1)

#             key_states = [
#                 F.linear(hidden_states, key_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             key_states = torch.cat(key_states, dim=-1)

#             value_states = [
#                 F.linear(hidden_states, value_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             value_states = torch.cat(value_states, dim=-1)

#         else:
#             query_states = self.q_proj(hidden_states)
#             key_states = self.k_proj(hidden_states)
#             value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)
#         attn_weights = torch.matmul(
#             query_states, key_states.transpose(2, 3)
#         ) / math.sqrt(self.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(
#             attn_weights, dim=-1, dtype=torch.float32
#         ).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(
#             attn_weights, p=self.attention_dropout, training=self.training
#         )
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, -1)

#         if self.config.pretraining_tp > 1:
#             attn_output = attn_output.split(
#                 self.hidden_size // self.config.pretraining_tp, dim=2
#             )
#             o_proj_slices = self.o_proj.weight.split(
#                 self.hidden_size // self.config.pretraining_tp, dim=1
#             )
#             attn_output = sum(
#                 [
#                     F.linear(attn_output[i], o_proj_slices[i])
#                     for i in range(self.config.pretraining_tp)
#                 ]
#             )
#         else:
#             attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class LlamaFlashAttention2(LlamaAttention):
#     """
#     Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if isinstance(past_key_value, StaticCache):
#             raise ValueError(
#                 "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
#                 "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
#             )

#         output_attentions = False

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         # therefore we just need to keep the original shape
#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
#         # to be able to avoid many of these transpose/reshape/view.
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)

#         dropout_rate = self.attention_dropout if self.training else 0.0

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32. (LlamaRMSNorm handles it correctly)

#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype

#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )

#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)

#         attn_output = _flash_attention_forward(
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             q_len,
#             position_ids=position_ids,
#             dropout=dropout_rate,
#             sliding_window=getattr(self, "sliding_window", None),
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#             is_causal=self.is_causal,
#         )

#         attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class LlamaSdpaAttention(LlamaAttention):
#     """
#     Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
#     SDPA API.
#     """

#     # Adapted from LlamaAttention.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if output_attentions:
#             # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
#             logger.warning_once(
#                 "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
#                 'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return super().forward(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#             )

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         causal_mask = attention_mask
#         if attention_mask is not None:
#             causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

#         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#         # Reference: https://github.com/pytorch/pytorch/issues/112577.
#         if query_states.device.type == "cuda" and causal_mask is not None:
#             query_states = query_states.contiguous()
#             key_states = key_states.contiguous()
#             value_states = value_states.contiguous()

#         # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
#         # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
#         is_causal = True if causal_mask is None and q_len > 1 else False

#         attn_output = torch.nn.functional.scaled_dot_product_attention(
#             query_states,
#             key_states,
#             value_states,
#             attn_mask=causal_mask,
#             dropout_p=self.attention_dropout if self.training else 0.0,
#             is_causal=is_causal,
#         )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(bsz, q_len, -1)

#         attn_output = self.o_proj(attn_output)

#         return attn_output, None, past_key_value


# LLAMA_ATTENTION_CLASSES = {
#     "eager": LlamaAttention,
#     "flash_attention_2": LlamaFlashAttention2,
#     "sdpa": LlamaSdpaAttention,
# }


# class TransformerBlock(nn.Module):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Attention(args)
#         self.feed_forward = FeedForward(
#             dim=args.dim,
#             hidden_dim=4 * args.dim,
#             multiple_of=args.multiple_of,
#             ffn_dim_multiplier=args.ffn_dim_multiplier,
#         )
#         self.layer_id = layer_id
#         self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
#         out = h + self.feed_forward(self.ffn_norm(h))
#         return out


# class LlamaDecoderLayer(nn.Module):
#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
#             config=config, layer_idx=layer_idx
#         )

#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[
#         torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
#     ]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*):
#                 attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
#                 query_sequence_length, key_sequence_length)` if default attention is used.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#             cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#                 Indices depicting the position of the input sequence tokens in the sequence
#             position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
#                 Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
#                 with `head_dim` being the embedding dimension of each attention head.
#             kwargs (`dict`, *optional*):
#                 Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
#                 into the model
#         """
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs
