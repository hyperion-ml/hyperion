"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks.transformer_v2 import (
    SDPBackendType,
    TransformerV2AttType,
    TransformerV2CrossAttBlock,
    TransformerV2FeedForwardType,
    TransformerV2NormLayerType,
    TransformerV2SelfAttBlock,
)
from ..layers import RotaryPosEncoder
from ..layers.attention_v2 import ScaledDotProdAttV2, TorchScaledDotProdAttV2
from ..layers.tensor_parallel import ColumnParallelLinear
from ..utils import seq_lengths_to_cross_attn_mask
from .net_arch import NetArch


class QFormerV2(NetArch):
    """Simplified Q-Former built on the V2 Transformer blocks.

    Attributes:
        in_feats: Input feature dimension.
        att_type: Attention implementation used in the transformer blocks.
        num_layers: Number of transformer layers.
        hidden_dim: Hidden dimension of the query stream.
        self_att_k_eq_v: Reuse raw key projections for self-attention values.
        cross_att_k_eq_v: Reuse raw key projections for cross-attention values.
        head_dim: Attention head width; None derives hidden_dim / num_heads.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads when using grouped-query attention.
        cross_att_freq: Frequency of cross-attention layers.
        att_dropout_rate: Attention dropout probability.
        att_bias: Whether attention projections use bias terms.
        enable_v_norm: Enable per-head value RMSNorm without learned scaling.
        enable_qk_norm: Enable per-head Q/K RMSNorm before RoPE and unit attention scaling.
        ff_type: Feed-forward implementation used in the transformer blocks.
        ff_dim_multiplier: Multiplier for the dense feed-forward width, including the dense g4moe branch.
        ff_multiple_of: Bottleneck size is rounded to a multiple of this value.
        ff_num_experts: Total routed experts when ff_type is g4moe.
        ff_top_k_experts: Experts selected per token when ff_type is g4moe.
        ff_moe_intermediate_dim: Intermediate width per routed expert before rounding by ff_multiple_of.
        ff_kernel_size: Kernel size used by convolutional feed-forward blocks.
        ff_dilation: Dilation used by convolutional feed-forward blocks.
        ff_act: Feed-forward activation name.
        ff_bias: Whether feed-forward projections use bias terms.
        rope_in_self_att: Enable rotary position encoding in self-attention.
        rope_in_cross_att: Enable rotary position encoding in cross-attention.
        rope_theta: Rotary position encoding base theta.
        rope_scale_freqs: Enable frequency scaling for long sequences.
        rope_update_max_seq_length: Track the maximum sequence length seen during training.
        rope_original_max_seq_length: Optional explicit original sequence length for RoPE.
        rope_scaling_factor: Rotary scaling factor.
        rope_low_freq_factor: Low-frequency RoPE scaling threshold.
        rope_high_freq_factor: High-frequency RoPE scaling threshold.
        out_feats: Optional output projection dimension.
        drop_path_rate: Global stochastic-depth rate.
        flash_attention_version: Process-wide native Torch Flash Attention version (2, 3, or 4).
        sdp_backend: Preferred scaled dot-product attention backend.
        norm_layer: Normalization layer type.
        norm_eps: Epsilon used by normalization layers.
        pre_post_norm: Add branch post-norms before residual addition using norm_layer.
        tied_layers: Share transformer blocks across layers when enabled.
        multilayer_input: Consume a list of encoder-layer feature tensors.
        distribute_query_across_layers: Split queries across cross-attention groups.
        use_layer_idx_encoder: Add a learned source-layer embedding to cross-attention inputs.
        model_parallel: Use built-in tensor-parallel projections with an externally initialized process group.

    """

    def __init__(
        self,
        in_feats: int,
        att_type: TransformerV2AttType = TransformerV2AttType.TORCH_SDP,
        num_layers: int = 3,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        self_att_k_eq_v: bool = False,
        cross_att_k_eq_v: bool = False,
        cross_att_freq: int = 1,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        enable_qk_norm: bool = False,
        enable_v_norm: bool = False,
        ff_type: TransformerV2FeedForwardType = TransformerV2FeedForwardType.MLP,
        ff_dim_multiplier: int = 4,
        ff_multiple_of: int = 256,
        ff_num_experts: Optional[int] = None,
        ff_top_k_experts: Optional[int] = None,
        ff_moe_intermediate_dim: Optional[int] = None,
        ff_kernel_size: int = 7,
        ff_dilation: int = 1,
        ff_act: str = "silu",
        ff_bias: bool = False,
        rope_in_self_att: bool = False,
        rope_in_cross_att: bool = False,
        rope_theta: float = 50000,
        rope_scale_freqs: bool = True,
        rope_update_max_seq_length: bool = True,
        rope_original_max_seq_length: Optional[int] = None,
        rope_scaling_factor: float = 8,
        rope_low_freq_factor: float = 1,
        rope_high_freq_factor: float = 4,
        out_feats: Optional[int] = None,
        drop_path_rate: float = 0.0,
        sdp_backend: SDPBackendType = SDPBackendType.default(),
        flash_attention_version: int = 2,
        norm_layer: TransformerV2NormLayerType = TransformerV2NormLayerType.LAYERNORM,
        norm_eps: float = 1e-5,
        pre_post_norm: bool = False,
        tied_layers: bool = False,
        multilayer_input: bool = False,
        distribute_query_across_layers: bool = False,
        use_layer_idx_encoder: bool = False,
        model_parallel: bool = False,
    ) -> None:
        """Initialize the Q-Former module.

        Args:
            in_feats: Input feature dimension of the encoder stream.
            att_type: Attention backend implementation.
            num_layers: Number of query-stream transformer layers.
            hidden_dim: Hidden dimension of the query stream.
            self_att_k_eq_v: Reuse raw key projections for self-attention values. Defaults to False.
            cross_att_k_eq_v: Reuse raw key projections for cross-attention values. Defaults to False.
            head_dim: Positive attention head width; None derives hidden_dim / num_heads. Defaults to None.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads; None uses num_heads.
            cross_att_freq: Cross-attention period; must be positive and divide num_layers.
            att_dropout_rate: Attention weight dropout probability.
            att_bias: Enable bias in attention projections.
            enable_v_norm: Enable per-head value RMSNorm without learned scaling, independently of QK normalization.
            enable_qk_norm: Enable per-head Q/K RMSNorm before RoPE and unit attention scaling.
            ff_type: Feed-forward variant: mlp, convnext, or g4moe.
            ff_dim_multiplier: Multiplier for the dense MLP width, including the dense g4moe branch.
            ff_multiple_of: Rounding multiple for dense and routed expert intermediate widths.
            ff_num_experts: Positive routed expert count, required when ff_type is g4moe.
            ff_top_k_experts: Experts selected per token; must be in [1, ff_num_experts] for g4moe.
            ff_moe_intermediate_dim: Positive width of each routed expert before rounding; required for g4moe.
            ff_kernel_size: Depthwise convolution kernel size for convnext.
            ff_dilation: Depthwise convolution dilation for convnext.
            ff_act: Gated MLP activation. Defaults to silu; use gelu-tanh to match Gemma 4.
            ff_bias: Enable bias in feed-forward projections; the MoE router remains bias-free.
            rope_in_self_att: Apply rotary positional encoding in self-attention.
            rope_in_cross_att: Apply rotary positional encoding in cross-attention.
            rope_theta: Base frequency parameter for rotary positional encoding.
            rope_scale_freqs: Enable long-context RoPE frequency scaling.
            rope_update_max_seq_length: Update the original RoPE context length during training.
            rope_original_max_seq_length: Original RoPE context length; None uses RotaryPosEncoder's built-in default.
            rope_scaling_factor: Scaling factor for low-frequency RoPE components.
            rope_low_freq_factor: Low-frequency threshold for RoPE scaling.
            rope_high_freq_factor: High-frequency threshold for RoPE scaling.
            out_feats: Optional output projection dimension; None or a nonpositive value disables it.
            drop_path_rate: Maximum stochastic-depth probability.
            flash_attention_version: Process-wide native Torch Flash Attention version; defaults to 2.
            sdp_backend: Preferred PyTorch scaled dot-product attention kernels.
            norm_layer: Branch normalization type; independent of pre_post_norm. QK/router norms remain RMS.
            norm_eps: Epsilon for branch, Q/K/V, and MoE router normalization.
            pre_post_norm: Add attention and feed-forward post-norms before residual addition; default False.
            tied_layers: Reuse one cross_att_freq-sized group of blocks throughout the query stream.
            multilayer_input: Consume encoder features from multiple source layers.
            distribute_query_across_layers: Split queries into groups introduced at successive cross-attention layers.
            use_layer_idx_encoder: Add a learned source-layer embedding; requires tied_layers=True.
            model_parallel: Use built-in tensor-parallel projections with an externally initialized process group.
        """
        super().__init__()
        self.multilayer_input = multilayer_input
        if not isinstance(in_feats, int):
            raise TypeError("in_feats must be an int representing encoder feature dim")

        self.in_feats = in_feats
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.self_att_k_eq_v = self_att_k_eq_v
        self.cross_att_k_eq_v = cross_att_k_eq_v

        self.att_type = att_type
        self.sdp_backend = sdp_backend
        self.flash_attention_version = flash_attention_version
        if self.att_type == TransformerV2AttType.TORCH_SDP:
            TorchScaledDotProdAttV2.set_flash_attention_version(flash_attention_version)
        self.att_dropout_rate = att_dropout_rate
        self.att_bias = att_bias
        self.enable_qk_norm = enable_qk_norm
        self.enable_v_norm = enable_v_norm
        self.cross_att_freq = cross_att_freq

        if cross_att_freq < 1:
            raise ValueError("cross_att_freq must be >= 1")

        if (num_layers // cross_att_freq) * cross_att_freq != num_layers:
            raise ValueError("num_layers should be multiple of cross_att_freq")

        self.ff_type = ff_type
        self.ff_dim_multiplier = ff_dim_multiplier
        self.ff_multiple_of = ff_multiple_of
        self.ff_num_experts = ff_num_experts
        self.ff_top_k_experts = ff_top_k_experts
        self.ff_moe_intermediate_dim = ff_moe_intermediate_dim
        self.ff_kernel_size = ff_kernel_size
        self.ff_dilation = ff_dilation
        self.ff_act = ff_act
        self.ff_bias = ff_bias

        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.norm_eps = norm_eps
        self.pre_post_norm = pre_post_norm
        self._norm_layer = TransformerV2NormLayerType.to_class(norm_layer)

        self.tied_layers = tied_layers
        self.distribute_query_across_layers = distribute_query_across_layers
        if self.distribute_query_across_layers:
            self.num_query_groups = self.num_layers // self.cross_att_freq
        else:
            self.num_query_groups = 1

        self.rope_in_self_att = rope_in_self_att
        self.rope_in_cross_att = rope_in_cross_att
        self.rope_theta = rope_theta
        self.rope_scale_freqs = rope_scale_freqs
        self.rope_update_max_seq_length = rope_update_max_seq_length
        self.rope_original_max_seq_length = rope_original_max_seq_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor

        if rope_in_self_att or rope_in_cross_att:
            rope_kwargs: Dict[str, Any] = dict(
                theta=rope_theta,
                scale_freqs=rope_scale_freqs,
                update_max_seq_length=rope_update_max_seq_length,
                scaling_factor=rope_scaling_factor,
                low_freq_factor=rope_low_freq_factor,
                high_freq_factor=rope_high_freq_factor,
            )
            if rope_original_max_seq_length is not None:
                rope_kwargs["original_max_seq_length"] = rope_original_max_seq_length
            self.rope = RotaryPosEncoder(**rope_kwargs)
        else:
            self.rope = None

        self.num_untied_layers = self.cross_att_freq if tied_layers else self.num_layers
        self.trans_blocks = nn.ModuleList()

        drop_rates = (
            torch.linspace(
                0.0,
                drop_path_rate,
                steps=max(self.num_untied_layers, 1),
                dtype=torch.float32,
            ).tolist()
            if drop_path_rate > 0.0
            else [0.0] * max(self.num_untied_layers, 1)
        )

        self.trans_blocks = nn.ModuleList()
        for i in range(self.num_untied_layers):
            drop_rate = drop_rates[i]
            if i % self.cross_att_freq == 0:
                block_i = TransformerV2CrossAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_feats=self.in_feats,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    self_att_k_eq_v=self.self_att_k_eq_v,
                    cross_att_k_eq_v=self.cross_att_k_eq_v,
                    ff_intermediate_feats=hidden_dim * self.ff_dim_multiplier,
                    ff_kernel_size=ff_kernel_size,
                    ff_dilation=ff_dilation,
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
                    rope=self.rope,
                    rope_in_self_att=rope_in_self_att,
                    rope_in_cross_att=rope_in_cross_att,
                    sdp_backend=self.sdp_backend,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    pre_post_norm=self.pre_post_norm,
                    drop_path_rate=drop_rate,
                    model_parallel=model_parallel,
                )
            else:
                block_i = TransformerV2SelfAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    k_eq_v=self.self_att_k_eq_v,
                    ff_intermediate_feats=hidden_dim * self.ff_dim_multiplier,
                    ff_kernel_size=ff_kernel_size,
                    ff_dilation=ff_dilation,
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
                    rope=self.rope if rope_in_self_att else None,
                    sdp_backend=self.sdp_backend,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    pre_post_norm=self.pre_post_norm,
                    drop_path_rate=drop_rate,
                    model_parallel=model_parallel,
                )
            self.trans_blocks.append(block_i)

        self.model_parallel = model_parallel

        # head feature block
        self.out_norm = self._norm_layer(hidden_dim, eps=norm_eps)
        if out_feats is not None and out_feats > 0:
            self.out_feats = out_feats
            if model_parallel:
                self.out_proj = ColumnParallelLinear(
                    hidden_dim,
                    out_feats,
                    bias=False,
                )
            else:
                self.out_proj = nn.Linear(hidden_dim, out_feats, bias=False)
        else:
            self.out_feats = None

        self.use_layer_idx_encoder = use_layer_idx_encoder
        if use_layer_idx_encoder:
            if not self.tied_layers:
                raise ValueError("layer idx encoder can be used only with tied layers")
            self.layer_idx_encoder = nn.Embedding(
                self.num_layers // self.cross_att_freq, self.in_feats
            )
        else:
            self.layer_idx_encoder = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all learnable weights following BLIP-2 defaults."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def _compute_out_size(self, in_size: int) -> int:
        """Return the temporal dimension after processing the query stream.

        Args:
            in_size: Temporal dimension of the query input.

        Returns:
            int: Output temporal dimension.

        """
        out_size = in_size
        return out_size

    def in_feats_shape(self) -> Tuple[Optional[int], Optional[int], int]:
        """Describe the expected `(batch, time, channels)` shape for encoder features.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Expected encoder feature shape.
        """
        return (None, None, self.in_feats)

    def query_shape(self) -> Tuple[Optional[int], Optional[int], int]:
        """Describe the `(batch, time, hidden_dim)` shape for the query tokens.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Expected query tensor shape.
        """
        return (None, None, self.hidden_dim)

    def out_shape(
        self,
        query_shape: Optional[Tuple[Optional[int], Optional[int], int]] = None,
    ) -> Tuple[Optional[int], Optional[int], int]:
        """Infer the output shape for a given query input shape.

        Args:
            query_shape: Optional `(batch, time, channels)` tuple describing the query input.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Output tensor shape.

        """
        out_channels = self.out_feats if self.out_feats is not None else self.hidden_dim
        if query_shape is None:
            return (None, None, out_channels)

        if len(query_shape) != 3:
            raise ValueError("query_shape must be (batch, time, channels)")
        if query_shape[1] is None:
            T = None
        else:
            T = self._compute_out_size(query_shape[1])

        return (query_shape[0], T, out_channels)

    def out_dim(self) -> int:
        """Return the output feature dimension after the Q-Former.

        Returns:
            int: Output feature dimension.
        """
        return self.out_feats if self.out_feats is not None else self.hidden_dim

    def out_channels(self) -> int:
        """Return the output feature dimension after the Q-Former.

        Returns:
            int: Output feature dimension.
        """
        return self.out_feats if self.out_feats is not None else self.hidden_dim

    @property
    def output_is_normalized(self) -> bool:
        """Return whether the output is normalized by the module.

        Returns:
            bool: ``True`` when no output projection is used.
        """
        return self.out_feats is None

    def forward(
        self,
        query_embeds: torch.Tensor,
        feats: Union[torch.Tensor, Sequence[torch.Tensor]],
        feats_lengths: Optional[
            Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]
        ] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Dispatch the forward pass to the appropriate pathway.

        Args:
            query_embeds: Query tensor of shape `(batch, query_len, hidden_dim)`.
            feats: Encoder representations; either a single tensor `(batch, seq_len, in_feats)`
                or a sequence of tensors when `multilayer_input` is enabled.
            feats_lengths: Optional sequence-length tensor(s) used to build attention masks.
            start_pos: Starting rotary/cache position for cross-attention.

        Returns:
            torch.Tensor: Output tensor with shape `(batch, query_len, out_dim)`.

        """
        if self.multilayer_input:
            return self.forward_multilayer_input(
                query_embeds, feats, feats_lengths, start_pos=start_pos
            )
        else:
            return self.forward_singlelayer_input(
                query_embeds, feats, feats_lengths, start_pos=start_pos
            )

    def forward_singlelayer_input(
        self,
        query_embeds: torch.Tensor,
        feats: Union[torch.Tensor, Sequence[torch.Tensor]],
        feats_lengths: Optional[
            Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]
        ] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Process a single stream of encoder features through the Q-Former.

        Args:
            query_embeds: Query tensor `(batch, query_len, hidden_dim)`.
            feats: Encoder features `(batch, seq_len, in_feats)` or a singleton sequence containing them.
            feats_lengths: Optional sequence-length tensor(s) aligned with `feats`.
            start_pos: Starting rotary/cache position for cross-attention.

        Returns:
            torch.Tensor: Output tensor with shape `(batch, query_len, out_dim)`.

        """
        query_dtype = query_embeds.dtype
        query_device = query_embeds.device
        if isinstance(feats, (list, tuple)):
            if len(feats) != 1:
                raise ValueError(
                    "forward_singlelayer_input expects a single feature tensor when feats is a sequence"
                )
            feats_tensor = feats[0]
            if isinstance(feats_lengths, (list, tuple)):
                if len(feats_lengths) != 1:
                    raise ValueError(
                        "forward_singlelayer_input expects a single sequence length entry when feats is a sequence"
                    )
                feats_lengths = feats_lengths[0]
        else:
            feats_tensor = feats

        if feats_lengths is not None and isinstance(feats_lengths, list):
            raise ValueError(
                "feats_lengths must be a tensor or None when feats is a single tensor"
            )

        if not torch.all(torch.isfinite(feats_tensor)):
            logging.warning("non-finite feats=%f", torch.mean(feats_tensor))

        if self.distribute_query_across_layers:
            query_length = query_embeds.size(1)
            if query_length % self.num_query_groups != 0:
                raise ValueError(
                    "query length must be divisible by num_query_groups when distribute_query_across_layers is enabled"
                )
            query_embeds = torch.split(
                query_embeds, query_embeds.size(1) // self.num_query_groups, dim=1
            )
            hidden_feats = query_embeds[0]
            mask_query_length = query_length
        else:
            hidden_feats = query_embeds
            mask_query_length = query_embeds.size(1)

        if feats_lengths is not None:
            feats_mask = seq_lengths_to_cross_attn_mask(
                None,
                feats_lengths,
                max_query_length=mask_query_length,
                max_kv_length=feats_tensor.size(1),
                dtype=query_dtype,
                device=query_device,
                none_if_all_max=True,
            )
        else:
            feats_mask = None

        for i in range(self.num_layers):
            layer_idx = i % self.num_untied_layers
            if i % self.cross_att_freq == 0:
                if self.distribute_query_across_layers and i > 0:
                    hidden_feats = torch.cat(
                        (hidden_feats, query_embeds[i // self.cross_att_freq]), dim=1
                    )

                if self.use_layer_idx_encoder:
                    layer_idx_embeds = self.layer_idx_encoder.weight[
                        i // self.cross_att_freq
                    ].view(1, 1, -1)
                    cur_feats = feats_tensor + layer_idx_embeds
                else:
                    cur_feats = feats_tensor
                hidden_feats = self.trans_blocks[layer_idx](
                    hidden_feats,
                    x_kv=cur_feats,
                    x_kv_mask=feats_mask,
                    start_pos_kv=start_pos,
                )
            else:
                hidden_feats = self.trans_blocks[layer_idx](hidden_feats)

            if not torch.all(torch.isfinite(hidden_feats)):
                logging.warning(
                    "non-finite x-enc-%d-avg=%f", i, torch.mean(hidden_feats)
                )

        if self.out_feats is not None:
            out_feats = self.out_proj(self.out_norm(hidden_feats))
        else:
            out_feats = self.out_norm(hidden_feats).type_as(hidden_feats)

        if not torch.all(torch.isfinite(out_feats)):
            logging.warning("non-finite x-out-avg=%f", torch.mean(out_feats))
        return out_feats

    def forward_multilayer_input(
        self,
        query_embeds: torch.Tensor,
        feats: Sequence[torch.Tensor],
        feats_lengths: Optional[
            Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]
        ] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Process encoder hidden states from multiple layers one cross-att step at a time.

        Args:
            query_embeds: Query tensor `(batch, query_len, hidden_dim)`.
            feats: Sequence of encoder feature tensors, one per cross-attention layer.
            feats_lengths: Optional sequence lengths. It can be:
                - a single tensor shared by all `feats` entries (all entries must have
                  the same temporal length), or
                - a sequence matching `feats`, potentially with per-entry lengths.
            start_pos: Starting rotary/cache position for cross-attention.

        Returns:
            torch.Tensor: Output tensor with shape `(batch, query_len, out_dim)`.

        """
        if not isinstance(feats, (list, tuple)):
            raise ValueError(
                "forward_multilayer_input expects feats to be a list/tuple of tensors"
            )

        if len(feats) != self.num_layers // self.cross_att_freq:
            raise ValueError(
                f"feats ({len(feats)}) must match num cross-attention layers ({self.num_layers // self.cross_att_freq}) when multilayer_input is enabled"
            )
        shared_feats_lengths = None
        if feats_lengths is not None:
            if torch.is_tensor(feats_lengths):
                shared_feats_lengths = feats_lengths
                shared_max_kv_length = feats[0].size(1)
                for idx, feat in enumerate(feats):
                    if feat.size(1) != shared_max_kv_length:
                        raise ValueError(
                            "when feats_lengths is a tensor, all feats must share the same time length"
                        )
            else:
                if len(feats) != len(feats_lengths):
                    raise ValueError(
                        "feats_lengths must align with feats for multilayer input"
                    )
        for idx, feat in enumerate(feats):
            if not torch.all(torch.isfinite(feat)):
                logging.warning("non-finite x-in-%d-avg=%f", idx, torch.mean(feat))
                break

        query_dtype = query_embeds.dtype
        query_device = query_embeds.device
        if self.distribute_query_across_layers:
            query_length = query_embeds.size(1)
            if query_length % self.num_query_groups != 0:
                raise ValueError(
                    "query length must be divisible by num_query_groups when distribute_query_across_layers is enabled"
                )
            query_embeds = torch.split(
                query_embeds, query_embeds.size(1) // self.num_query_groups, dim=1
            )
            hidden_feats = query_embeds[0]
            mask_query_length = query_length
        else:
            hidden_feats = query_embeds
            mask_query_length = query_embeds.size(1)

        if shared_feats_lengths is not None:
            feats_mask = seq_lengths_to_cross_attn_mask(
                None,
                shared_feats_lengths,
                max_query_length=mask_query_length,
                max_kv_length=shared_max_kv_length,
                dtype=query_dtype,
                device=query_device,
                none_if_all_max=True,
            )
        else:
            feats_mask = None

        for i in range(self.num_layers):
            layer_idx = i % self.num_untied_layers
            if i % self.cross_att_freq == 0:
                if self.distribute_query_across_layers and i > 0:
                    hidden_feats = torch.cat(
                        (hidden_feats, query_embeds[i // self.cross_att_freq]), dim=1
                    )

                feats_idx = i // self.cross_att_freq
                cur_feats = feats[feats_idx]

                if shared_feats_lengths is None and feats_lengths is not None:
                    cur_feats_lengths = feats_lengths[feats_idx]
                    if cur_feats_lengths is not None:
                        feats_mask = seq_lengths_to_cross_attn_mask(
                            None,
                            cur_feats_lengths,
                            max_query_length=mask_query_length,
                            max_kv_length=cur_feats.size(1),
                            dtype=query_dtype,
                            device=query_device,
                            none_if_all_max=True,
                        )
                    else:
                        feats_mask = None
                elif shared_feats_lengths is None:
                    feats_mask = None

                if self.use_layer_idx_encoder:
                    layer_idx_embeds = self.layer_idx_encoder.weight[feats_idx].view(
                        1, 1, -1
                    )
                    cur_feats = cur_feats + layer_idx_embeds

                hidden_feats = self.trans_blocks[layer_idx](
                    hidden_feats,
                    x_kv=cur_feats,
                    x_kv_mask=feats_mask,
                    start_pos_kv=start_pos,
                )
            else:
                hidden_feats = self.trans_blocks[layer_idx](hidden_feats)

            if not torch.all(torch.isfinite(hidden_feats)):
                logging.warning(
                    "non-finite x-enc-%d-avg=%f", i, torch.mean(hidden_feats)
                )

        if self.out_feats is not None:
            out_feats = self.out_proj(self.out_norm(hidden_feats))
        else:
            out_feats = self.out_norm(hidden_feats).type_as(hidden_feats)

        if not torch.all(torch.isfinite(out_feats)):
            logging.warning("non-finite x-out-avg=%f", torch.mean(out_feats))
        return out_feats

    def requires_ddp_find_unused_parameters(self) -> bool:
        """Return whether expert routing requires DDP unused-parameter detection.

        Returns:
            ``True`` when the feed-forward blocks use G4MoE.
        """
        return self.ff_type == TransformerV2FeedForwardType.G4MoE

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the constructor arguments.

        Args:
            no_class_name: If ``True``, omit the ``class_name`` entry from the base config.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstructing the module.
        """
        config = {
            "in_feats": self.in_feats,
            "att_type": self.att_type,
            "sdp_backend": self.sdp_backend,
            "flash_attention_version": self.flash_attention_version,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "self_att_k_eq_v": self.self_att_k_eq_v,
            "cross_att_k_eq_v": self.cross_att_k_eq_v,
            "cross_att_freq": self.cross_att_freq,
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
            "ff_kernel_size": self.ff_kernel_size,
            "ff_dilation": self.ff_dilation,
            "ff_act": self.ff_act,
            "ff_bias": self.ff_bias,
            "rope_in_self_att": self.rope_in_self_att,
            "rope_in_cross_att": self.rope_in_cross_att,
            "rope_theta": self.rope_theta,
            "rope_scale_freqs": self.rope_scale_freqs,
            "rope_update_max_seq_length": self.rope_update_max_seq_length,
            "rope_original_max_seq_length": self.rope_original_max_seq_length,
            "rope_scaling_factor": self.rope_scaling_factor,
            "rope_low_freq_factor": self.rope_low_freq_factor,
            "rope_high_freq_factor": self.rope_high_freq_factor,
            "out_feats": self.out_feats,
            "norm_eps": self.norm_eps,
            "pre_post_norm": self.pre_post_norm,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "tied_layers": self.tied_layers,
            "multilayer_input": self.multilayer_input,
            "distribute_query_across_layers": self.distribute_query_across_layers,
            "use_layer_idx_encoder": self.use_layer_idx_encoder,
            "model_parallel": self.model_parallel,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, override_dropouts: bool, drop_path_rate: float, att_dropout_rate: float
    ) -> None:
        """Optionally override dropout hyperparameters during fine-tuning.

        Args:
            override_dropouts: If ``True``, propagate the supplied dropout values.
            drop_path_rate: New global stochastic depth probability.
            att_dropout_rate: New attention dropout probability.

        """
        if override_dropouts:
            logging.info("changing qformer dropouts")
            self.change_dropouts(drop_path_rate, att_dropout_rate)

    def change_dropouts(self, drop_path_rate: float, att_dropout_rate: float) -> None:
        """Propagate new dropout values through all attention and DropPath modules.

        Args:
            drop_path_rate: Target stochastic depth probability.
            att_dropout_rate: Target attention dropout probability.

        """
        from ..layers import DropPath1d

        drop_rates = (
            torch.linspace(
                0.0,
                drop_path_rate,
                steps=max(self.num_untied_layers, 1),
                dtype=torch.float32,
            ).tolist()
            if drop_path_rate > 0.0
            else [0.0] * max(self.num_untied_layers, 1)
        )

        for block, module_drop_rate in zip(self.trans_blocks, drop_rates):
            if block.drop_path is None:
                if module_drop_rate > 0.0:
                    block.drop_path = DropPath1d(module_drop_rate)
                    block.drop_path.train(self.training)
            else:
                block.drop_path.p = module_drop_rate

        for module in self.modules():
            if isinstance(module, ScaledDotProdAttV2):
                module.dropout_rate = att_dropout_rate

        self.drop_path_rate = drop_path_rate
        self.att_dropout_rate = att_dropout_rate

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments down to those accepted by the constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by ``__init__``.

        """
        return filter_func_args(QFormerV2.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register constructor arguments with a JSONArgParse parser.

        Args:
            parser: Target `ArgumentParser` instance.
            prefix: Optional nested argument namespace.
            skip: Optional set of field names to exclude.

        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        def add_argument(name: str, *args: Any, **kwargs: Any) -> None:
            if name not in skip:
                parser.add_argument(*args, **kwargs)

        if "in_feats" not in skip:
            parser.add_argument("--in-feats", type=int, help="input features dimension")

        add_argument(
            "att_type",
            "--att-type",
            default=TransformerV2AttType.TORCH_SDP.value,
            choices=TransformerV2AttType.choices(),
            help="type of attention layer in [sdp, torch_sdp, hf_flash_sdp]",
        )
        add_argument(
            "num_layers",
            "--num-layers",
            default=3,
            type=int,
            help="transformer block repeats in each encoder stage",
        )
        add_argument(
            "hidden_dim",
            "--hidden-dim",
            default=768,
            type=int,
            help="transformer block hidden features in each encoder stage",
        )
        add_argument(
            "num_heads",
            "--num-heads",
            default=12,
            type=int,
            help="num of attention heads",
        )
        add_argument(
            "num_kv_heads",
            "--num-kv-heads",
            default=None,
            type=int,
            help="num. of key, value attention heads when using GQA",
        )
        add_argument(
            "self_att_k_eq_v",
            "--self-att-k-eq-v",
            default=False,
            action=ActionYesNo,
            help="reuse raw key projections for self-attention values",
        )
        add_argument(
            "cross_att_k_eq_v",
            "--cross-att-k-eq-v",
            default=False,
            action=ActionYesNo,
            help="reuse raw key projections for cross-attention values",
        )
        add_argument(
            "head_dim",
            "--head-dim",
            default=None,
            type=int,
            help="attention head width; None derives hidden_dim / num_heads",
        )
        add_argument(
            "cross_att_freq",
            "--cross-att-freq",
            default=1,
            type=int,
            help="The frequency of adding cross-attention to the Transformer layers.",
        )
        add_argument(
            "att_dropout_rate",
            "--att-dropout-rate",
            default=0.0,
            type=float,
            help="attention dropout rate",
        )
        add_argument(
            "att_bias",
            "--att-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of attention blocks",
        )
        add_argument(
            "enable_qk_norm",
            "--enable-qk-norm",
            default=False,
            action=ActionYesNo,
            help="enable per-head Q/K RMSNorm and unit attention scaling",
        )
        add_argument(
            "enable_v_norm",
            "--enable-v-norm",
            default=False,
            action=ActionYesNo,
            help="enable per-head value RMSNorm without learned scaling",
        )
        add_argument(
            "ff_type",
            "--ff-type",
            default=TransformerV2FeedForwardType.MLP.value,
            choices=TransformerV2FeedForwardType.choices(),
            help="type of feed forward layer in [mlp, convnext, g4moe]",
        )
        add_argument(
            "ff_dim_multiplier",
            "--ff-dim-multiplier",
            default=4,
            type=int,
            help="hidden dimension multiplier for the dense feed-forward width (dense branch for g4moe)",
        )
        add_argument(
            "ff_multiple_of",
            "--ff-multiple-of",
            default=256,
            type=int,
            help="round dense and expert intermediate widths up to this multiple",
        )
        add_argument(
            "ff_num_experts",
            "--ff-num-experts",
            default=None,
            type=int,
            help="positive number of routed experts; required for g4moe",
        )
        add_argument(
            "ff_top_k_experts",
            "--ff-top-k-experts",
            default=None,
            type=int,
            help="experts selected per token in [1, ff_num_experts]; required for g4moe",
        )
        add_argument(
            "ff_moe_intermediate_dim",
            "--ff-moe-intermediate-dim",
            default=None,
            type=int,
            help="positive expert width before rounding by ff_multiple_of; required for g4moe",
        )
        add_argument(
            "ff_kernel_size",
            "--ff-kernel-size",
            default=7,
            type=int,
            help="kernel size when using convnext feed forward layer",
        )
        add_argument(
            "ff_dilation",
            "--ff-dilation",
            default=1,
            type=int,
            help="dilation when using convnext feedforward layers",
        )
        add_argument(
            "ff_act",
            "--ff-act",
            default="silu",
            help="gated feed-forward activation (use gelu-tanh to match Gemma 4)",
        )
        add_argument(
            "ff_bias",
            "--ff-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of feed forward blocks",
        )
        add_argument(
            "rope_theta",
            "--rope-theta",
            default=50000,
            type=float,
            help="ROPE base theta",
        )
        add_argument(
            "rope_scale_freqs",
            "--rope-scale-freqs",
            default=True,
            action=ActionYesNo,
            help="scale ROPE frequencies when seq lenght is larger than the maximmum length of the original training sequences",
        )
        add_argument(
            "rope_update_max_seq_length",
            "--rope-update-max-seq-length",
            default=True,
            action=ActionYesNo,
            help="update the invernal ROPE variable that keeps track of the max seq length seen on training",
        )
        add_argument(
            "rope_original_max_seq_length",
            "--rope-original-max-seq-length",
            default=None,
            type=int,
            help="original RoPE context length override; None uses the positional encoder default",
        )
        add_argument(
            "rope_scaling_factor",
            "--rope-scaling-factor",
            default=8,
            type=float,
            help="ROPE scaling factors",
        )
        add_argument(
            "rope_low_freq_factor",
            "--rope-low-freq-factor",
            default=1,
            type=float,
            help="ROPE frequencies are not scaled for wavelengths < max_seq_length / self.low_freq_factor",
        )
        add_argument(
            "rope_high_freq_factor",
            "--rope-high-freq-factor",
            default=4,
            type=float,
            help="ROPE frequencies are scaled by scaling for wavelengths > max_seq_length / self.high_freq_factor",
        )
        add_argument(
            "rope_in_self_att",
            "--rope-in-self-att",
            default=False,
            action=ActionYesNo,
            help="use Rotary positional encoder or not positional encoder at all in self-attention",
        )
        add_argument(
            "rope_in_cross_att",
            "--rope-in-cross-att",
            default=False,
            action=ActionYesNo,
            help="use Rotary positional encoder or not positional encoder at all in cross-attention",
        )
        add_argument(
            "distribute_query_across_layers",
            "--distribute-query-across-layers",
            default=False,
            action=ActionYesNo,
            help="split queries into num_layers / cross_att_freq groups and introduce one group per cross-attention layer",
        )
        add_argument(
            "tied_layers",
            "--tied-layers",
            default=False,
            action=ActionYesNo,
            help="reuse one group of query-stream transformer blocks across layers",
        )
        add_argument(
            "multilayer_input",
            "--multilayer-input",
            default=False,
            action=ActionYesNo,
            help="consume hidden features from multiple encoder layers",
        )
        add_argument(
            "use_layer_idx_encoder",
            "--use-layer-idx-encoder",
            default=False,
            action=ActionYesNo,
            help="add a learned embedding of the encoder layer index to cross-attention inputs",
        )
        add_argument(
            "out_feats",
            "--out-feats",
            default=None,
            type=int,
            help="output projection dimension; None or a nonpositive value disables projection",
        )
        add_argument(
            "drop_path_rate",
            "--drop-path-rate",
            default=0.0,
            type=float,
            help="drop path rate",
        )
        add_argument(
            "flash_attention_version",
            "--flash-attention-version",
            default=2,
            type=int,
            choices=[2, 3, 4],
            help="process-wide native Torch Flash Attention version; FA3/FA4 require newer PyTorch and kernel support",
        )
        add_argument(
            "sdp_backend",
            "--sdp-backend",
            default=SDPBackendType.default().value,
            choices=SDPBackendType.choices(),
            help="Preferred sequence of SDP kernels to attempt when calling Torch SDP function",
        )
        add_argument(
            "norm_layer",
            "--norm-layer",
            default=TransformerV2NormLayerType.LAYERNORM.value,
            choices=TransformerV2NormLayerType.choices(),
            help="branch norm type in [layer-norm, rms-norm]; independent of pre_post_norm",
        )
        add_argument(
            "pre_post_norm",
            "--pre-post-norm",
            default=False,
            action=ActionYesNo,
            help="add branch post-norms before residual addition using norm_layer (default: pre-norm only)",
        )
        add_argument(
            "norm_eps",
            "--norm-eps",
            default=1e-5,
            type=float,
            help="epsilon for branch, Q/K/V, and MoE router normalization",
        )
        add_argument(
            "model_parallel",
            "--model-parallel",
            default=False,
            action=ActionYesNo,
            help="use tensor-parallel projections with an externally initialized process group",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments to those handled by `change_config`.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by ``change_config``.

        """
        return filter_func_args(QFormerV2.change_config, kwargs)

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register fine-tuning specific overrides with a JSONArgParse parser.

        Args:
            parser: Target `ArgumentParser` instance.
            prefix: Optional nested argument namespace.
            skip: Optional set of field names to exclude.

        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "override_dropouts" not in skip:
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

        if "drop_path_rate" not in skip:
            try:
                parser.add_argument(
                    "--drop-path-rate",
                    default=0,
                    type=float,
                    help="layer drop probability",
                )
            except Exception:
                pass

        if "att_dropout_rate" not in skip:
            try:
                parser.add_argument(
                    "--att-dropout-rate",
                    default=0,
                    type=float,
                    help="attention layers dropout rate",
                )
            except Exception:
                pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
