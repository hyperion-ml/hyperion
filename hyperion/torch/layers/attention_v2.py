"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging.version import Version
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .norm_layers import RMSNorm
from .pos_encoder import RotaryPosEncoder
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    get_tensor_parallel_world_size,
)

if TYPE_CHECKING:
    from enum import Enum as _SDPBackendEnum
else:
    _SDPBackendEnum = SDPBackend

SDPBackendReturn = Union[_SDPBackendEnum, List[_SDPBackendEnum]]
CacheState = Dict[str, Union[torch.Tensor, int]]


class SDPBackendType(str, Enum):
    MATH = "math"
    FLASH = "flash"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"
    FLASH_EFFICIENT = "flash->efficient"
    CUDNN_EFFICIENT = "cudnn->efficient"
    FLASH_CUDNN_EFFICIENT = "flash->cudnn->efficient"
    FLASH_EFFICIENT_CUDNN = "flash->efficient->cudnn"

    @staticmethod
    def choices() -> List[str]:
        return [e.value for e in SDPBackendType]

    @staticmethod
    def to_backend(
        value: "SDPBackendType",
    ) -> SDPBackendReturn:
        if value == SDPBackendType.MATH:
            return SDPBackend.MATH
        elif value == SDPBackendType.FLASH:
            return [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        elif value == SDPBackendType.EFFICIENT:
            return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        elif value == SDPBackendType.CUDNN:
            return [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
        elif value == SDPBackendType.FLASH_EFFICIENT:
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        elif value == SDPBackendType.CUDNN_EFFICIENT:
            return [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        elif value == SDPBackendType.FLASH_CUDNN_EFFICIENT:
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        elif value == SDPBackendType.FLASH_EFFICIENT_CUDNN:
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.MATH,
            ]
        else:
            raise ValueError(f"Unknown SDPBackendType: {value}")

    @staticmethod
    def default() -> "SDPBackendType":
        return SDPBackendType.FLASH_EFFICIENT_CUDNN


class ScaledDotProdAttV2(nn.Module):
    """Scaled dot-product attention with optional rotary embeddings and cache-aware projections.

    Attributes:
        num_feats (int): Input feature dimension.
        num_heads (int): Number of query attention heads.
        num_kv_feats (int): Key/value feature dimension.
        num_kv_heads (int): Number of key/value heads (can differ from `num_heads`).
        shared_kv (bool): Consume processed K/V supplied by a source layer; compute only Q.
        k_eq_v (bool): Reuse the raw key projection as values; omit the separate value projection.
        head_dim (int): Dimension per head after projection.
        dropout_rate (float): Dropout probability applied to attention weights.
        rope (Optional[RotaryPosEncoder]): Rotary positional encoder to inject rope phases.
        is_causal (bool): Flag indicating whether the attention should behave causally.
            * In the base implementation this flag is not applied—callers must encode causality in the mask they pass.
            * In `TorchScaledDotProdAttV2` the flag is honored only when no mask is supplied; as soon as a mask is provided it is assumed to encode any causal or padding constraints.
            * In `HFFlashScaledDotProdAttV2` the flag always enforces a causal triangle in addition to any user-provided mask.
        sliding_window (Optional[int]): Size of the sliding window for flash attention.
        num_local_heads (int): Number of heads handled by the local rank in model-parallel mode.
        num_local_kv_heads (int): Number of kv heads handled by the local rank.
        num_rep (int): Replication factor between attention heads and kv heads.
        enable_v_norm (bool): Whether values receive per-head RMSNorm without a learned scale.
        v_norm (Optional[RMSNorm]): Value normalization over head dimensions without learned scaling.
        enable_qk_norm (bool): Normalize queries and keys per head before RoPE.
        norm_eps (float): Epsilon used by query/key/value RMS normalization.
        q_norm (Optional[RMSNorm]): Learned query normalization over head dimensions.
        k_norm (Optional[RMSNorm]): Learned key normalization over head dimensions.
        att_scale (float): Attention score multiplier; one when QK normalization is enabled.
    """

    def __init__(
        self,
        num_feats: int,
        num_heads: int,
        num_kv_feats: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        dropout_rate: float = 0.0,
        att_bias: bool = False,
        rope: Optional[RotaryPosEncoder] = None,
        is_causal: bool = False,
        sliding_window: Optional[int] = None,
        model_parallel: bool = False,
        enable_qk_norm: bool = False,
        enable_v_norm: bool = False,
        norm_eps: float = 1e-6,
        head_dim: Optional[int] = None,
        k_eq_v: bool = False,
        shared_kv: bool = False,
        **kwargs,
    ):
        """Construct a multi-head attention module.

        Args:
            num_feats (int): Input feature dimension (`d_model`).
            num_heads (int): Number of query heads.
            shared_kv (bool): Consume processed K/V without K/V projections, norms,
                RoPE, or cache writes. Query preparation remains enabled.
            k_eq_v (bool): Reuse the raw key projection for V; the value input is ignored when enabled.
            head_dim (Optional[int]): Positive head width; None derives num_feats / num_heads.
            num_kv_feats (Optional[int]): Feature dimension for key/value projections. Defaults to `num_feats`.
            num_kv_heads (Optional[int]): Number of key/value heads (for GQA/MQA). Defaults to `num_heads`.
            dropout_rate (float): Dropout probability applied to attention weights.
            att_bias (bool): Whether projections include a bias term.
            rope (Optional[RotaryPosEncoder]): Rotary positional encoder used before attention.
            is_causal (bool): Whether the module should behave causally (see class docstring for details).
            sliding_window (Optional[int]): Sliding-window size for Flash Attention kernels.
            model_parallel (bool): If `True`, use tensor-parallel linear layers built on PyTorch collectives.
            enable_v_norm (bool): Normalize values per head without learned scaling; independent of QK normalization.
            enable_qk_norm (bool): Enable learned per-head RMSNorm for Q and K and unit attention scaling.
            norm_eps (float): Epsilon for query/key/value RMSNorm. Defaults to `1e-6`.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
        ):
            raise ValueError("num_heads must be a positive integer")
        if head_dim is None:
            if num_feats % num_heads != 0:
                raise ValueError(
                    "num_feats must be divisible by num_heads when head_dim is None"
                )
            head_dim = num_feats // num_heads
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer or None")
        if rope is not None and head_dim % 2:
            raise ValueError("head_dim must be even when using RoPE")
        self.head_dim = head_dim
        self.shared_kv = shared_kv
        self.k_eq_v = k_eq_v
        self.num_feats = num_feats
        self.num_kv_feats = num_feats if num_kv_feats is None else num_kv_feats
        self.dropout_rate = dropout_rate
        self.rope = rope
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.enable_v_norm = enable_v_norm
        self.v_norm = (
            RMSNorm(self.head_dim, eps=norm_eps, with_scale=False)
            if enable_v_norm and not shared_kv
            else None
        )
        self.enable_qk_norm = enable_qk_norm
        self.norm_eps = norm_eps
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if enable_qk_norm else None
        self.k_norm = (
            RMSNorm(self.head_dim, eps=norm_eps)
            if enable_qk_norm and not shared_kv
            else None
        )
        self.att_scale = 1.0 if enable_qk_norm else self.head_dim**-0.5
        self._warned_qkv_cast_from_fp32 = False

        if model_parallel:
            model_parallel_size = get_tensor_parallel_world_size()
            self.num_local_heads = num_heads // model_parallel_size
            self.num_local_kv_heads = self.num_kv_heads // model_parallel_size
            self.num_rep = self.num_local_heads // self.num_local_kv_heads

            self.q_proj = ColumnParallelLinear(
                self.num_feats,
                self.num_heads * self.head_dim,
                bias=att_bias,
                gather_output=False,
            )
            self.k_proj = (
                None
                if shared_kv
                else ColumnParallelLinear(
                    self.num_kv_feats,
                    self.num_kv_heads * self.head_dim,
                    bias=att_bias,
                    gather_output=False,
                )
            )
            self.v_proj = (
                None
                if shared_kv or k_eq_v
                else ColumnParallelLinear(
                    self.num_kv_feats,
                    self.num_kv_heads * self.head_dim,
                    bias=att_bias,
                    gather_output=False,
                )
            )
            self.o_proj = RowParallelLinear(
                num_heads * self.head_dim,
                num_feats,
                bias=att_bias,
                input_is_parallel=True,
            )

        else:
            self.num_local_heads = num_heads
            self.num_local_kv_heads = self.num_kv_heads
            self.num_rep = self.num_local_heads // self.num_local_kv_heads

            self.q_proj = nn.Linear(
                self.num_feats,
                self.num_heads * self.head_dim,
                bias=att_bias,
            )
            self.k_proj = (
                None
                if shared_kv
                else nn.Linear(
                    self.num_kv_feats,
                    self.num_kv_heads * self.head_dim,
                    bias=att_bias,
                )
            )
            self.v_proj = (
                None
                if shared_kv or k_eq_v
                else nn.Linear(
                    self.num_kv_feats,
                    self.num_kv_heads * self.head_dim,
                    bias=att_bias,
                )
            )
            self.o_proj = nn.Linear(
                num_heads * self.head_dim,
                num_feats,
                bias=att_bias,
            )

        self._assert_args()

    def _assert_args(self):
        assert self.sliding_window is None, "Base class does not support sliding_window"
        if self.num_local_heads % self.num_local_kv_heads != 0:
            raise ValueError(
                f"num_local_heads ({self.num_local_heads}) must be divisible by "
                f"num_local_kv_heads ({self.num_local_kv_heads})"
            )

    @torch.no_grad()
    def init_state(
        self,
        batch_size: int,
        max_cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CacheState:
        """Initialize empty dynamic caches and their maximum retention length.

        Args:
            batch_size (int): Batch dimension of the initial empty cache tensors.
            max_cache_length (int): Maximum number of cached timesteps, capped at
                sliding_window when the layer has a finite attention window.
            device (Optional[torch.device]): Target device. Defaults to projection weight device.
            dtype (Optional[torch.dtype]): Target dtype. Defaults to projection weight dtype.

        Returns:
            CacheState: Dictionary with keys:
                - ``key``: key cache tensor
                - ``value``: value cache tensor
                - ``cache_length``: cached valid length
                - ``cache_offset``: absolute position for cache index 0
                - ``max_cache_length``: maximum retained history, including the window cap

        Raises:
            ValueError: Shared-KV layers cannot allocate their own cache, or the
                requested retention limit is negative.
        """

        if self.shared_kv:
            raise ValueError(
                "Shared-KV attention uses its source layer cache; it cannot allocate its own cache"
            )
        if device is None:
            device = self.q_proj.weight.device
        if dtype is None:
            dtype = self.q_proj.weight.dtype

        if self.sliding_window is not None:
            max_cache_length = min(max_cache_length, self.sliding_window)
        if max_cache_length < 0:
            raise ValueError("max_cache_length must be non-negative")
        cache_shape = (
            batch_size,
            0,
            self.num_local_kv_heads,
            self.head_dim,
        )
        cache_k = torch.empty(cache_shape, device=device, dtype=dtype)
        cache_v = torch.empty(cache_shape, device=device, dtype=dtype)
        return {
            "key": cache_k,
            "value": cache_v,
            "cache_length": 0,
            "cache_offset": 0,
            "max_cache_length": max_cache_length,
        }

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand key/value heads to match the number of attention heads.

        Args:
            x (torch.Tensor): Tensor shaped `(batch, seq_len, kv_heads, head_dim)`.

        Returns:
            torch.Tensor: Tensor shaped `(batch, seq_len, num_heads, head_dim)` after replication.
        """
        if self.num_rep == 1:
            return x

        bsz, seq_length, num_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(
            bsz, seq_length, num_kv_heads, self.num_rep, head_dim
        )
        return x.reshape(bsz, seq_length, num_kv_heads * self.num_rep, head_dim)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention outputs using explicit batched matrix multiplications.

        Args:
            query (torch.Tensor): Query tensor of shape `(batch, seq_len_q, heads, head_dim)`.
            key (torch.Tensor): Key tensor of shape `(batch, seq_len_k, kv_heads, head_dim)`.
            value (torch.Tensor): Value tensor matching the key shape.
            mask (Optional[torch.Tensor]): Optional mask. Supports broadcastable additive
                float masks and boolean keep masks (`True` keeps, `False` masks).

        Returns:
            torch.Tensor: Attention output of shape `(batch, seq_len_q, local_heads * head_dim)`.
        """
        assert (
            not self.is_causal or mask is not None
        ), "Causality must be enforced via the mask in the base implementation."
        bsz, q_length, num_heads, _ = query.size()
        k_length = key.size(1)
        kv_heads = key.size(2)
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                assert (
                    mask.shape[0] == bsz
                ), f"mask batch axis ({mask.shape[0]}) must match batch size ({bsz})"
                assert (
                    mask.shape[-1] >= k_length
                ), f"mask key axis ({mask.shape[-1]}) must be >= k_length ({k_length})"
                attn_mask = mask[:, None, None, :k_length]
            else:
                assert mask.dim() >= 2, "mask must have at least 2 dimensions"
                assert (
                    mask.shape[-2] >= q_length
                ), f"mask query axis ({mask.shape[-2]}) must be >= q_length ({q_length})"
                assert (
                    mask.shape[-1] >= k_length
                ), f"mask key axis ({mask.shape[-1]}) must be >= k_length ({k_length})"
                attn_mask = mask[..., :q_length, :k_length]
            if attn_mask.dtype == torch.bool:
                min_value = torch.finfo(query.dtype).min
                attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(
                    ~attn_mask, min_value
                )
            else:
                assert torch.is_floating_point(
                    attn_mask
                ), "ScaledDotProdAttV2 expects float additive masks or bool masks."

        query = query.transpose(1, 2)  # (bsz, q_heads, query_len, head_dim)
        key = key.transpose(1, 2)  # (bs, kv_heads, key_len, head_dim)
        value = value.transpose(1, 2)  # (bs, kv_heads, key_len, head_dim)

        if num_heads == kv_heads:
            scores = torch.matmul(query, key.transpose(2, 3)) * self.att_scale
            if attn_mask is not None:
                scores = scores + attn_mask
            scores = nn.functional.softmax(scores.float(), dim=-1).type_as(query)
            if self.dropout_rate > 0.0:
                scores = nn.functional.dropout(
                    scores, p=self.dropout_rate, training=self.training
                )
            output = torch.matmul(scores, value)
        else:
            if num_heads % kv_heads != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})"
                )
            num_rep = num_heads // kv_heads
            query = query.reshape(bsz, kv_heads, num_rep, q_length, self.head_dim)
            # scores = torch.einsum("bgrqd,bgkd->bgrqk", query, key) / math.sqrt(
            #     self.head_dim
            # )
            scores = (
                torch.matmul(query, key.transpose(2, 3).unsqueeze(2)) * self.att_scale
            )
            # scores = (bsz, kv_heads, num_rep, query_len, key_len)
            if attn_mask is not None:
                while attn_mask.dim() < scores.dim():
                    attn_mask = attn_mask.unsqueeze(1)
                scores = scores + attn_mask
            scores = nn.functional.softmax(scores.float(), dim=-1).type_as(query)
            if self.dropout_rate > 0.0:
                scores = nn.functional.dropout(
                    scores, p=self.dropout_rate, training=self.training
                )
            # output = torch.einsum("bgrqk,bgkd->bgrqd", scores, value)
            output = torch.matmul(
                scores, value.unsqueeze(2)
            )  # (bsz, kv_heads, num_rep, query_len, head_dim)
            output = output.reshape(bsz, num_heads, q_length, self.head_dim)

        return output.transpose(1, 2).contiguous().view(bsz, q_length, -1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query_start_pos: int = 0,
        key_start_pos: int = 0,
        state: Optional[CacheState] = None,
        return_kv: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, CacheState],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[CacheState]],
    ]:
        """Project inputs, optionally apply RoPE, and compute attention.

        Args:
            query (torch.Tensor): Query states `(batch, seq_len_q, num_feats)`.
            key (torch.Tensor): Hidden states `(batch, seq_len_k, num_kv_feats)`, or processed
                keys `(batch, seq_len_k, local_kv_heads, head_dim)` in shared mode.
            value (torch.Tensor): Same shape as key. In shared mode these are processed values.
                Ignored when k_eq_v=True only in independent mode.
            mask (Optional[torch.Tensor]): Optional mask forwarded to `compute_attention`.
                Supports additive float masks and boolean keep masks.
            query_start_pos (int, optional): Starting offset for query rope rotation. Defaults to 0.
            key_start_pos (int, optional): Starting offset for key rope rotation. Defaults to 0.
                Ignored in shared mode because source keys are already rotated.
            state (Optional[CacheState]): External cache with ``key``, ``value``,
                ``cache_length``, ``cache_offset``, and ``max_cache_length``. Must be None in shared mode;
                source layers own cache updates. Callers supply masks for source cache offsets.

            return_kv: Return the full processed K/V used for attention, including
                cached history and the entire current chunk before retention.
                Defaults to False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, CacheState]: Attention output and
                updated cache when ``state`` is provided. With return_kv=True,
                returns (output, key, value, updated_state), where updated_state
                is None when caching is disabled.
        """
        if self.shared_kv:
            if state is not None:
                raise ValueError(
                    "Shared-KV attention cannot update a cache; pass source K/V directly"
                )
            query, key, value = self._prepare_shared_qkv(
                query, key, value, query_start_pos
            )
            new_state = None
        else:
            query, key, value, new_state = self._prepare_qkv(
                query, key, value, query_start_pos, key_start_pos, state
            )

        output = self.compute_attention(query, key, value, mask)
        output = self.o_proj(output)
        if new_state is not None:
            cache_start = (
                min(int(new_state["cache_offset"]), key_start_pos)
                if int(new_state["cache_length"]) > 0
                else key_start_pos
            )
            self._update_cache(key, value, new_state, cache_start)
        if return_kv:
            return output, key, value, new_state
        if new_state is not None:
            return output, new_state
        return output

    def _prepare_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_pos: int,
        key_start_pos: int,
        state: Optional[CacheState],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[CacheState]]:
        """Prepare independent Q/K/V with cached history and the full current chunk.

        Args:
            query: Query hidden states (batch, query_time, num_feats).
            key: Key hidden states (batch, key_time, num_kv_feats).
            value: Value hidden states; ignored when k_eq_v is enabled.
            query_start_pos: Absolute query position for RoPE.
            key_start_pos: Absolute key position for RoPE and cache writes.
            state: Optional cache owned by this layer.

        Returns:
            Prepared Q/K/V and source cache to update after attention, or None.
        """
        bsz, q_length, _ = query.size()
        _, k_length, _ = key.size()
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = key if self.k_eq_v else self.v_proj(value)

        query = query.view(bsz, q_length, self.num_local_heads, self.head_dim)
        key = key.view(bsz, k_length, self.num_local_kv_heads, self.head_dim)
        value = value.view(bsz, k_length, self.num_local_kv_heads, self.head_dim)
        if self.v_norm is not None:
            value = self.v_norm(value).type_as(value)
        if self.enable_qk_norm:
            query = self.q_norm(query).type_as(query)
            key = self.k_norm(key).type_as(key)
        if self.rope is not None:
            query = self.rope(query, query_start_pos)
            key = self.rope(key, key_start_pos)

        query, key, value = self._cast_qkv_for_attention(query, key, value)

        new_state: Optional[CacheState] = None
        if state is not None:
            key, value = self._prepare_cached_kv(key, value, state, key_start_pos)
            new_state = state

        return query, key, value, new_state

    def _prepare_shared_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q and consume source K/V without reprocessing or cache writes.

        Args:
            query: Query hidden states (batch, query_time, num_feats).
            key: Processed source keys (batch, key_time, local_kv_heads, head_dim).
            value: Processed source values with the same shape as key.
            query_start_pos: Absolute query position for RoPE.

        Returns:
            Prepared Q and source K/V in the attention compute dtype.
        """
        if query.ndim != 3:
            raise ValueError("Shared-KV query must have shape (batch, time, num_feats)")
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError(
                "Shared K/V must have matching (batch, time, kv_heads, head_dim) shapes"
            )
        if key.shape[0] != query.shape[0] or key.shape[2:] != (
            self.num_local_kv_heads,
            self.head_dim,
        ):
            raise ValueError(
                "Shared K/V batch size, KV head count, or head dimension does not match this layer"
            )
        if not torch.is_floating_point(key) or not torch.is_floating_point(value):
            raise ValueError("Shared K/V must be floating-point tensors")
        bsz, q_length, _ = query.shape
        query = self.q_proj(query).view(
            bsz, q_length, self.num_local_heads, self.head_dim
        )
        if self.q_norm is not None:
            query = self.q_norm(query).type_as(query)
        if self.rope is not None:
            query = self.rope(query, query_start_pos)
        query, key, value = self._cast_qkv_for_attention(query, key, value)
        if any(
            t.device != query.device or t.dtype != query.dtype for t in (key, value)
        ):
            raise ValueError(
                "Shared K/V must match the query attention device and dtype"
            )
        return query, key, value

    def _cast_qkv_for_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast Q/K/V from fp32 to the active low-precision compute dtype when available.

        Args:
            query (torch.Tensor): Query tensor with shape `(batch, q_len, heads, head_dim)`.
            key (torch.Tensor): Key tensor with shape `(batch, k_len, kv_heads, head_dim)`.
            value (torch.Tensor): Value tensor with shape `(batch, k_len, kv_heads, head_dim)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Possibly cast query, key, and value tensors.
        """
        if query.dtype != torch.float32:
            return query, key, value
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_proj.weight.dtype
        if target_dtype == torch.float32:
            return query, key, value
        if not self._warned_qkv_cast_from_fp32:
            logging.warning(
                "The input hidden states seem to be silently casted in float32, this might be related to "
                "upcasted embedding or layer norm layers in float32. We will cast back the input to %s.",
                target_dtype,
            )
            self._warned_qkv_cast_from_fp32 = True
        return query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)

    def _prepare_cached_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: CacheState,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge cached history and the full chunk without modifying the cache.

        Args:
            key: Current processed keys (batch, time, kv_heads, head_dim).
            value: Current processed values with the same shape as key.
            state: Source cache before this call.
            start_pos: Absolute position of the first current key.

        Returns:
            Full attention K/V, including any cached suffix after an overwrite.
        """
        cache_offset = int(state["cache_offset"])
        cache_length = int(state["cache_length"])
        cache_end = cache_offset + cache_length
        end_pos = start_pos + key.size(1)
        if start_pos > cache_end or (cache_length > 0 and end_pos < cache_offset):
            raise ValueError(
                "Non-contiguous cache update: current chunk and cached keys must "
                "overlap or be adjacent"
            )
        if cache_length == 0:
            return key, value

        prefix_length = max(0, start_pos - cache_offset)
        suffix_start = max(0, end_pos - cache_offset)
        batch_size = key.size(0)
        cache_k = state["key"][:batch_size, :cache_length].to(key)
        cache_v = state["value"][:batch_size, :cache_length].to(value)
        key = torch.cat(
            [cache_k[:, :prefix_length], key, cache_k[:, suffix_start:]], dim=1
        )
        value = torch.cat(
            [cache_v[:, :prefix_length], value, cache_v[:, suffix_start:]], dim=1
        )
        return key, value

    def _update_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: CacheState,
        start_pos: int,
    ) -> None:
        """Retain suffix views of the full K/V after attention has completed.

        Args:
            key: Full keys used for this attention call.
            value: Full values used for this attention call.
            state: Cache dictionary containing a separate max_cache_length limit.
            start_pos: Absolute position of the first full-attention key.
        """
        end_pos = start_pos + key.size(1)
        cache_offset = max(
            int(state["cache_offset"]), end_pos - int(state["max_cache_length"])
        )
        keep_start = cache_offset - start_pos
        state["key"] = key[:, keep_start:]
        state["value"] = value[:, keep_start:]
        state["cache_length"] = end_pos - cache_offset
        state["cache_offset"] = cache_offset


class TorchScaledDotProdAttV2(ScaledDotProdAttV2):
    """Scaled dot-product attention backed by PyTorch's fused implementation."""

    @staticmethod
    def set_flash_attention_version(flash_attention_version: int = 2) -> None:
        """Select the process-wide native SDPA Flash Attention implementation.

        Args:
            flash_attention_version: Requested version (2, 3, or 4). FA2 is
                unchanged on PyTorch <= 2.9.1; newer versions use the registry.
                Other SDPA backends are unaffected.
        """
        if (
            isinstance(flash_attention_version, bool)
            or not isinstance(flash_attention_version, int)
            or flash_attention_version not in (2, 3, 4)
        ):
            raise ValueError("flash_attention_version must be 2, 3, or 4")
        if Version(torch.__version__.split("+")[0]) <= Version("2.9.1"):
            if flash_attention_version != 2:
                raise RuntimeError("Native FA3/FA4 selection requires PyTorch > 2.9.1")
            return
        attention = torch.nn.attention
        if not hasattr(attention, "activate_flash_attention_impl"):
            raise RuntimeError(
                "This PyTorch build lacks Flash Attention implementation selection"
            )
        current = attention.current_flash_attention_impl()
        requested = (
            None if flash_attention_version == 2 else f"FA{flash_attention_version}"
        )
        if current == requested:
            return
        if requested is None:
            if not hasattr(attention, "restore_flash_attention_impl"):
                raise RuntimeError("Restoring native FA2 requires PyTorch >= 2.11")
            attention.restore_flash_attention_impl()
        else:
            if requested not in attention.list_flash_attention_impls():
                raise RuntimeError(f"{requested} is unavailable in this PyTorch build")
            attention.activate_flash_attention_impl(requested)

    def __init__(
        self,
        *args,
        sdp_backend: SDPBackendType = SDPBackendType.default(),
        **kwargs,
    ):
        """Create a PyTorch SDPA-backed attention layer.

        Args:
            sdp_backend (SDPBackendType): Preferred sequence of SDP kernels to attempt when
                calling `torch.nn.functional.scaled_dot_product_attention`.

        Returns:
            None: This constructor initializes the module in place.
        """

        super().__init__(*args, **kwargs)
        backend = SDPBackendType.to_backend(sdp_backend)
        self._sdp_backends = backend

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Delegate to PyTorch's `scaled_dot_product_attention` implementation.

        This implementation uses `enable_gqa` when `num_heads != num_kv_heads`,
        which requires PyTorch 2.5 or newer.

        Args:
            query (torch.Tensor): Query tensor of shape `(batch, q_len, heads, head_dim)`.
            key (torch.Tensor): Key tensor of shape `(batch, k_len, kv_heads, head_dim)`.
            value (torch.Tensor): Value tensor of shape `(batch, k_len, kv_heads, head_dim)`.
            mask (Optional[torch.Tensor]): Optional attention mask. Supports
                `(batch, k_len)` key-padding masks or broadcastable attention masks,
                in either boolean or floating-point form.

        Returns:
            torch.Tensor: Attention output of shape `(batch, q_len, local_heads * head_dim)`.
        """
        # Input q, k, v = (batch, length, num_heads, head_dim)
        bsz, q_length, num_heads, _ = query.size()
        k_length = key.size(1)
        kv_heads = key.size(2)
        query = query.transpose(1, 2)  # (bsz, heads, query_len head_dim)
        key = key.transpose(1, 2)  # (bs, kv_heads, cache_len + key_len, head_dim)
        value = value.transpose(1, 2)  # (bs, kv_heads, cache_len + key_len, head_dim)

        attn_mask = mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                assert (
                    attn_mask.shape[0] == bsz
                ), f"mask batch axis ({attn_mask.shape[0]}) must match batch size ({bsz})"
                assert (
                    attn_mask.shape[-1] >= k_length
                ), f"mask key axis ({attn_mask.shape[-1]}) must be >= k_length ({k_length})"
                attn_mask = attn_mask[:, None, None, :k_length]
            else:
                assert attn_mask.dim() >= 2, "mask must have at least 2 dimensions"
                assert (
                    attn_mask.shape[-2] >= q_length
                ), f"mask query axis ({attn_mask.shape[-2]}) must be >= q_length ({q_length})"
                assert (
                    attn_mask.shape[-1] >= k_length
                ), f"mask key axis ({attn_mask.shape[-1]}) must be >= k_length ({k_length})"
                attn_mask = attn_mask[..., :q_length, :k_length]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query.device.type == "cuda" and attn_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        assert (
            not self.is_causal or q_length == key.size(-2) or attn_mask is not None
        ), (
            "Causality must be enforced via the mask when the key length differs from "
            "the query length in the TorchScaledDotProdAttV2 implementation."
        )
        if num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})"
            )
        is_causal = self.is_causal if attn_mask is None and q_length > 1 else False

        with sdpa_kernel(self._sdp_backends):
            output = nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=(num_heads != kv_heads),
                scale=self.att_scale,
            )
        return output.transpose(1, 2).contiguous().view(bsz, q_length, -1)


class HFFlashScaledDotProdAttV2(ScaledDotProdAttV2):
    """Scaled dot-product attention dispatched to Flash Attention kernels.

    Attributes:
        flash_attention_version: Requested Flash Attention version (2, 3, or 4).
        attn_implementation: HuggingFace implementation name.
        is_causal: Whether to apply causal attention, inherited from the base.
        sliding_window: Optional local attention window, inherited from the base.
    """

    def __init__(self, *args, flash_attention_version: int = 4, **kwargs):
        """Create a HuggingFace Flash Attention-backed layer.

        Args:
            *args: Positional arguments forwarded to `ScaledDotProdAttV2`.
            flash_attention_version: Flash Attention version, one of 2, 3, or 4.
            **kwargs: Keyword arguments forwarded to `ScaledDotProdAttV2`.
        """
        if (
            isinstance(flash_attention_version, bool)
            or not isinstance(flash_attention_version, int)
            or flash_attention_version not in (2, 3, 4)
        ):
            raise ValueError("flash_attention_version must be 2, 3, or 4")
        self.flash_attention_version = flash_attention_version
        self.attn_implementation = f"flash_attention_{flash_attention_version}"
        super().__init__(*args, **kwargs)

    def _assert_args(self):
        if self.num_local_heads % self.num_local_kv_heads != 0:
            raise ValueError(
                f"num_local_heads ({self.num_local_heads}) must be divisible by "
                f"num_local_kv_heads ({self.num_local_kv_heads})"
            )

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Use Flash Attention kernels via HuggingFace utilities.

        This path keeps key/value heads unexpanded and relies on backend GQA support.

        Args:
            query (torch.Tensor): Query tensor of shape `(batch, q_len, heads, head_dim)`.
            key (torch.Tensor): Key tensor of shape `(batch, k_len, kv_heads, head_dim)`.
            value (torch.Tensor): Value tensor of shape `(batch, k_len, kv_heads, head_dim)`.
            mask (Optional[torch.Tensor]): Optional key-padding mask with shape `(batch, k_len)`
                where boolean masks are interpreted as keep masks, and numeric masks
                use non-negative values as valid tokens. Non-causal cross-attention
                also accepts a (batch, 1, query_len, key_len) padding-only mask
                whose rows are identical; arbitrary pairwise masks are unsupported.

        Returns:
            torch.Tensor: Attention output of shape `(batch, q_len, local_heads * head_dim)`.
        """
        # Input q, k, v = (batch, length, num_heads, head_dim)
        # Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]
        bsz, q_length, num_heads, _ = query.size()
        k_length = key.size(1)
        kv_heads = key.size(2)
        if num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})"
            )
        attn_mask = None
        cross_padding_mask = mask is not None and mask.dim() == 4
        if cross_padding_mask:
            if self.is_causal or mask.size(1) != 1 or mask.size(2) < q_length:
                raise ValueError(
                    "Cross-attention masks must have shape (batch, 1, query_len, key_len) and be non-causal"
                )
            keep_mask = mask if mask.dtype == torch.bool else mask >= 0
            if not torch.equal(
                keep_mask[:, :, :q_length],
                keep_mask[:, :, :1].expand(-1, -1, q_length, -1),
            ):
                raise ValueError(
                    "Flash cross-attention supports only key-padding masks shared by all queries"
                )
            mask = keep_mask[:, 0, 0]
        if mask is not None:
            assert (
                mask.dim() == 2
            ), "HFFlashScaledDotProdAttV2 expects mask with shape (batch, k_len)."
            assert (
                mask.shape[0] == bsz
            ), f"mask batch axis ({mask.shape[0]}) must match batch size ({bsz})"
            assert (
                mask.shape[-1] >= k_length
            ), f"mask key axis ({mask.shape[-1]}) must be >= k_length ({k_length})"
            attn_mask = mask[:, :k_length]
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask >= 0

        dropout_rate = self.dropout_rate if self.training else 0.0
        if (
            attn_mask is not None
            and not self.is_causal
            and (cross_padding_mask or q_length != k_length)
        ):
            return self._compute_cross_attention(
                query, key, value, attn_mask, dropout_rate
            )
        output = _flash_attention_forward(
            query,
            key,
            value,
            attn_mask,
            q_length,
            dropout=dropout_rate,
            softmax_scale=self.att_scale,
            sliding_window=self.sliding_window,
            use_top_left_mask=False,
            is_causal=self.is_causal,
            attn_implementation=self.attn_implementation,
        )
        return output.reshape(bsz, q_length, -1).contiguous()

    def _compute_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
        dropout_rate: float,
    ) -> torch.Tensor:
        """Attend from fully valid queries to independently padded keys.

        Args:
            query: Query states shaped (batch, query_len, heads, head_dim).
            key: Key states shaped (batch, key_len, kv_heads, head_dim).
            value: Value states with the same layout as key.
            key_mask: Boolean key validity shaped (batch, key_len).
            dropout_rate: Attention dropout probability.

        Returns:
            Attention output shaped (batch, query_len, heads * head_dim).
        """
        batch_size, query_length, num_heads, head_dim = query.shape
        lengths = key_mask.sum(-1, dtype=torch.int32)
        active = lengths > 0
        output = torch.zeros_like(query)
        if query_length == 0 or not active.any():
            return output.reshape(batch_size, query_length, num_heads * head_dim)
        active_lengths = lengths[active]
        active_batch = active_lengths.numel()
        cu_q = (
            torch.arange(active_batch + 1, device=query.device, dtype=torch.int32)
            * query_length
        )
        cu_k = torch.cat(
            (active_lengths.new_zeros(1), active_lengths.cumsum(0, dtype=torch.int32))
        )
        # HF reshapes outputs using the input batch dimension. Pack into one
        # outer batch and retain the actual example boundaries in cu_q/cu_k.
        packed_output = _flash_attention_forward(
            query[active].reshape(1, -1, num_heads, head_dim),
            key[key_mask].unsqueeze(0),
            value[key_mask].unsqueeze(0),
            attention_mask=None,
            query_length=query_length,
            is_causal=False,
            dropout=dropout_rate,
            softmax_scale=self.att_scale,
            cu_seq_lens_q=cu_q,
            cu_seq_lens_k=cu_k,
            sliding_window=self.sliding_window,
            max_length_q=query_length,
            max_length_k=int(active_lengths.max().item()),
            attn_implementation=self.attn_implementation,
        )
        output = output.index_copy(
            0,
            active.nonzero(as_tuple=True)[0],
            packed_output.reshape(active_batch, query_length, num_heads, head_dim),
        )
        return output.reshape(
            batch_size, query_length, num_heads * head_dim
        ).contiguous()
