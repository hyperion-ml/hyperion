"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_flash_attention_utils import _flash_attention_forward

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
        **kwargs,
    ):
        """Construct a multi-head attention module.

        Args:
            num_feats (int): Input feature dimension (`d_model`).
            num_heads (int): Number of query heads.
            num_kv_feats (Optional[int]): Feature dimension for key/value projections. Defaults to `num_feats`.
            num_kv_heads (Optional[int]): Number of key/value heads (for GQA/MQA). Defaults to `num_heads`.
            dropout_rate (float): Dropout probability applied to attention weights.
            att_bias (bool): Whether projections include a bias term.
            rope (Optional[RotaryPosEncoder]): Rotary positional encoder used before attention.
            is_causal (bool): Whether the module should behave causally (see class docstring for details).
            sliding_window (Optional[int]): Sliding-window size for Flash Attention kernels.
            model_parallel (bool): If `True`, use tensor-parallel linear layers built on PyTorch collectives.

        Returns:
            None: This constructor initializes the module in place.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = num_feats // self.num_heads
        assert num_feats == num_heads * self.head_dim
        self.num_feats = num_feats
        self.num_kv_feats = num_feats if num_kv_feats is None else num_kv_feats
        self.dropout_rate = dropout_rate
        self.rope = rope
        self.is_causal = is_causal
        self.sliding_window = sliding_window
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
            self.k_proj = ColumnParallelLinear(
                self.num_kv_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
                gather_output=False,
            )
            self.v_proj = ColumnParallelLinear(
                self.num_kv_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
                gather_output=False,
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
            self.k_proj = nn.Linear(
                self.num_kv_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
            )
            self.v_proj = nn.Linear(
                self.num_kv_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
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
        """Allocate zero-initialized caches for streaming/key-value reuse.

        Args:
            batch_size (int): Maximum batch size the cache should support.
            max_cache_length (int): Maximum number of cached timesteps.
            device (Optional[torch.device]): Target device. Defaults to projection weight device.
            dtype (Optional[torch.dtype]): Target dtype. Defaults to projection weight dtype.

        Returns:
            CacheState: Dictionary with keys:
                - ``key``: key cache tensor
                - ``value``: value cache tensor
                - ``cache_length``: cached valid length
                - ``cache_offset``: absolute position for cache index 0
        """

        if device is None:
            device = self.q_proj.weight.device
        if dtype is None:
            dtype = self.q_proj.weight.dtype

        cache_shape = (
            batch_size,
            max_cache_length,
            self.num_local_kv_heads,
            self.head_dim,
        )
        cache_k = torch.zeros(cache_shape, device=device, dtype=dtype)
        cache_v = torch.zeros(cache_shape, device=device, dtype=dtype)
        return {
            "key": cache_k,
            "value": cache_v,
            "cache_length": 0,
            "cache_offset": 0,
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
            torch.Tensor: Attention output of shape `(batch, seq_len_q, num_feats)`.
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
            scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
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
            scores = torch.matmul(query, key.transpose(2, 3).unsqueeze(2)) / math.sqrt(
                self.head_dim
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, CacheState]]:
        """Project inputs, optionally apply RoPE, and compute attention.

        Args:
            query (torch.Tensor): Query states `(batch, seq_len_q, num_feats)`.
            key (torch.Tensor): Key states `(batch, seq_len_k, num_feats or num_kv_feats)`.
            value (torch.Tensor): Value states sharing shape with `key`.
            mask (Optional[torch.Tensor]): Optional mask forwarded to `compute_attention`.
                Supports additive float masks and boolean keep masks.
            query_start_pos (int, optional): Starting offset for query rope rotation. Defaults to 0.
            key_start_pos (int, optional): Starting offset for key rope rotation. Defaults to 0.
            state (Optional[CacheState]): External cache with ``key``, ``value``,
                ``cache_length``, and ``cache_offset``.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, CacheState]: Attention output and
                updated cache when ``state`` is provided.
        """
        bsz, q_length, _ = query.size()
        _, k_length, _ = key.size()
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.view(bsz, q_length, self.num_local_heads, self.head_dim)
        key = key.view(bsz, k_length, self.num_local_kv_heads, self.head_dim)
        value = value.view(bsz, k_length, self.num_local_kv_heads, self.head_dim)
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if self.rope is not None:
            query = self.rope(query, query_start_pos)
            key = self.rope(key, key_start_pos)

        query, key, value = self._cast_qkv_for_attention(query, key, value)

        new_state: Optional[CacheState] = None
        if state is not None:
            key, value, new_state = self._update_cache(
                key,
                value,
                state,
                start_pos=key_start_pos,
            )

        output = self.compute_attention(query, key, value, mask)
        output = self.o_proj(output)
        if new_state is not None:
            return output, new_state
        return output

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

    def _update_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: CacheState,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, CacheState]:
        """Align device/dtype and write projections into the external cache.

        Args:
            key (torch.Tensor): Incoming key tensor `(batch, seq_len, kv_heads, head_dim)`.
            value (torch.Tensor): Incoming value tensor `(batch, seq_len, kv_heads, head_dim)`.
            state (CacheState): Cache dictionary with tensors and scalar metadata.
            start_pos (int): Absolute write start position of the incoming chunk.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, CacheState]: Visible cached key/value tensors and updated cache state.
        """

        batch_size = key.size(0)
        cache_k = state["key"]
        cache_v = state["value"]
        cache_length = int(state["cache_length"])
        cache_offset = int(state["cache_offset"])

        if cache_k.device != key.device or cache_k.dtype != key.dtype:
            cache_k = cache_k.to(device=key.device, dtype=key.dtype)
            state["key"] = cache_k
        if cache_v.device != value.device or cache_v.dtype != value.dtype:
            cache_v = cache_v.to(device=value.device, dtype=value.dtype)
            state["value"] = cache_v

        cache_capacity = cache_k.size(1)
        seq_len = key.size(1)
        end_pos = start_pos + seq_len

        current_offset = cache_offset
        current_length = cache_length
        current_end = current_offset + current_length

        # Enforce contiguous updates based on the original incoming start position.
        # This check must happen before we drop tokens that fall outside the cache window.
        if start_pos > current_end:
            raise ValueError(
                "Non-contiguous cache update: "
                f"start_pos ({start_pos}) exceeds current KV cache end "
                f"({current_end})."
            )

        # Determine the absolute window to keep (last `cache_capacity` positions) without rewinding.
        new_offset = max(current_offset, end_pos - cache_capacity)
        drop_from_new = max(0, new_offset - start_pos)
        if drop_from_new > 0:
            key = key[:, drop_from_new:]
            value = value[:, drop_from_new:]
            start_pos = start_pos + drop_from_new

        write_len = key.size(1)
        write_start = start_pos - new_offset

        # Shift existing cache contents to discard old entries.
        shift = max(0, new_offset - current_offset)
        if shift > current_length:
            shift = current_length
        if shift > 0 and current_length > 0:
            keep = max(0, current_length - shift)
            if keep > 0:
                cache_k[:batch_size, :keep] = cache_k[:batch_size, shift : shift + keep]
                cache_v[:batch_size, :keep] = cache_v[:batch_size, shift : shift + keep]
            cache_length = keep
        elif shift > 0:
            keep = 0
            cache_length = 0
        else:
            keep = current_length

        if write_len > 0:
            write_end = write_start + write_len
            if write_end > cache_capacity:
                raise ValueError(
                    f"Attempting to write beyond cache capacity ({write_end}>{cache_capacity})."
                )
            cache_k[:batch_size, write_start:write_end] = key
            cache_v[:batch_size, write_start:write_end] = value
            cache_length = max(keep, write_end)
        else:
            cache_length = keep

        cache_offset = new_offset

        max_length = cache_length
        key = cache_k[:batch_size, :max_length]
        value = cache_v[:batch_size, :max_length]

        state["key"] = cache_k
        state["value"] = cache_v
        state["cache_length"] = cache_length
        state["cache_offset"] = cache_offset

        return key, value, state


class TorchScaledDotProdAttV2(ScaledDotProdAttV2):
    """Scaled dot-product attention backed by PyTorch's fused implementation."""

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
            torch.Tensor: Attention output of shape `(batch, q_len, num_feats)`.
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
            )
        return output.transpose(1, 2).contiguous().view(bsz, q_length, -1)


class HFFlashScaledDotProdAttV2(ScaledDotProdAttV2):
    """Scaled dot-product attention dispatched to Flash Attention kernels."""

    def __init__(self, *args, **kwargs):
        """Create a HuggingFace Flash Attention-backed layer.

        Args:
            *args: Positional arguments forwarded to `ScaledDotProdAttV2`.
            **kwargs: Keyword arguments forwarded to `ScaledDotProdAttV2`.

        Returns:
            None: This constructor initializes the module in place.
        """
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
                use non-negative values as valid tokens.

        Returns:
            torch.Tensor: Attention output of shape `(batch, q_len, num_feats)`.
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
        output = _flash_attention_forward(
            query,
            key,
            value,
            attn_mask,
            q_length,
            dropout=dropout_rate,
            sliding_window=self.sliding_window,
            use_top_left_mask=False,
            is_causal=self.is_causal,
        )
        return output.reshape(bsz, q_length, -1).contiguous()
