"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import List, Optional, Tuple, Type, Union

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear


class ScaledDotProdAttV2(nn.Module):
    def __init__(
        self,
        num_feats: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate=0.0,
        use_cache: bool = False,
        internal_cache: bool = True,
        max_batch_size: int = 0,
        max_seq_length: int = 0,
        att_bias: bool = False,
        rope=None,
        is_causal: bool = False,
        model_parallel: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = num_feats // self.num_heads
        assert num_feats == num_feats * self.head_dim
        self.num_feats = num_feats
        self.dropout_rate = dropout_rate
        self.use_cache = use_cache
        self.internal_cache = internal_cache
        self.rope = rope
        self.is_causal = is_causal

        if model_parallel:
            model_parallel_size = fs_init.get_model_parallel_world_size()
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
                self.num_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
                gather_output=False,
            )
            self.v_proj = ColumnParallelLinear(
                self.num_feats,
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
                self.num_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
            )
            self.v_proj = nn.Linear(
                self.num_feats,
                self.num_kv_heads * self.head_dim,
                bias=att_bias,
            )
            self.o_proj = nn.Linear(
                num_heads * self.head_dim,
                num_feats,
                bias=att_bias,
            )

        if use_cache and internal_cache:
            self.cache_k = torch.zeros(
                (
                    max_batch_size,
                    max_seq_length,
                    self.num_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    max_batch_size,
                    max_seq_length,
                    self.num_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()

    def _repeat_kv(self, x):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
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
    ):
        bsz, num_heads, q_length, head_dim = query.size()
        query = query.transpose(1, 2)  # (bsz, heads, query_len head_dim)
        key = key.transpose(1, 2)  # (bs, heads, cache_len + key_len, head_dim)
        value = value.transpose(1, 2)  # (bs, heads, cache_len + key_len, head_dim)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(query)
        if self.dropout_rate > 0.0:
            scores = nn.functional.dropout(
                scores, p=self.dropout_rate, training=self.training
            )

        output = torch.matmul(scores, value)  # (bs, n_local_heads, seqlen, head_dim)
        return output.transpose(1, 2).contiguous().view(bsz, q_length, -1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query_start_pos: int = 0,
        key_start_pos: int = 0,
    ):
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

        if self.use_cache:
            assert self.internal_cache
            self.cache_k = self.cache_k.to(query)
            self.cache_v = self.cache_v.to(query)

            self.cache_k[:bsz, key_start_pos : key_start_pos + k_length] = key
            self.cache_v[:bsz, key_start_pos : key_start_pos + k_length] = value

            key = self.cache_k[:bsz, : key_start_pos + k_length]
            value = self.cache_v[:bsz, : key_start_pos + k_length]

        key = self._repeat_kv(key)  # (bsz, key_len, heads, head_dim)
        value = self._repeat_kv(value)

        output = self.compute_attention(query, key, value, mask)
        return self.o_proj(output)


class TorchScaledDotProdAttV2(ScaledDotProdAttV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Input q, k, v = (batch, length, num_heads, head_dim)
        bsz, q_length, _, _ = query.size()
        query = query.transpose(1, 2)  # (bsz, heads, query_len head_dim)
        key = key.transpose(1, 2)  # (bs, heads, cache_len + key_len, head_dim)
        value = value.transpose(1, 2)  # (bs, heads, cache_len + key_len, head_dim)

        # don't know if we need this that is in hf implementation
        # causal_mask = attention_mask
        # if attention_mask is not None:
        #     causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query.device.type == "cuda" and mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = self.is_causal if mask is None and q_length > 1 else False

        output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=is_causal,
        )
        return output.transpose(1, 2).contiguous().view(bsz, q_length, -1)


class FlashScaledDotProdAttV2(ScaledDotProdAttV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Input q, k, v = (batch, length, num_heads, head_dim)
        # Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]
        bsz, q_length, _, _ = query.size()

        # TODO look into this form hf code:
        # # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # # therefore the input hidden states gets silently casted in float32. Hence, we need
        # # cast them back in the correct dtype just to be sure everything works as expected.
        # # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # # in fp32. (LlamaRMSNorm handles it correctly)

        # input_dtype = query_states.dtype
        # if input_dtype == torch.float32:
        #     if torch.is_autocast_enabled():
        #         target_dtype = torch.get_autocast_gpu_dtype()
        #     # Handle the case where the model is quantized
        #     elif hasattr(self.config, "_pre_quantization_dtype"):
        #         target_dtype = self.config._pre_quantization_dtype
        #     else:
        #         target_dtype = self.q_proj.weight.dtype

        #     logger.warning_once(
        #         f"The input hidden states seems to be silently casted in float32, this might be related to"
        #         f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #         f" {target_dtype}."
        #     )

        # query_states = query_states.to(target_dtype)
        # key_states = key_states.to(target_dtype)
        # value_states = value_states.to(target_dtype)

        dropout_rate = self.dropout_rate if self.training else 0.0
        output = _flash_attention_forward(
            query,
            key,
            value,
            mask,
            q_length,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=False,
            is_causal=self.is_causal,
        )
        return output.reshape(bsz, q_length, -1).contiguous()
