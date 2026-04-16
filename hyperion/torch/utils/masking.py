"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Union

import torch
import torch.nn as nn


def scale_seq_lengths(
    lengths: Optional[torch.Tensor],
    max_out_length: int,
    max_in_length: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Scale sequence lengths to a different maximum length using floor division.

    Args:
      lengths: Sequence lengths with shape ``(batch,)``. If ``None``, returns ``None``.
      max_out_length: Target maximum length after scaling.
      max_in_length: Source maximum length. If ``None``, uses ``lengths.max()``.

    Returns:
      Scaled lengths tensor with the same shape as ``lengths`` or ``None``.
    """
    if lengths is None:
        return None

    if max_in_length is None:
        max_in_length = int(lengths.max().item())

    if max_in_length == max_out_length:
        return lengths

    return torch.div(lengths * max_out_length, max_in_length, rounding_mode="floor")


def seq_lengths_to_mask(
    lengths: Optional[torch.Tensor],
    max_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    time_dim: int = 1,
    ndim: Optional[int] = None,
    none_if_all_max: bool = False,
) -> Optional[torch.Tensor]:
    """Creates a binary masks indicating the valid values in a sequence.

    Args:
      lengths: sequence lengths with shape=(batch,). If None, it returns None
      max_length: maximum length of the sequence.
      dtype: dtype for the mask.
      time_dim: dimension > 0 corresponding to time in the mask. This will
                return a view of the mask which will adapt to the shape
                of the tensor where we want to apply the mask.
                This has to be a positive integer.
      ndim: number of dimensions in the mask tensor, if None, it is equal to time_dim + 1.
      none_if_all_max: if True and all lengths are equal to max. length, it returns None

    Returns:
      Binary mask with shape=(batch,...,max_length,...) or None
    """
    if lengths is None:
        return None

    assert time_dim > 0, f"time_dim must be a positive intege, got {time_dim}"
    assert lengths.dim() == 1, f"lengths must be a 1D tensor, got {lengths.dim()}D"

    if max_length is None:
        max_length = int(lengths.max().item())

    if none_if_all_max and torch.all(lengths == max_length):
        return None

    idx = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    # compute mask shape=(batch, max_length)
    mask = idx.unsqueeze(0) < lengths.unsqueeze(1)

    if ndim is None:
        ndim = time_dim + 1

    # view to match the tensor where we want to apply the mask
    if ndim > 2:
        shape = [1] * ndim
        shape[0] = lengths.size(0)
        shape[time_dim] = -1
        mask = mask.view(*shape)

    # change dtype if needed
    if dtype is not None:
        mask = mask.to(dtype)

    return mask


def need_attn_mask(
    lengths: Optional[torch.Tensor],
    max_length: Optional[int] = None,
    cache_length: int = 0,
    look_ahead: int = 0,
    is_causal: bool = False,
    is_torch_sdp_attn: bool = False,
    is_hf_flash_attn: bool = False,
    none_if_all_max: bool = True,
) -> bool:
    """Checks if we need to create an attention mask from sequence lengths.

    Args:
      lengths: sequence lengths with shape=(batch,). If None, it returns None
      max_length: maximum length of the sequence.
      cache_length: length of the cache for decoder self-attention.
      look_ahead: number of look-ahead steps for non-causal attention.
      is_causal: whether the attention is causal.
      is_hf_flash_attn: whether we are using Hugging Face flash attention.
      is_torch_sdp_attn: whether we are using torch scaled dot-product attention.
      none_if_all_max: if True and all lengths are equal to max. length, it returns None

    Returns:
      True if we need to create an attention mask, False otherwise.
    """
    assert (
        look_ahead == 0 or not is_causal
    ), "look_ahead is only valid for non-causal attention"
    assert (
        look_ahead == 0 or not is_hf_flash_attn
    ), "look_ahead is not supported with HF flash attention"

    if look_ahead > 0:
        return True

    if max_length is None and lengths is not None:
        max_length = int(lengths.max().item())

    if lengths is None or (none_if_all_max and torch.all(lengths == max_length)):
        if is_causal:
            if is_hf_flash_attn:
                return False
            elif is_torch_sdp_attn and cache_length == 0:
                return False
            else:
                return True
        else:
            return False

    return True


def seq_lengths_to_self_attn_mask(
    lengths: Optional[torch.Tensor],
    max_length: Optional[int] = None,
    cache_length: int = 0,
    look_ahead: int = 0,
    is_causal: bool = False,
    is_hf_flash_attn: bool = False,
    is_torch_sdp_attn: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    none_if_all_max: bool = True,
) -> Optional[torch.Tensor]:
    """Create a self-attention mask from sequence lengths.

    Args:
      lengths: Sequence lengths with shape ``(batch,)``. ``None`` means full-length sequences.
      max_length: Maximum query length. If ``None``, inferred from ``lengths``.
      cache_length: Number of cached key/value frames prepended to current frames.
      look_ahead: Number of future steps allowed in non-causal attention.
      is_causal: Whether to apply a causal mask.
      is_hf_flash_attn: Whether to return a Hugging Face FlashAttention-compatible padding mask.
      is_torch_sdp_attn: Whether the caller uses torch scaled dot-product attention.
      dtype: Output mask dtype.
      device: Device where the mask is created. Defaults to ``lengths.device`` or CPU.
      none_if_all_max: If ``True`` and no mask is needed, return ``None``.

    Returns:
      If ``is_hf_flash_attn`` is ``True``, a padding mask of shape ``(batch, max_length)``.
      Otherwise, a mask broadcastable to ``(batch_or_1, 1, max_q_length, max_kv_length)``,
      or ``None`` when masking is unnecessary.
    """

    assert (
        lengths is None or lengths.dim() == 1
    ), f"lengths must be a 1D tensor, got {lengths.dim()}D"
    if max_length is None and lengths is not None:
        max_length = int(lengths.max().item())

    need_mask = need_attn_mask(
        lengths,
        max_length,
        cache_length,
        look_ahead,
        is_causal,
        is_torch_sdp_attn,
        is_hf_flash_attn,
        none_if_all_max,
    )

    if not need_mask:
        return None
    if max_length is None:
        raise ValueError("max_length must be provided when lengths is None and a mask is required.")

    device = (
        device
        if device is not None
        else lengths.device if lengths is not None else "cpu"
    )

    if is_hf_flash_attn:
        # HF FlashAttention expects a padding mask with shape (batch, seq_length).
        return seq_lengths_to_mask(
            lengths,
            max_length,
            dtype,
            time_dim=1,
            none_if_all_max=none_if_all_max,
        )

    max_kv_length = max_length + cache_length
    masked_value = True if dtype == torch.bool else torch.finfo(dtype).min
    if lengths is None or torch.all(lengths == max_length):
        # we create a broadcastable mask of size (1, 1, max_q_length, max_kv_length)
        # zero means valid position, -inf means invalid position"
        mask = torch.zeros(
            (1, 1, max_length, max_kv_length),
            device=device,
            dtype=dtype,
        )
    else:
        # we create a mask of size (batch, 1, max_q_length, max_kv_length)
        mask = torch.zeros(
            (lengths.size(0), 1, max_length, max_kv_length),
            device=device,
            dtype=dtype,
        )
        batch_lengths = lengths.to(device=device, dtype=torch.long)
        query_idx = torch.arange(max_length, device=device)
        query_mask = query_idx.unsqueeze(0) >= batch_lengths.unsqueeze(1)
        mask = mask.masked_fill(query_mask.unsqueeze(1).unsqueeze(-1), masked_value)

        kv_idx = torch.arange(max_kv_length, device=device)
        kv_mask = kv_idx.unsqueeze(0) >= (batch_lengths + cache_length).unsqueeze(1)
        mask = mask.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), masked_value)

    if look_ahead > 0 or is_causal:
        causal_mask = torch.ones(
            (1, 1, max_length, max_kv_length), device=mask.device, dtype=torch.bool
        )
        torch.triu(causal_mask, diagonal=1 + look_ahead + cache_length, out=causal_mask)
        mask = mask.masked_fill(causal_mask, masked_value)

    return mask


def seq_lengths_to_cross_attn_mask(
    query_lengths: Union[torch.Tensor, None],
    kv_lengths: Union[torch.Tensor, None],
    max_query_length: Optional[int] = None,
    max_kv_length: Optional[int] = None,
    kv_cache_lengths: Optional[torch.Tensor] = None,
    max_kv_cache_length: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    none_if_all_max: bool = True,
) -> Optional[torch.Tensor]:
    """Create attention masks for cross-attention layers given query and key/value lengths.

    Args:
        query_lengths: Tensor of shape ``(batch,)`` with valid query lengths. ``None`` means full-length queries.
        kv_lengths: Tensor of shape ``(batch,)`` with valid key/value lengths. ``None`` means full-length KV sequences.
        max_query_length: Optional maximum query length; inferred from ``query_lengths`` when ``None``.
        max_kv_length: Optional maximum KV length; inferred from ``kv_lengths`` when ``None``.
        kv_cache_lengths: Optional tensor describing cached KV timesteps per batch element.
        max_kv_cache_length: Optional maximum KV cache length; inferred from ``kv_cache_lengths`` when ``None``.
        dtype: Mask dtype to generate (float masks use ``-inf`` for invalid positions).
        device: Device where the mask is allocated.
        none_if_all_max: Return ``None`` when all sequences are full length and no masking is required.

    Returns:
        Additive or boolean mask broadcastable to ``(batch, num_heads, query, key)`` or ``None``.
    """

    if query_lengths is not None:
        assert (
            query_lengths.dim() == 1
        ), f"query_lengths must be a 1D tensor, got {query_lengths.dim()}D"
    if kv_lengths is not None:
        assert (
            kv_lengths.dim() == 1
        ), f"kv_lengths must be a 1D tensor, got {kv_lengths.dim()}D"

    if query_lengths is None and kv_lengths is None:
        return None

    if max_query_length is None:
        assert (
            query_lengths is not None
        ), "max_query_length must be provided if query_lengths is None"
        max_query_length = int(query_lengths.max().item())

    if max_kv_length is None:
        assert (
            kv_lengths is not None
        ), "max_kv_length must be provided if kv_lengths is None"
        max_kv_length = int(kv_lengths.max().item())

    all_queries_full = (
        True if query_lengths is None else torch.all(query_lengths == max_query_length)
    )
    all_kv_full = True if kv_lengths is None else torch.all(kv_lengths == max_kv_length)

    if (
        none_if_all_max
        and all_queries_full
        and all_kv_full
        and kv_cache_lengths is None
    ):
        return None

    if kv_cache_lengths is not None:
        assert (
            kv_cache_lengths.dim() == 1
        ), f"kv_cache_lengths must be a 1D tensor, got {kv_cache_lengths.dim()}D"
        assert (
            kv_lengths is not None
        ), "kv_lengths must be provided when kv_cache_lengths is used."
        assert kv_cache_lengths.size(0) == kv_lengths.size(
            0
        ), "kv_cache_lengths and kv_lengths must have the same batch size."
        kv_cache_lengths = kv_cache_lengths.to(torch.long)
        inferred_max_cache = int(kv_cache_lengths.max().item())
        if max_kv_cache_length is None:
            max_kv_cache_length = inferred_max_cache
        else:
            max_kv_cache_length = int(max_kv_cache_length)
            if max_kv_cache_length < inferred_max_cache:
                raise ValueError(
                    "max_kv_cache_length is smaller than the maximum value in kv_cache_lengths."
                )
    else:
        max_kv_cache_length = (
            0 if max_kv_cache_length is None else int(max_kv_cache_length)
        )

    effective_kv_length = max_kv_length + max_kv_cache_length

    batch_size = (
        query_lengths.size(0)
        if query_lengths is not None
        else kv_lengths.size(0) if kv_lengths is not None else 1
    )
    device = (
        device
        if device is not None
        else (
            query_lengths.device
            if query_lengths is not None
            else kv_lengths.device if kv_lengths is not None else "cpu"
        )
    )

    mask = torch.zeros(
        (
            batch_size,
            1,
            max_query_length,
            effective_kv_length,
        ),
        device=device,
        dtype=dtype,
    )
    if mask.dtype == torch.bool:
        masked_value = True
    elif mask.is_floating_point():
        masked_value = torch.finfo(mask.dtype).min
    else:
        raise TypeError("Mask dtype must be boolean or floating point.")

    if query_lengths is not None and not torch.all(query_lengths == max_query_length):
        q_lengths = query_lengths.to(device=device, dtype=torch.long)
        q_idx = torch.arange(max_query_length, device=device)
        q_mask = q_idx.unsqueeze(0) >= q_lengths.unsqueeze(1)
        mask = mask.masked_fill(q_mask.unsqueeze(1).unsqueeze(-1), masked_value)

    kv_idx = torch.arange(effective_kv_length, device=device)
    if kv_cache_lengths is not None:
        cache_lengths_vec = kv_cache_lengths.to(device=device, dtype=torch.long)
    elif max_kv_cache_length > 0:
        cache_lengths_vec = torch.full(
            (batch_size,),
            max_kv_cache_length,
            device=device,
            dtype=torch.long,
        )
    else:
        cache_lengths_vec = None

    if cache_lengths_vec is not None:
        cache_valid = kv_idx.unsqueeze(0) < cache_lengths_vec.unsqueeze(1)
    else:
        cache_valid = (
            kv_idx.unsqueeze(0) < max_kv_cache_length
            if max_kv_cache_length > 0
            else None
        )

    if kv_lengths is not None:
        kv_lengths = kv_lengths.to(device=device, dtype=torch.long)
        rel_idx = kv_idx - max_kv_cache_length
        current_valid = (kv_idx.unsqueeze(0) >= max_kv_cache_length) & (
            rel_idx.unsqueeze(0) < kv_lengths.unsqueeze(1)
        )
    else:
        current_valid = kv_idx.unsqueeze(0) >= max_kv_cache_length

    if cache_valid is None:
        valid_positions = current_valid
    else:
        valid_positions = cache_valid | current_valid

    kv_mask = ~valid_positions
    mask = mask.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), masked_value)

    return mask


def make_attn_mask_causal(mask: torch.Tensor) -> torch.Tensor:
    """Apply a lower-triangular causal constraint to an attention mask."""
    size = mask.size(-1)
    causal_mask = torch.ones(size, size, device=mask.device, dtype=torch.bool)
    torch.tril(causal_mask, out=causal_mask)
    return mask & causal_mask


def make_dec_causal_att_mask(y: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """Create a causal decoder attention mask from token ids and padding index.

    Args:
      y: Decoder token ids with shape ``(batch, time)``.
      padding_idx: Token id used for padding.

    Returns:
      Boolean mask that combines padding and causal constraints.
    """
    mask = (y != padding_idx).unsqueeze(-2)
    return make_attn_mask_causal(mask)
