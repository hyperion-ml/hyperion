"""Consume source-layer K/V without repeating their preparation."""

import pytest
import torch
import torch.nn.functional as F

from hyperion.torch.layers.attention_v2 import (
    HFFlashScaledDotProdAttV2,
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)
from hyperion.torch.layers.pos_encoder import RotaryPosEncoder


@pytest.mark.parametrize(
    "attention_class", [ScaledDotProdAttV2, TorchScaledDotProdAttV2]
)
@pytest.mark.parametrize("enable_qk_norm", [False, True])
@pytest.mark.parametrize("k_eq_v", [False, True])
def test_shared_kv_output_and_source_gradients(
    attention_class: type, enable_qk_norm: bool, k_eq_v: bool
) -> None:
    """Check custom-width GQA against SDPA and retain gradients through the source.

    Args:
        attention_class: Attention backend.
        enable_qk_norm: Whether queries and source keys receive learned RMSNorm.
        k_eq_v: Whether the source uses its raw key projection for values.
    """
    options = dict(
        num_feats=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=6,
        enable_qk_norm=enable_qk_norm,
        enable_v_norm=True,
        k_eq_v=k_eq_v,
        sdp_backend=SDPBackendType.MATH,
    )
    source = attention_class(**options, rope=RotaryPosEncoder(scale_freqs=False))
    consumer = attention_class(
        **options, shared_kv=True, rope=RotaryPosEncoder(scale_freqs=False)
    )
    assert consumer.k_proj is consumer.v_proj is None
    assert consumer.k_norm is consumer.v_norm is None
    assert (consumer.q_norm is not None) == enable_qk_norm
    q = torch.randn(2, 3, 16, requires_grad=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    _, key, value, _ = source._prepare_qkv(x, x, x, 7, 7, None)
    original_key, original_value = key.detach().clone(), value.detach().clone()
    reference_q = consumer.q_proj(q).reshape(2, 3, 4, 6)
    if enable_qk_norm:
        reference_q = consumer.q_norm(reference_q)
    reference_q = consumer.rope(reference_q, 9)
    reference = F.scaled_dot_product_attention(
        reference_q.transpose(1, 2),
        key.transpose(1, 2).repeat_interleave(2, dim=1),
        value.transpose(1, 2).repeat_interleave(2, dim=1),
        scale=consumer.att_scale,
    )
    reference = consumer.o_proj(reference.transpose(1, 2).reshape(2, 3, 24))
    output = consumer(q, key, value, query_start_pos=9, key_start_pos=100)
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(key, original_key)
    torch.testing.assert_close(value, original_value)
    output.square().sum().backward()
    for parameter in [consumer.q_proj.weight, source.k_proj.weight]:
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert x.grad is not None
    if not k_eq_v:
        assert source.v_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "attention_class",
    [ScaledDotProdAttV2, TorchScaledDotProdAttV2, HFFlashScaledDotProdAttV2],
)
def test_shared_kv_validation_and_cache_ownership(attention_class: type) -> None:
    """All backends inherit shared preparation and reject owned caches.

    Args:
        attention_class: Attention backend.
    """
    att = attention_class(16, 4, num_kv_heads=2, shared_kv=True)
    q = torch.randn(2, 3, 16)
    key = torch.randn(2, 5, 2, 4, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    prepared_q, prepared_k, prepared_v = att._prepare_shared_qkv(q, key, value, 0)
    assert prepared_k is key and prepared_v is value
    assert prepared_k.dtype == prepared_v.dtype == prepared_q.dtype
    (prepared_k.sum() + prepared_v.sum()).backward()
    assert key.grad is not None and value.grad is not None
    with pytest.raises(ValueError, match="cannot allocate"):
        att.init_state(2, 5)
    with pytest.raises(ValueError, match="cannot update"):
        att(q, key, value, state={})
    for shape in [(2, 5, 16), (1, 5, 2, 4), (2, 5, 4, 4), (2, 5, 2, 6)]:
        invalid = torch.randn(shape)
        with pytest.raises(ValueError, match="Shared K/V"):
            att._prepare_shared_qkv(q, invalid, invalid, 0)
    with pytest.raises(ValueError, match="matching"):
        att._prepare_shared_qkv(q, key, value[:, :4], 0)
    with pytest.raises(ValueError, match="device and dtype"):
        att._prepare_shared_qkv(q, key.double(), value.double(), 0)
