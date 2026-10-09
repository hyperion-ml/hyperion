"""Attention projection widths independent of the residual stream."""

import pytest
import torch
import torch.nn.functional as F

from hyperion.torch.layers.attention_v2 import (
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)


@pytest.mark.parametrize(
    "attention_class", [ScaledDotProdAttV2, TorchScaledDotProdAttV2]
)
@pytest.mark.parametrize("head_dim", [2, 8])
def test_head_dim_projection_attention_and_cache(
    attention_class: type, head_dim: int
) -> None:
    """Check GQA output against an independent SDPA reference with custom widths.

    Args:
        attention_class: Attention backend.
        head_dim: Explicit dimension per head.
    """
    att = attention_class(
        num_feats=15,
        num_heads=3,
        num_kv_heads=1,
        num_kv_feats=10,
        head_dim=head_dim,
        sdp_backend=SDPBackendType.MATH,
    )
    q = torch.randn(2, 4, 15, requires_grad=True)
    kv = torch.randn(2, 6, 10, requires_grad=True)
    assert att.q_proj.out_features == 3 * head_dim
    assert att.k_proj.out_features == head_dim
    assert att.v_proj.out_features == head_dim
    assert att.o_proj.in_features == 3 * head_dim
    projected_q = att.q_proj(q).reshape(2, 4, 3, head_dim).transpose(1, 2)
    projected_k = (
        att.k_proj(kv)
        .reshape(2, 6, 1, head_dim)
        .transpose(1, 2)
        .repeat_interleave(3, dim=1)
    )
    projected_v = (
        att.v_proj(kv)
        .reshape(2, 6, 1, head_dim)
        .transpose(1, 2)
        .repeat_interleave(3, dim=1)
    )
    reference = F.scaled_dot_product_attention(projected_q, projected_k, projected_v)
    reference = att.o_proj(reference.transpose(1, 2).reshape(2, 4, 3 * head_dim))
    output = att(q, kv, kv)
    assert output.shape == (2, 4, 15)
    torch.testing.assert_close(output, reference)
    output.square().sum().backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(kv.grad).all()
    state = att.init_state(batch_size=2, max_cache_length=6)
    cached_output, updated = att(q.detach(), kv.detach(), kv.detach(), state=state)
    assert updated["key"].shape == (2, 6, 1, head_dim)
    torch.testing.assert_close(cached_output, output.detach())


def test_head_dim_fallback_and_validation() -> None:
    """Retain derived widths and reject invalid explicit dimensions."""
    assert ScaledDotProdAttV2(num_feats=16, num_heads=4).head_dim == 4
    with pytest.raises(ValueError, match="divisible"):
        ScaledDotProdAttV2(num_feats=15, num_heads=4)
    for head_dim in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="head_dim"):
            ScaledDotProdAttV2(num_feats=16, num_heads=4, head_dim=head_dim)
