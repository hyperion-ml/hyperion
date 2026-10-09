"""Reuse raw key projections as values without reusing processed keys."""

import pytest
import torch
import torch.nn.functional as F

from hyperion.torch.layers.attention_v2 import (
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)
from hyperion.torch.layers.pos_encoder import RotaryPosEncoder


@pytest.mark.parametrize(
    "attention_class", [ScaledDotProdAttV2, TorchScaledDotProdAttV2]
)
@pytest.mark.parametrize("enable_v_norm", [False, True])
@pytest.mark.parametrize("enable_qk_norm", [False, True])
def test_raw_k_projection_reuse(
    attention_class: type, enable_v_norm: bool, enable_qk_norm: bool
) -> None:
    """Compare with independent K/V projections whose weights are equal.

    Args:
        attention_class: Attention backend.
        enable_v_norm: Whether to apply unscaled value RMSNorm.
        enable_qk_norm: Whether to apply learned Q/K normalization.
    """
    options = dict(
        num_feats=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=6,
        enable_qk_norm=enable_qk_norm,
        enable_v_norm=enable_v_norm,
        norm_eps=2e-6,
        sdp_backend=SDPBackendType.MATH,
    )
    shared = attention_class(
        **options, k_eq_v=True, rope=RotaryPosEncoder(scale_freqs=False)
    )
    if enable_qk_norm:
        with torch.no_grad():
            shared.k_norm.weight.copy_(torch.linspace(0.5, 1.5, 6))
    baseline = attention_class(**options, rope=RotaryPosEncoder(scale_freqs=False))
    state_dict = shared.state_dict()
    state_dict["v_proj.weight"] = state_dict["k_proj.weight"].clone()
    baseline.load_state_dict(state_dict)
    assert shared.v_proj is None
    assert not any(name.startswith("v_proj.") for name in shared.state_dict())
    x = torch.randn(2, 5, 16, requires_grad=True)
    # The separate value input is ignored in projection-reuse mode.
    output = shared(x, x, torch.randn_like(x))
    torch.testing.assert_close(output, baseline(x, x, x))
    output.square().sum().backward()
    assert torch.isfinite(shared.k_proj.weight.grad).all()
    assert shared.k_proj.weight.grad.abs().sum() > 0
    cache = shared.init_state(batch_size=2, max_cache_length=5)
    cached_output, updated = shared(x.detach(), x.detach(), x.detach(), state=cache)
    torch.testing.assert_close(cached_output, output.detach())
    raw = shared.k_proj(x.detach()).reshape(2, 5, 2, 6)
    expected_v = F.rms_norm(raw.float(), (6,), None, 2e-6) if enable_v_norm else raw
    expected_k = shared.k_norm(raw) if enable_qk_norm else raw
    expected_k = shared.rope(expected_k)
    torch.testing.assert_close(updated["value"][:2, :5], expected_v)
    torch.testing.assert_close(updated["key"][:2, :5], expected_k)
    assert updated["key"].data_ptr() != updated["value"].data_ptr()
