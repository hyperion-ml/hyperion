"""Unscaled RMSNorm and optional attention value normalization."""

import pytest
import torch
import torch.nn.functional as F

from hyperion.torch.layers.attention_v2 import (
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)
from hyperion.torch.layers.norm_layers import RMSNorm


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_rms_norm_optional_scale(with_scale: bool, dtype: torch.dtype) -> None:
    """Compare outputs and gradients against the functional RMSNorm reference.

    Args:
        with_scale: Whether normalization learns a scale.
        dtype: Input dtype.
    """
    norm = RMSNorm(8, eps=2e-6, with_scale=with_scale)
    if dtype == torch.float64:
        norm = norm.to(dtype=dtype)
    if with_scale:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    else:
        assert not list(norm.parameters())
        assert "weight" not in norm.state_dict()
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    expected = F.rms_norm(reference_x.to(compute_dtype), (8,), None, 2e-6)
    if with_scale:
        expected = expected * norm.weight.detach().to(compute_dtype)
    expected = expected.to(dtype)
    actual = norm(x)
    torch.testing.assert_close(actual, expected)
    loss_weights = torch.randn_like(actual)
    (actual * loss_weights).sum().backward()
    (expected * loss_weights).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    assert actual.dtype == dtype
    if with_scale:
        assert torch.isfinite(norm.weight.grad).all()
    with pytest.raises(ValueError, match="expected last dimension 8"):
        norm(torch.randn(2, 7))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "xpu"])
@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_rms_norm_layer_norm_dtype_policy(
    device: str, with_scale: bool, autocast_enabled: bool
) -> None:
    """Compare output dtypes with native LayerNorm, including mixed weights.

    Args:
        device: Backend whose autocast policy is checked.
        with_scale: Whether to include learned FP32 weights.
        autocast_enabled: Whether to enable backend autocast.
    """
    if device != "cpu":
        backend = (
            torch.backends.mps if device == "mps" else getattr(torch, device, None)
        )
        if backend is None or not backend.is_available():
            pytest.skip(f"{device} is unavailable")
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    x = torch.randn(2, 3, 8, device=device, dtype=dtype, requires_grad=True)
    norm = RMSNorm(8, with_scale=with_scale).to(device)
    layer_norm = torch.nn.LayerNorm(8, elementwise_affine=with_scale).to(device)
    with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
        actual = norm(x)
        expected_dtype = layer_norm(x).dtype
    assert actual.dtype == expected_dtype
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    if with_scale:
        assert norm.weight.dtype == torch.float32
        assert norm.weight.grad.dtype == torch.float32
        assert torch.isfinite(norm.weight.grad).all()


@pytest.mark.parametrize(
    "attention_class", [ScaledDotProdAttV2, TorchScaledDotProdAttV2]
)
@pytest.mark.parametrize("qk_norm", [False, True])
def test_attention_value_norm_and_cache(attention_class: type, qk_norm: bool) -> None:
    """Verify normalized V reaches attention and cache independently of QK norms.

    Args:
        attention_class: Manual or Torch SDPA backend.
        qk_norm: Whether to also normalize queries and keys.
    """
    options = dict(
        num_feats=16,
        num_heads=4,
        num_kv_heads=2,
        enable_qk_norm=qk_norm,
        norm_eps=2e-6,
        sdp_backend=SDPBackendType.MATH,
    )
    attention = attention_class(**options, enable_v_norm=True)
    baseline = attention_class(**options, enable_v_norm=False)
    baseline.load_state_dict(attention.state_dict())

    def reference_value_projection(module, inputs, output):
        """Normalize each projected value head independently for the reference.

        Args:
            module: Value projection module.
            inputs: Projection inputs.
            output: Flattened projected value heads.

        Returns:
            Normalized flattened value heads.
        """
        heads = output.reshape(*output.shape[:-1], 2, 4)
        return F.rms_norm(heads.float(), (4,), None, 2e-6).to(output.dtype).flatten(-2)

    handle = baseline.v_proj.register_forward_hook(reference_value_projection)
    x = torch.randn(2, 5, 16, requires_grad=True)
    output = attention(x, x, x)
    torch.testing.assert_close(output, baseline(x, x, x))
    (output * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(attention.v_proj.weight.grad).all()
    assert attention.v_proj.weight.grad.abs().sum() > 0
    assert attention.v_norm.weight is None
    assert attention.att_scale == (1.0 if qk_norm else 4**-0.5)
    state = attention.init_state(batch_size=2, max_cache_length=5)
    cached_output, updated = attention(x.detach(), x.detach(), x.detach(), state=state)
    torch.testing.assert_close(cached_output, output.detach())
    projected = attention.v_proj(x.detach()).reshape(2, 5, 2, 4)
    expected_values = F.rms_norm(projected.float(), (4,), None, 2e-6)
    torch.testing.assert_close(updated["value"][:2, :5], expected_values)
    handle.remove()
