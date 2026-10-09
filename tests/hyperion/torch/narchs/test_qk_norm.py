"""QK normalization configuration and gradient checks for V2 architectures."""

import pytest
import torch
from jsonargparse import ArgumentParser

from hyperion.torch.layers.attention_v2 import ScaledDotProdAttV2
from hyperion.torch.layers.norm_layers import RMSNorm
from hyperion.torch.narchs.qformer_v2 import QFormerV2
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _make_architecture(name: str, enabled: bool) -> TransformerEncoderV2 | QFormerV2:
    """Build a small architecture with grouped-query attention.

    Args:
        name: Architecture variant to construct.
        enabled: Whether to enable QK normalization.

    Returns:
        Architecture configured for a CPU forward pass.
    """
    common = dict(
        in_feats=8,
        num_heads=4,
        num_kv_heads=2,
        ff_multiple_of=8,
        enable_qk_norm=enabled,
        norm_eps=2e-6,
        rope_original_max_seq_length=32,
    )
    if name == "encoder":
        return TransformerEncoderV2(
            **common,
            local_rope_scale_freqs=False,
            global_rope_scale_freqs=False,
            stem_type="conv1d",
            stem_hidden_channels=[16],
            stem_kernel_sizes=[3],
            stem_strides=[1],
            stem_dropout_rate=0.0,
            encb_repeats=[2],
            hidden_dims=[16],
        )
    return QFormerV2(
        **common,
        rope_scale_freqs=False,
        num_layers=4,
        hidden_dim=16,
        cross_att_freq=2,
        tied_layers=name == "qformer_tied",
        rope_in_self_att=True,
        rope_in_cross_att=True,
    )


@pytest.mark.parametrize("name", ["encoder", "qformer", "qformer_tied"])
@pytest.mark.parametrize("enabled", [False, True])
def test_qk_norm_architecture(name: str, enabled: bool) -> None:
    """Check all branches, config reconstruction, and gradients through Q/K norms.

    Args:
        name: Architecture variant.
        enabled: Whether to enable QK normalization.
    """
    torch.manual_seed(42)
    model = _make_architecture(name, enabled)
    config = model.get_config(no_class_name=True)
    assert config["enable_qk_norm"] is enabled
    restored = type(model)(**config)
    restored.load_state_dict(model.state_dict())
    attention_layers = [m for m in model.modules() if isinstance(m, ScaledDotProdAttV2)]
    assert (
        len(attention_layers) == {"encoder": 2, "qformer": 6, "qformer_tied": 3}[name]
    )
    for attention in attention_layers:
        assert attention.enable_qk_norm is enabled
        assert attention.att_scale == (1.0 if enabled else attention.head_dim**-0.5)
        if enabled:
            assert isinstance(attention.q_norm, RMSNorm)
            assert isinstance(attention.k_norm, RMSNorm)
            assert attention.q_norm.eps == model.norm_eps
            assert attention.k_norm.eps == model.norm_eps
        else:
            assert attention.q_norm is None
            assert attention.k_norm is None
    feats = torch.randn(2, 7, 8)
    if name == "encoder":
        output, _ = model(feats)
    else:
        output = model(torch.randn(2, 3, 16), feats)
    assert torch.isfinite(output).all()
    # A weighted loss avoids cancellation from the final LayerNorm.
    (output * torch.randn_like(output)).sum().backward()
    if enabled:
        for attention in attention_layers:
            for norm in (attention.q_norm, attention.k_norm):
                assert norm.weight.grad is not None
                assert torch.isfinite(norm.weight.grad).all()
                assert norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("architecture", [TransformerEncoderV2, QFormerV2])
def test_qk_norm_parser(architecture: type) -> None:
    """Check defaults, nested CLI options, filtering, and parser skip support.

    Args:
        architecture: Architecture class exposing parser helpers.
    """
    parser = ArgumentParser()
    architecture.add_class_args(parser, prefix="arch")
    assert parser.parse_args([]).arch.enable_qk_norm is False
    assert parser.parse_args(["--arch.enable-qk-norm"]).arch.enable_qk_norm is True
    assert architecture.filter_args(enable_qk_norm=True) == {"enable_qk_norm": True}
    skipped_parser = ArgumentParser()
    architecture.add_class_args(skipped_parser, skip={"enable_qk_norm"})
    assert "enable_qk_norm" not in skipped_parser.parse_args([]).as_dict()
