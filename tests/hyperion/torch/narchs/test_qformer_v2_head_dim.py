"""Explicit QFormer head widths for self-attention and cross-attention."""

import pytest
import torch
from jsonargparse import ArgumentParser

from hyperion.torch.layers.attention_v2 import ScaledDotProdAttV2
from hyperion.torch.narchs.qformer_v2 import QFormerV2


@pytest.mark.parametrize("head_dim", [None, 6])
@pytest.mark.parametrize("tied_layers", [False, True])
@pytest.mark.parametrize("self_att_k_eq_v", [False, True])
@pytest.mark.parametrize("cross_att_k_eq_v", [False, True])
def test_qformer_head_dim(
    head_dim: int | None,
    tied_layers: bool,
    self_att_k_eq_v: bool,
    cross_att_k_eq_v: bool,
) -> None:
    """Check all branches, output dimensions, gradients, and config restoration.

    Args:
        head_dim: Optional explicit head width.
        tied_layers: Whether to reuse blocks across layers.
        self_att_k_eq_v: Whether self-attention shares the raw key projection.
        cross_att_k_eq_v: Whether cross-attention shares the raw key projection.
    """
    model = QFormerV2(
        in_feats=8,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=2,
        num_layers=4,
        cross_att_freq=2,
        tied_layers=tied_layers,
        head_dim=head_dim,
        self_att_k_eq_v=self_att_k_eq_v,
        cross_att_k_eq_v=cross_att_k_eq_v,
        ff_multiple_of=4,
        rope_in_self_att=True,
        rope_in_cross_att=True,
        rope_scale_freqs=False,
        enable_qk_norm=True,
        enable_v_norm=True,
    ).eval()
    expected_width = 4 if head_dim is None else head_dim
    for attention in model.modules():
        if isinstance(attention, ScaledDotProdAttV2):
            assert attention.head_dim == expected_width
            assert attention.q_norm.dim == expected_width
            assert attention.v_norm.dim == expected_width
            assert attention.q_proj.out_features == 4 * expected_width
            assert attention.o_proj.out_features == 16
    for block in model.trans_blocks:
        assert block.attention.k_eq_v is self_att_k_eq_v
        assert (block.attention.v_proj is None) is self_att_k_eq_v
        if hasattr(block, "cross_attention"):
            assert block.cross_attention.k_eq_v is cross_att_k_eq_v
            assert (block.cross_attention.v_proj is None) is cross_att_k_eq_v
    queries = torch.randn(2, 3, 16, requires_grad=True)
    features = torch.randn(2, 7, 8, requires_grad=True)
    output = model(queries, features)
    assert output.shape == (2, 3, 16)
    (output * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(features.grad).all()
    config = model.get_config(no_class_name=True)
    assert config["head_dim"] == head_dim
    assert config["self_att_k_eq_v"] is self_att_k_eq_v
    assert config["cross_att_k_eq_v"] is cross_att_k_eq_v
    restored = QFormerV2(**config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        restored(queries.detach(), features.detach()), output.detach()
    )


def test_qformer_head_dim_parser() -> None:
    """Check nested CLI options, None default, filtering, and skipping."""
    parser = ArgumentParser()
    QFormerV2.add_class_args(parser, prefix="arch")
    assert parser.parse_args([]).arch.head_dim is None
    assert parser.parse_args(["--arch.head-dim=6"]).arch.head_dim == 6
    assert QFormerV2.filter_args(head_dim=6) == {"head_dim": 6}
    skipped = ArgumentParser()
    QFormerV2.add_class_args(skipped, skip={"head_dim"})
    assert "head_dim" not in skipped.parse_args([]).as_dict()
