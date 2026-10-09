"""Padding isolation and gradient coverage for encoder endpoints."""

import pytest
import torch

from hyperion.torch.layer_blocks.transformer_v2 import TransformerV2ConvEndpoint
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


@pytest.mark.parametrize("in_scale,out_scale", [(1, 4), (1, 3), (3, 1)])
@pytest.mark.parametrize("mask_kind", ["2d", "4d", "additive"])
def test_endpoint_padding_isolation(
    in_scale: int, out_scale: int, mask_kind: str
) -> None:
    """Compare padded endpoint outputs against processing only valid frames.

    Args:
        in_scale: Input temporal scale.
        out_scale: Target temporal scale.
        mask_kind: Padding mask representation.
    """
    torch.manual_seed(7)
    block = TransformerV2ConvEndpoint(4, 4, in_scale, out_scale).eval()
    with torch.no_grad():
        block.norm.bias.fill_(2.0)
        block.resample[0].weight.fill_(-0.5)
        block.resample[0].bias.fill_(-1.0)
    valid_input = torch.randn(1, 6, 4)
    padded = torch.cat((valid_input, torch.full((1, 6, 4), 100.0)), dim=1)
    keep = torch.arange(12)[None, :] < 6
    mask = keep
    if mask_kind != "2d":
        mask = keep[:, None, None, :].expand(-1, 1, 12, -1)
    if mask_kind == "additive":
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
            ~mask, -float("inf")
        )
    expected = block(valid_input)
    actual = block(padded, mask)
    torch.testing.assert_close(actual[:, : expected.size(1)], expected)
    assert torch.count_nonzero(actual[:, expected.size(1) :]) == 0
    empty = block(padded, torch.zeros_like(keep))
    assert torch.isfinite(empty).all()
    assert torch.count_nonzero(empty) == 0


@pytest.mark.parametrize("concat", [False, True])
def test_final_endpoint_keeps_all_stages_in_gradient_graph(concat: bool) -> None:
    """Keep final-stage gradients when the caller selects only an early endpoint.

    Args:
        concat: Whether to concatenate endpoint features instead of averaging.
    """
    model = TransformerEncoderV2(
        in_feats=4,
        stem_type="conv1d",
        stem_hidden_channels=[4],
        stem_kernel_sizes=[3],
        stem_strides=[1],
        stem_dropout_rate=0.0,
        encb_repeats=[1, 1],
        hidden_dims=[4, 4],
        downb_strides=[2],
        num_heads=2,
        ff_multiple_of=4,
        local_rope_scale_freqs=False,
        global_rope_scale_freqs=False,
        multilayer=True,
        multilayer_concat=concat,
        endpoint_layers=[0],
    )
    assert model.endpoint_layers == [0, 1]
    assert model.get_config(no_class_name=True)["endpoint_layers"] == [0, 1]
    output, lengths = model(torch.randn(2, 12, 4), torch.tensor([12, 8]))
    output.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert lengths is not None and (lengths <= output.size(1)).all()
    assert model.out_shape((2, 12, 4))[1] == output.size(1)
