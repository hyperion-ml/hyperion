"""Local/global scheduling, positional encoding, and configuration round trips."""

import inspect

import pytest
import torch
from jsonargparse import ArgumentParser

from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _encoder(**kwargs) -> TransformerEncoderV2:
    """Construct a tiny encoder with changing widths and temporal resolution.

    Args:
        **kwargs: Configuration overrides.

    Returns:
        Encoder for CPU checks.
    """
    options = dict(
        in_feats=8,
        stem_type="conv1d",
        stem_hidden_channels=[8],
        stem_kernel_sizes=[3],
        stem_strides=[1],
        stem_dropout_rate=0,
        encb_repeats=[2, 3],
        hidden_dims=[8, 16],
        downb_strides=[2],
        num_heads=2,
        ff_multiple_of=4,
        local_to_global_ratio=2,
        local_rope_scale_freqs=False,
        global_rope_scale_freqs=False,
    )
    options.update(kwargs)
    return TransformerEncoderV2(**options)


def test_schedule_across_stages_and_roundtrip() -> None:
    """Check schedule, stage-specific caches, gradients, and restored weights."""
    model = _encoder(global_rope_partial_rotary_factor=0.5).eval()
    assert model.layer_types == ["local", "local", "global", "local", "global"]
    assert model.trans_blocks[0][0].attention.rope is model.local_rope[0]
    assert model.trans_blocks[1][0].attention.rope is model.global_rope[1]
    assert model.local_rope[0] is not model.local_rope[1]
    x = torch.randn(2, 20, 8, requires_grad=True)
    y, lengths = model(x, torch.tensor([20, 17]))
    assert lengths is not None
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    restored = TransformerEncoderV2(**model.get_config(no_class_name=True)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x.detach(), torch.tensor([20, 17]))[0], y)
    assert restored.layer_types == model.layer_types


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0, ["global"] * 5),
        (4, ["local"] * 4 + ["global"]),
        (8, ["local"] * 4 + ["global"]),
    ],
)
def test_ratio_and_final_layer(ratio: int, expected: list[str]) -> None:
    """Check all-global and incomplete final schedule groups.

    Args:
        ratio: Number of local layers per global layer.
        expected: Expected flattened schedule.
    """
    assert _encoder(local_to_global_ratio=ratio).layer_types == expected


def test_cli_defaults_and_nested_roundtrip() -> None:
    """Verify defaults and all new nested parser options agree with constructors."""
    parser = ArgumentParser()
    TransformerEncoderV2.add_class_args(parser, prefix="encoder")
    defaults = parser.parse_args([]).encoder.as_dict()
    names = [
        "local_attention_sliding_window",
        "global_attention_sliding_window",
        "local_to_global_ratio",
        "local_rope_theta",
        "global_rope_theta",
        "local_rope_partial_rotary_factor",
        "global_rope_partial_rotary_factor",
        "local_rope_scale_freqs",
        "global_rope_scale_freqs",
    ]
    signature = inspect.signature(TransformerEncoderV2)
    for name in names:
        assert defaults[name] == signature.parameters[name].default
    args = parser.parse_args(
        [
            "--encoder.local-to-global-ratio",
            "4",
            "--encoder.local-attention-sliding-window",
            "32",
            "--encoder.global-attention-sliding-window",
            "128",
            "--encoder.global-rope-partial-rotary-factor",
            "0.25",
        ]
    )
    config = args.as_dict()
    config["encoder"]["local_rope_scale_freqs"] = False
    config["encoder"]["global_rope_scale_freqs"] = False
    args = parser.parse_object(config)
    filtered = TransformerEncoderV2.filter_args(**args.encoder.as_dict())
    assert filtered["local_to_global_ratio"] == 4
    assert filtered["global_rope_partial_rotary_factor"] == 0.25
    assert filtered["local_rope_scale_freqs"] is False
    assert filtered["global_rope_scale_freqs"] is False
    assert all(name in filtered for name in names)


@pytest.mark.parametrize(
    "options",
    [
        dict(local_to_global_ratio=-1),
        dict(local_to_global_ratio=True),
        dict(local_attention_sliding_window=0),
        dict(local_attention_sliding_window=32),
        dict(local_attention_sliding_window=32, global_attention_sliding_window=16),
    ],
)
def test_invalid_schedules_and_windows(options: dict) -> None:
    """Reject invalid schedules or unsupported finite-window backends.

    Args:
        options: Invalid constructor settings.
    """
    with pytest.raises(ValueError):
        _encoder(**options)


def test_finite_window_routing() -> None:
    """Check local and larger global windows reach the native Flash backend."""
    model = _encoder(
        att_type="hf_flash_sdp",
        local_attention_sliding_window=4,
        global_attention_sliding_window=12,
    )
    windows = [
        block.attention.sliding_window
        for stage in model.trans_blocks
        for block in stage
    ]
    assert windows == [4, 4, 12, 4, 12]
    assert model.global_attention_sliding_window == 12
    restored = TransformerEncoderV2(**model.get_config(no_class_name=True))
    assert [
        block.attention.sliding_window
        for stage in restored.trans_blocks
        for block in stage
    ] == windows


def test_local_global_head_dims() -> None:
    """Custom widths preserve residual shapes, norm widths, and saved configuration."""
    model = _encoder(
        local_head_dim=6,
        global_head_dim=12,
        global_rope_partial_rotary_factor=0.25,
        enable_qk_norm=True,
        enable_v_norm=True,
    ).eval()
    attentions = [block.attention for stage in model.trans_blocks for block in stage]
    assert [att.head_dim for att in attentions] == [6, 6, 12, 6, 12]
    for att in attentions:
        assert att.q_norm.dim == att.head_dim
        assert att.v_norm.dim == att.head_dim
        assert att.o_proj.out_features == att.num_feats
    x = torch.randn(2, 20, 8)
    output, _ = model(x)
    assert output.shape[-1] == 16
    restored = TransformerEncoderV2(**model.get_config(no_class_name=True)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x)[0], output)
    parser = ArgumentParser()
    TransformerEncoderV2.add_class_args(parser)
    args = parser.parse_args(["--local-head-dim=6", "--global-head-dim=12"])
    assert TransformerEncoderV2.filter_args(**args.as_dict())["global_head_dim"] == 12


def test_stage_derived_head_dims() -> None:
    """Default None widths are derived independently at each encoder stage."""
    model = _encoder()
    assert model.local_head_dim is None and model.global_head_dim is None
    assert [
        block.attention.head_dim for stage in model.trans_blocks for block in stage
    ] == [4, 4, 8, 8, 8]


def test_global_k_eq_v_routing() -> None:
    """Only global layers reuse projections; preserve the option on reconstruction."""
    model = _encoder(global_k_eq_v=True)
    attentions = [block.attention for stage in model.trans_blocks for block in stage]
    assert [att.k_eq_v for att in attentions] == [False, False, True, False, True]
    assert [att.v_proj is None for att in attentions] == [
        False,
        False,
        True,
        False,
        True,
    ]
    config = model.get_config(no_class_name=True)
    assert config["global_k_eq_v"] is True
    restored = TransformerEncoderV2(**config)
    restored.load_state_dict(model.state_dict())
    assert [
        block.attention.k_eq_v for stage in restored.trans_blocks for block in stage
    ] == [False, False, True, False, True]
