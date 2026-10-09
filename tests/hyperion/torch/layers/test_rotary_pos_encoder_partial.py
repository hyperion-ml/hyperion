"""Proportional RoPE and frequency-scaling cache regressions."""

import math

import pytest
import torch

from hyperion.torch.layers.pos_encoder import RotaryPosEncoder


def _reference(
    x: torch.Tensor, theta: float, fraction: float, start_pos: int = 0
) -> torch.Tensor:
    """Evaluate interleaved proportional rotations without using the RoPE cache.

    Args:
        x: Input features.
        theta: Frequency base.
        fraction: Rotated fraction.
        start_pos: First position.

    Returns:
        Independently rotated features.
    """
    d = x.shape[-1]
    pairs = int(fraction * d // 2)
    result = x.clone()
    positions = torch.arange(start_pos, start_pos + x.shape[1], dtype=torch.float32)[
        None, :, None
    ]
    for i in range(pairs):
        angle = positions * theta ** (-2 * i / d)
        result[..., 2 * i] = (
            x[..., 2 * i] * angle.cos() - x[..., 2 * i + 1] * angle.sin()
        )
        result[..., 2 * i + 1] = (
            x[..., 2 * i] * angle.sin() + x[..., 2 * i + 1] * angle.cos()
        )
    return result


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25])
def test_proportional_frequency_spacing_and_identity(fraction: float) -> None:
    """Verify full-head frequency spacing, unchanged pairs, and gradients.

    Args:
        fraction: Fraction of head dimensions to rotate.
    """
    x = torch.randn(2, 7, 3, 16, requires_grad=True)
    rope = RotaryPosEncoder(
        theta=1000000, scale_freqs=False, partial_rotary_factor=fraction
    )
    output = rope(x, start_pos=3)
    torch.testing.assert_close(output, _reference(x, 1000000, fraction, 3))
    rotated_dim = 2 * int(fraction * 16 // 2)
    torch.testing.assert_close(
        output[..., rotated_dim:], x[..., rotated_dim:], rtol=0, atol=0
    )
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_scaling_in_training_and_reference_growth() -> None:
    """Scaling remains active in training and rebuilds after reference growth."""
    rope = RotaryPosEncoder(
        original_max_seq_length=8, scale_freqs=True, update_max_seq_length=True
    )
    frequencies = 2 * math.pi / torch.tensor([1.0, 4.0, 16.0])
    expected = frequencies * torch.tensor([1.0, 5 / 12, 1 / 8])
    torch.testing.assert_close(rope._scale_freqs(frequencies), expected)
    x = torch.randn(1, 4, 2, 16)
    training_output = rope(x)
    rope.eval()
    torch.testing.assert_close(rope(x), training_output)
    rope.train()
    longer = torch.randn(1, 32, 2, 16)
    output = rope(longer)
    assert rope.max_seq_length.item() == 32
    fixed = RotaryPosEncoder(original_max_seq_length=32, update_max_seq_length=False)
    torch.testing.assert_close(output, fixed(longer))
    torch.testing.assert_close(rope(x), fixed(x))


def test_cache_changes_dimensions_settings_and_loaded_reference() -> None:
    """Cached rotations are rebuilt after widths, settings, and state changes."""
    rope = RotaryPosEncoder(original_max_seq_length=16)
    rope(torch.randn(1, 6, 1, 8))
    x = torch.randn(1, 4, 1, 16)
    torch.testing.assert_close(rope(x), RotaryPosEncoder(original_max_seq_length=16)(x))
    rope.partial_rotary_factor = 0.5
    torch.testing.assert_close(
        rope(x),
        RotaryPosEncoder(original_max_seq_length=16, partial_rotary_factor=0.5)(x),
    )
    fixed = RotaryPosEncoder(original_max_seq_length=32, partial_rotary_factor=0.5)
    rope.load_state_dict(fixed.state_dict())
    torch.testing.assert_close(rope(x), fixed(x))


@pytest.mark.parametrize(
    "options",
    [
        dict(theta=0),
        dict(partial_rotary_factor=0),
        dict(partial_rotary_factor=1.5),
        dict(scaling_factor=0),
        dict(low_freq_factor=4, high_freq_factor=1),
    ],
)
def test_invalid_rope_options(options: dict) -> None:
    """Reject invalid frequency configurations.

    Args:
        options: Invalid constructor settings.
    """
    with pytest.raises(ValueError):
        RotaryPosEncoder(**options)
