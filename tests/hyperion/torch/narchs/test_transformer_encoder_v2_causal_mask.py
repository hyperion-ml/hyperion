"""Causal masks account for padding, source caches, and cache rollover."""

from unittest.mock import patch

import pytest
import torch

from hyperion.torch.layers.attention_v2 import SDPBackendType, TorchScaledDotProdAttV2
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _encoder(**kwargs) -> TransformerEncoderV2:
    """Build a small encoder with pointwise convolutions.

    Args:
        **kwargs: Constructor settings to override.

    Returns:
        Encoder without convolutional look-ahead in its first stage.
    """
    options = dict(
        in_feats=8,
        stem_type="conv1d",
        stem_hidden_channels=[8],
        stem_kernel_sizes=[1],
        stem_strides=[1],
        stem_dropout_rate=0.0,
        hidden_dims=[8],
        encb_repeats=[6],
        num_heads=2,
        num_kv_heads=1,
        local_to_global_ratio=1,
        num_kv_shared_layers=2,
        ff_multiple_of=4,
        is_causal=True,
        sdp_backend=SDPBackendType.MATH,
        local_rope_scale_freqs=False,
        global_rope_scale_freqs=False,
        rope_update_max_seq_length=False,
    )
    options.update(kwargs)
    return TransformerEncoderV2(**options).eval()


@pytest.mark.parametrize("att_type", ["sdp", "torch_sdp", "hf_flash_sdp"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_mask_shapes_and_cached_padding(att_type: str, is_causal: bool) -> None:
    """Check backend mask formats and current padding before cache truncation.

    Args:
        att_type: Attention backend identifier.
        is_causal: Whether to combine an absolute-position causal constraint.
    """
    model = _encoder(att_type=att_type, is_causal=is_causal)
    x = torch.randn(2, 2, 8)
    lengths = torch.tensor([2, 1])
    cache = model.trans_blocks[0][0].init_state(2, 4)
    cache["cache_length"] = 4
    # Attention sees positions 0 through 5; only 2 through 5 are retained.
    mask = model._make_attention_mask(x, lengths, 4, cache)
    key_valid = torch.tensor([[True] * 6, [True] * 5 + [False]])
    if not is_causal or att_type == "hf_flash_sdp":
        torch.testing.assert_close(mask, key_valid)
    else:
        expected = torch.tensor([[True] * 5 + [False], [True] * 6])
        torch.testing.assert_close(
            mask, expected[None, None] & key_valid[:, None, None]
        )
    if not is_causal or att_type == "hf_flash_sdp":
        assert model._make_attention_mask(x, None, 0, None) is None
    elif att_type == "torch_sdp":
        assert model._make_attention_mask(x, None, 0, None) is None
    else:
        torch.testing.assert_close(
            model._make_attention_mask(x, None, 0, None),
            torch.ones(1, 1, 2, 2, dtype=torch.bool).tril(),
        )


@pytest.mark.parametrize("att_type", ["sdp", "torch_sdp"])
def test_causal_prefix_ignores_future_with_lengths(att_type: str) -> None:
    """Future valid frames and padded frames cannot affect an earlier prefix.

    Args:
        att_type: Manual or Torch attention backend.
    """
    model = _encoder(att_type=att_type)
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    lengths = torch.tensor([6, 4])
    with torch.no_grad():
        output, _ = model(x, lengths)
        other, _ = model(changed, lengths)
    torch.testing.assert_close(output[:, :3], other[:, :3])


@pytest.mark.parametrize("att_type", ["sdp", "torch_sdp"])
@pytest.mark.parametrize("provide_lengths", [False, True])
def test_cached_causal_matches_full_forward(
    att_type: str, provide_lengths: bool
) -> None:
    """Chunked causal attention matches a full call, including shared consumers.

    Args:
        att_type: Manual or Torch attention backend.
        provide_lengths: Whether lengths explicitly mark fully valid chunks.
    """
    model = _encoder(att_type=att_type)
    x = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 6]) if provide_lengths else None
    chunk_lengths = torch.tensor([3, 3]) if provide_lengths else None
    with torch.no_grad():
        expected, _ = model(x, lengths)
        cache = model.init_state(2, 6)
        first, _, cache = model(x[:, :3], chunk_lengths, state=cache)
        second, _, cache = model(x[:, 3:], chunk_lengths, start_pos=3, state=cache)
    torch.testing.assert_close(
        torch.cat([first, second], dim=1), expected, atol=1e-5, rtol=1e-4
    )
    assert all(entry.self_att is None for entry in cache.block_states[-2:])


@pytest.mark.parametrize("is_causal", [False, True])
def test_final_partial_chunk_and_rollover(is_causal: bool) -> None:
    """A padded final chunk fits rolled-over caches without historical validity fields."""
    model = _encoder(is_causal=is_causal)
    cache = model.init_state(1, 4)
    with torch.no_grad():
        _, _, cache = model(torch.randn(1, 4, 8), torch.tensor([4]), state=cache)
        output, _, cache = model(
            torch.randn(1, 2, 8), torch.tensor([1]), start_pos=4, state=cache
        )
    assert torch.isfinite(output[:, :1]).all()
    for entry in cache.block_states[:4]:
        assert entry.self_att["cache_offset"] == 2
        assert entry.self_att["cache_length"] == 4
        assert set(entry.self_att) == {
            "key",
            "value",
            "cache_length",
            "cache_offset",
            "max_cache_length",
        }


@pytest.mark.parametrize("att_type", ["sdp", "torch_sdp"])
def test_chunk_larger_than_cache_matches_full_forward(att_type: str) -> None:
    """Source and shared layers see the whole chunk before cache truncation.

    Args:
        att_type: Manual or Torch attention backend.
    """
    model = _encoder(att_type=att_type)
    x = torch.randn(2, 5, 8)
    cache = model.init_state(2, 2)
    lengths = torch.tensor([5, 4])
    with torch.no_grad():
        expected, _ = model(x, lengths)
        output, _, cache = model(x, lengths, state=cache)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)
    for entry in cache.block_states[:4]:
        assert entry.self_att["cache_offset"] == 3
        assert entry.self_att["cache_length"] == 2


def test_exported_kv_includes_history_and_full_chunk() -> None:
    """Exported K/V stays independent of the truncated cache across rollover."""
    att = TorchScaledDotProdAttV2(8, 2, sdp_backend=SDPBackendType.MATH).eval()
    cache = att.init_state(2, 2)
    x = torch.randn(2, 5, 8)
    y = torch.randn(2, 4, 8)
    with torch.no_grad():
        _, first_k, first_v, cache = att(x, x, x, state=cache, return_kv=True)
        retained_k = first_k[:, -2:].clone()
        retained_v = first_v[:, -2:].clone()
        first_copy = first_k.clone()
        expected_k = att.k_proj(y).view(2, 4, 2, 4)
        expected_v = att.v_proj(y).view(2, 4, 2, 4)
        _, full_k, full_v, cache = att(
            y, y, y, key_start_pos=5, query_start_pos=5, state=cache, return_kv=True
        )
    torch.testing.assert_close(full_k, torch.cat([retained_k, expected_k], dim=1))
    torch.testing.assert_close(full_v, torch.cat([retained_v, expected_v], dim=1))
    torch.testing.assert_close(first_k, first_copy)
    torch.testing.assert_close(cache["key"][:, :2], full_k[:, -2:])
    torch.testing.assert_close(cache["value"][:, :2], full_v[:, -2:])
    assert cache["cache_offset"] == 7
    assert cache["cache_length"] == 2


def test_masks_follow_stage_downsampling() -> None:
    """Later-stage masks use stage positions and source-cache lengths.

    No output equality across chunks is assumed for the strided convolution.
    """
    model = _encoder(encb_repeats=[6, 6], hidden_dims=[8, 12], downb_strides=[2])
    cache = model.init_state(2, 8)
    with torch.no_grad():
        _, _, cache = model(torch.randn(2, 4, 8), torch.tensor([4, 4]), state=cache)
        output, lengths, cache = model(
            torch.randn(2, 4, 8), torch.tensor([4, 2]), start_pos=4, state=cache
        )
    assert output.shape == (2, 2, 12)
    assert torch.isfinite(output).all()
    assert lengths.tolist() == [2, 1]
    assert cache.block_states[6].self_att["cache_length"] == 4


def test_one_mask_per_superblock() -> None:
    """Mask construction runs once per stage, before caches advance."""
    model = _encoder(encb_repeats=[6, 6], hidden_dims=[8, 12], downb_strides=[2])
    cache = model.init_state(2, 8)
    with patch.object(
        model, "_make_attention_mask", wraps=model._make_attention_mask
    ) as make_mask:
        with torch.no_grad():
            _, _, cache = model(torch.randn(2, 4, 8), torch.tensor([4, 4]), state=cache)
        assert make_mask.call_count == 2
        make_mask.reset_mock()
        with torch.no_grad():
            output, _, _ = model(
                torch.randn(2, 4, 8), torch.tensor([4, 2]), start_pos=4, state=cache
            )
        assert make_mask.call_count == 2
        assert [call.args[2] for call in make_mask.call_args_list] == [4, 2]
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("global_window", [None, 7])
@pytest.mark.parametrize("max_cache_length", [2, 20])
def test_window_cache_capacity_and_mask_routing(
    global_window: int | None, max_cache_length: int
) -> None:
    """Different local/global histories get matching masks and shared source K/V.

    Args:
        global_window: Global attention window, or None for unbounded attention.
        max_cache_length: First-stage cache limit before applying window limits.
    """
    model = _encoder(
        att_type="hf_flash_sdp",
        encb_repeats=[4, 4],
        hidden_dims=[8, 12],
        downb_strides=[2],
        local_attention_sliding_window=3,
        global_attention_sliding_window=global_window,
    )
    cache = model.init_state(2, max_cache_length)
    captured = {}
    for stage_idx, stage in enumerate(model.trans_blocks):
        stage_limit = (
            max(1, (max_cache_length + 1) // 2) if stage_idx else max_cache_length
        )
        for layer_idx, block in enumerate(stage):
            flat_idx = stage_idx * 4 + layer_idx
            layer_cache = cache.block_states[flat_idx].self_att
            if block.attention.shared_kv:
                assert layer_cache is None
            else:
                window = block.attention.sliding_window
                expected_capacity = (
                    stage_limit if window is None else min(stage_limit, window)
                )
                assert layer_cache["max_cache_length"] == expected_capacity
                assert layer_cache["key"].size(1) == 0
                assert layer_cache["value"].size(1) == 0

            def capture_attention(q, k, v, mask, flat_idx=flat_idx):
                """Capture cache/mask wiring without requiring a GPU Flash kernel.

                Args:
                    q: Prepared query tensor.
                    k: Full processed keys used for this call.
                    v: Full processed values used for this call.
                    mask: Backend padding mask.
                    flat_idx: Encoder layer index.

                Returns:
                    Zero attention output with the backend's expected shape.
                """
                assert mask.shape == (q.size(0), k.size(1))
                captured[flat_idx] = (k, v, mask)
                return torch.zeros_like(q.flatten(2))

            block.attention.compute_attention = capture_attention

    with torch.no_grad():
        _, _, cache = model(torch.randn(2, 8, 8), torch.tensor([8, 8]), state=cache)
        old_lengths = [
            None if entry.self_att is None else entry.self_att["cache_length"]
            for entry in cache.block_states
        ]
        with patch.object(
            model, "_make_attention_mask", wraps=model._make_attention_mask
        ) as make_mask:
            _, _, cache = model(
                torch.randn(2, 2, 8), torch.tensor([2, 1]), start_pos=8, state=cache
            )
            assert make_mask.call_count == (2 if max_cache_length == 2 else 4)
    for stage_idx in range(2):
        current_length = 2 if stage_idx == 0 else 1
        for source, consumer in [
            (stage_idx * 4, stage_idx * 4 + 2),
            (stage_idx * 4 + 1, stage_idx * 4 + 3),
        ]:
            key, value, mask = captured[source]
            assert key.size(1) == old_lengths[source] + current_length
            assert captured[consumer][0] is key
            assert captured[consumer][1] is value
            assert captured[consumer][2] is mask
        if stage_idx == 0:
            for idx in [0, 1]:
                assert captured[idx][2][1, -1].item() is False
                assert captured[idx][2][1, :-1].all()
