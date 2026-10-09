"""Dynamic attention caches retain suffix views without reserving the limit."""

import pytest
import torch

from hyperion.torch.layers.attention_v2 import (
    HFFlashScaledDotProdAttV2,
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)


@pytest.mark.parametrize(
    "attention_class",
    [ScaledDotProdAttV2, TorchScaledDotProdAttV2, HFFlashScaledDotProdAttV2],
)
def test_dynamic_cache_starts_empty(attention_class: type) -> None:
    """A large retention limit does not allocate a large initial tensor.

    Args:
        attention_class: Attention backend class.
    """
    att = attention_class(8, 2, num_kv_heads=1)
    state = att.init_state(2, 100000)
    assert state["max_cache_length"] == 100000
    assert state["cache_length"] == state["cache_offset"] == 0
    for name in ["key", "value"]:
        assert state[name].shape == (2, 0, 1, 4)
        assert state[name].untyped_storage().nbytes() == 0


def test_window_caps_retention_without_allocating_window() -> None:
    """Window and caller limits are applied independently of current K/V size."""
    att = HFFlashScaledDotProdAttV2(8, 2, sliding_window=16)
    for requested, expected in [(100000, 16), (4, 4)]:
        state = att.init_state(2, requested)
        assert state["max_cache_length"] == expected
        assert state["key"].size(1) == state["value"].size(1) == 0


@pytest.mark.parametrize(
    "attention_class", [ScaledDotProdAttV2, TorchScaledDotProdAttV2]
)
def test_dynamic_cache_exports_full_kv_and_retains_views(attention_class: type) -> None:
    """Growth and rollover preserve output values and export full current K/V.

    Args:
        attention_class: Manual or Torch attention implementation.
    """
    att = attention_class(8, 2, num_kv_heads=1, sdp_backend=SDPBackendType.MATH).eval()
    state = att.init_state(2, 4)
    offset = 0
    previous_k = previous_v = None
    with torch.no_grad():
        for chunk_length in [2, 1, 5, 1]:
            x = torch.randn(2, chunk_length, 8)
            old_k, old_v = state["key"], state["value"]
            _, current_k, current_v, _ = att._prepare_qkv(x, x, x, offset, offset, None)
            expected_k = torch.cat([old_k, current_k], dim=1)
            expected_v = torch.cat([old_v, current_v], dim=1)
            expected_q = att.q_proj(x).view(2, chunk_length, 2, 4)
            expected = att.o_proj(
                att.compute_attention(expected_q, expected_k, expected_v, None)
            )
            output, full_k, full_v, state = att(
                x,
                x,
                x,
                query_start_pos=offset,
                key_start_pos=offset,
                state=state,
                return_kv=True,
            )
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(full_k, expected_k)
            torch.testing.assert_close(full_v, expected_v)
            retained = min(offset + chunk_length, 4)
            assert state["cache_length"] == retained
            assert state["cache_offset"] == offset + chunk_length - retained
            assert state["key"].size(1) == state["value"].size(1) == retained
            for name, full in [("key", full_k), ("value", full_v)]:
                torch.testing.assert_close(state[name], full[:, -retained:])
                assert (
                    state[name].untyped_storage().data_ptr()
                    == full.untyped_storage().data_ptr()
                )
            if previous_k is not None:
                torch.testing.assert_close(old_k, previous_k)
                torch.testing.assert_close(old_v, previous_v)
            previous_k, previous_v = state["key"].clone(), state["value"].clone()
            offset += chunk_length


def test_dynamic_cache_overwrite_preserves_cached_suffix() -> None:
    """Overwriting an existing position neither loses the suffix nor rewinds offset."""
    att = TorchScaledDotProdAttV2(8, 2, sdp_backend=SDPBackendType.MATH).eval()
    state = att.init_state(1, 4)
    with torch.no_grad():
        x = torch.randn(1, 6, 8)
        _, _, _, state = att(x, x, x, state=state, return_kv=True)
        old_k, old_v = state["key"].clone(), state["value"].clone()
        update = torch.randn(1, 1, 8)
        _, new_k, new_v, _ = att._prepare_qkv(update, update, update, 3, 3, None)
        _, full_k, full_v, state = att(
            update,
            update,
            update,
            query_start_pos=3,
            key_start_pos=3,
            state=state,
            return_kv=True,
        )
    torch.testing.assert_close(
        full_k, torch.cat([old_k[:, :1], new_k, old_k[:, 2:]], dim=1)
    )
    torch.testing.assert_close(
        full_v, torch.cat([old_v[:, :1], new_v, old_v[:, 2:]], dim=1)
    )
    assert state["cache_offset"] == 2
    assert state["cache_length"] == state["max_cache_length"] == 4
