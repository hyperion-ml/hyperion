"""Stage-local shared suffixes reuse processed K/V and source caches."""

import pytest
import torch
from jsonargparse import ArgumentParser

from hyperion.torch.layers.attention_v2 import SDPBackendType
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _make_encoder(**kwargs) -> TransformerEncoderV2:
    """Construct a small encoder with deterministic attention.

    Args:
        **kwargs: Constructor settings to override.

    Returns:
        Encoder suitable for source/cache checks.
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
        local_head_dim=4,
        global_head_dim=6,
        global_k_eq_v=True,
        enable_qk_norm=True,
        enable_v_norm=True,
        num_kv_shared_layers=2,
        ff_multiple_of=4,
        sdp_backend=SDPBackendType.MATH,
        local_rope_scale_freqs=False,
        global_rope_scale_freqs=False,
        rope_update_max_seq_length=False,
    )
    options.update(kwargs)
    return TransformerEncoderV2(**options).eval()


def test_shared_sources_and_gradients() -> None:
    """Consumers use the last independent same-type source without copying K/V."""
    model = _make_encoder()
    assert model.layer_types == ["local", "global"] * 3
    assert model.kv_source_layers == [None, None, None, None, 2, 3]
    captured = {}
    blocks = list(model.trans_blocks[0])
    for idx, block in enumerate(blocks):
        attention = block.attention
        compute_attention = attention.compute_attention

        def capture(q, k, v, mask, idx=idx, compute_attention=compute_attention):
            """Record processed source/consumer tensors before backend dispatch.

            Args:
                q: Query tensor.
                k: Key tensor.
                v: Value tensor.
                mask: Attention mask.
                idx: Flat block index.
                compute_attention: Original backend method.

            Returns:
                Original backend attention output.
            """
            captured[idx] = (k, v)
            return compute_attention(q, k, v, mask)

        attention.compute_attention = capture
    x = torch.randn(2, 5, 8, requires_grad=True)
    output, _ = model(x)
    for consumer_idx, source_idx in [(4, 2), (5, 3)]:
        consumer = blocks[consumer_idx].attention
        assert consumer.k_proj is consumer.v_proj is None
        assert consumer.k_norm is consumer.v_norm is None
        for source_tensor, consumer_tensor in zip(
            captured[source_idx], captured[consumer_idx]
        ):
            assert source_tensor is consumer_tensor
            source_tensor.retain_grad()
    output.square().sum().backward()
    for idx in [2, 3]:
        assert captured[idx][0].grad is not None
        assert captured[idx][1].grad is not None
        assert torch.isfinite(blocks[idx].attention.k_proj.weight.grad).all()
    assert torch.isfinite(x.grad).all()
    config = model.get_config(no_class_name=True)
    assert config["num_kv_shared_layers"] == [2]
    restored = TransformerEncoderV2(**config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x.detach())[0], output.detach())


def test_shared_cache_ownership() -> None:
    """Only independent layers allocate caches; consumers see source updates."""
    model = _make_encoder()
    state = model.init_state(2, 8)
    assert len(state.block_states) == 6
    assert all(entry.self_att is None for entry in state.block_states[4:])
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        expected, _ = model(x)
        actual, _, state = model(x, state=state)
        torch.testing.assert_close(actual, expected)
        actual, _, state = model(torch.randn(2, 2, 8), start_pos=4, state=state)
    assert torch.isfinite(actual).all()
    assert all(entry.self_att["cache_length"] == 6 for entry in state.block_states[:4])
    assert all(entry.self_att is None for entry in state.block_states[4:])


def test_superblocks_keep_separate_sources() -> None:
    """Different stage dimensions and downsampling never share source tensors."""
    model = _make_encoder(
        encb_repeats=[6, 6],
        hidden_dims=[8, 12],
        downb_strides=[2],
        num_kv_shared_layers=[2, 2],
    )
    assert model.kv_source_layers == [
        None,
        None,
        None,
        None,
        2,
        3,
        None,
        None,
        None,
        None,
        8,
        9,
    ]
    assert model(torch.randn(2, 8, 8))[0].shape == (2, 4, 12)
    broadcast = _make_encoder(
        encb_repeats=[6, 6], hidden_dims=[8, 8], num_kv_shared_layers=2
    )
    assert broadcast.num_kv_shared_layers == [2, 2]


@pytest.mark.parametrize("count", [-1, True, 6, [1, 2], 5])
def test_invalid_shared_counts(count) -> None:
    """Reject invalid counts and suffixes without a same-type independent source.

    Args:
        count: Invalid sharing configuration.
    """
    with pytest.raises(ValueError, match="num_kv_shared_layers|no non-shared"):
        _make_encoder(num_kv_shared_layers=count)


def test_no_sharing_and_global_only() -> None:
    """Zero preserves independent layers; global-only consumers share one source."""
    independent = _make_encoder(num_kv_shared_layers=0)
    assert independent.kv_source_layers == [None] * 6
    assert all(not block.attention.shared_kv for block in independent.trans_blocks[0])
    global_only = _make_encoder(local_to_global_ratio=0, num_kv_shared_layers=3)
    assert global_only.kv_source_layers == [None, None, None, 2, 2, 2]
    assert torch.isfinite(global_only(torch.randn(2, 4, 8))[0]).all()


def test_shared_kv_parser() -> None:
    """Check integer/list CLI configurations, filtering, nesting, and skipping."""
    parser = ArgumentParser()
    TransformerEncoderV2.add_class_args(parser, prefix="arch")
    assert parser.parse_args([]).arch.num_kv_shared_layers == 0
    assert (
        parser.parse_args(["--arch.num-kv-shared-layers=2"]).arch.num_kv_shared_layers
        == 2
    )
    assert parser.parse_args(
        ["--arch.num-kv-shared-layers=[0, 2]"]
    ).arch.num_kv_shared_layers == [0, 2]
    assert TransformerEncoderV2.filter_args(num_kv_shared_layers=[0, 2]) == {
        "num_kv_shared_layers": [0, 2]
    }
    skipped = ArgumentParser()
    TransformerEncoderV2.add_class_args(skipped, skip={"num_kv_shared_layers"})
    assert "num_kv_shared_layers" not in skipped.parse_args([]).as_dict()
