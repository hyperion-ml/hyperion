"""Causal convolutions retain context and stride alignment across chunks."""

import pytest
import torch

from hyperion.torch.layer_blocks.transformer_v2 import (
    TransformerEncoderV2StreamingConv1dStemBlock,
    TransformerV2StreamingConvDownsampleBlock,
)
from hyperion.torch.layers import StreamingCausalConv1d
from hyperion.torch.layers.attention_v2 import SDPBackendType
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _encoder(**kwargs) -> TransformerEncoderV2:
    """Build an encoder with strided causal convolutions and shared attention.

    Args:
        **kwargs: Constructor options to override.

    Returns:
        Small encoder in evaluation mode.
    """
    options = dict(
        in_feats=8,
        stem_type="conv1d",
        stem_hidden_channels=[8, 8],
        stem_kernel_sizes=[3, 4],
        stem_strides=[2, 2],
        stem_dropout_rate=0.0,
        hidden_dims=[8, 12],
        encb_repeats=[4, 4],
        downb_strides=[3],
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


@pytest.mark.parametrize("att_type", ["sdp", "torch_sdp"])
def test_arbitrary_chunks_match_full_causal_forward(att_type: str) -> None:
    """Different chunk boundaries preserve convolution and attention outputs.

    Args:
        att_type: Attention backend.
    """
    model = _encoder(att_type=att_type)
    assert isinstance(model.stem_block, TransformerEncoderV2StreamingConv1dStemBlock)
    assert isinstance(
        model.downsample_blocks[1], TransformerV2StreamingConvDownsampleBlock
    )
    assert model.in_context()[1] == 0
    x = torch.randn(2, 29, 8)
    with torch.no_grad():
        expected, expected_lengths = model(x, torch.tensor([29, 27]))
        for chunks in [[1] * 29, [3, 2, 7, 1, 5, 11]]:
            state = model.init_state(2, 16)
            outputs = []
            emitted_lengths = torch.zeros(2, dtype=torch.long)
            start = 0
            for length in chunks:
                lengths = torch.tensor([length, min(length, max(27 - start, 0))])
                output, out_lengths, state = model(
                    x[:, start : start + length], lengths, start_pos=start, state=state
                )
                outputs.append(output)
                emitted_lengths += out_lengths
                start += length
            actual = torch.cat(outputs, dim=1)
            common_length = int(expected_lengths.min())
            torch.testing.assert_close(
                actual[:, :common_length],
                expected[:, :common_length],
                atol=1e-5,
                rtol=1e-4,
            )
            torch.testing.assert_close(emitted_lengths, expected_lengths)
            assert state.block_states[0].self_att["cache_length"] == 8
            assert state.block_states[4].self_att["cache_length"] == 3


def test_causal_convolutions_ignore_future_frames_and_allow_training() -> None:
    """Causal stem/downsampling cannot leak future frames and remain differentiable."""
    model = _encoder()
    x = torch.randn(2, 29, 8, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 13:] = torch.randn_like(changed[:, 13:]) * 10
    output, lengths = model(x, torch.tensor([29, 27]))
    other, _ = model(changed, torch.tensor([29, 27]))
    # Final-stage outputs are anchored at input frames 0, 12, 24.
    torch.testing.assert_close(output[:, :2], other[:, :2])
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    for layer in model.stem_block.conv_layers:
        assert torch.isfinite(layer.conv.weight.grad).all()
    assert torch.isfinite(model.downsample_blocks[1].conv.weight.grad).all()
    assert lengths.tolist() == [3, 3]


@pytest.mark.parametrize("kernel_size,stride", [(1, 1), (4, 3)])
def test_streaming_downsample_matches_full_forward(
    kernel_size: int, stride: int
) -> None:
    """Phase-aware valid lengths match a full convolution, including a partial end.

    Args:
        kernel_size: Temporal convolution kernel size.
        stride: Downsampling stride.
    """
    block = TransformerV2StreamingConvDownsampleBlock(8, 12, kernel_size, stride).eval()
    x = torch.randn(2, 11, 8)
    with torch.no_grad():
        expected, expected_lengths = block(x, torch.tensor([11, 9]))
        state = block.init_state(2)
        outputs = []
        lengths = torch.zeros(2, dtype=torch.long)
        start = 0
        for chunk_size in [1, 2, 1, 4, 3]:
            chunk_lengths = torch.tensor(
                [chunk_size, min(chunk_size, max(9 - start, 0))]
            )
            output, out_lengths, state = block.stream(
                x[:, start : start + chunk_size], state, chunk_lengths
            )
            outputs.append(output)
            lengths += out_lengths
            start += chunk_size
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    torch.testing.assert_close(lengths, expected_lengths)
    assert lengths.tolist() == [(11 + stride - 1) // stride, (9 + stride - 1) // stride]
    assert state["tail"].size(-1) == max(kernel_size, stride) - 1


def test_pointwise_streaming_conv_keeps_no_history() -> None:
    """Kernel-one convolution never retains an ever-growing input tail."""
    conv = StreamingCausalConv1d(2, 3, 1).eval()
    x = torch.randn(1, 2, 7)
    with torch.no_grad():
        state = conv.init_state(1)
        first, state = conv.stream(x[..., :3], state)
        second, state = conv.stream(x[..., 3:], state)
        torch.testing.assert_close(torch.cat([first, second], dim=-1), conv(x))
    assert state["tail"].size(-1) == 0


@pytest.mark.parametrize(
    "options,match",
    [
        ({"stem_type": "conv2d"}, "conv1d stem"),
        ({"ff_type": "convnext"}, "ConvNeXt"),
        ({"multilayer": True}, "same temporal scale"),
    ],
)
def test_unsupported_causal_blocks_raise(options: dict, match: str) -> None:
    """Unsupported causal configurations fail during construction.

    Args:
        options: Unsupported constructor options.
        match: Expected error message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _encoder(**options)


@pytest.mark.parametrize("multilayer_concat", [False, True])
def test_same_scale_endpoints_preserve_streaming(multilayer_concat: bool) -> None:
    """Pointwise endpoint projections accept chunks with no emitted frames.

    Args:
        multilayer_concat: Whether endpoints are concatenated or averaged.
    """
    model = _encoder(
        downb_strides=[1], multilayer=True, multilayer_concat=multilayer_concat
    )
    x = torch.randn(1, 9, 8)
    with torch.no_grad():
        expected, _ = model(x)
        state = model.init_state(1, 8)
        outputs = []
        for start in range(x.size(1)):
            output, _, state = model(
                x[:, start : start + 1], start_pos=start, state=state
            )
            outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, atol=1e-5, rtol=1e-4
    )
