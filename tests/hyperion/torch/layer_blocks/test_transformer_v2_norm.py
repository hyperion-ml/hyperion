"""Pre/post normalization placement in V2 Transformer branches."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ArgumentParser

from hyperion.torch.layer_blocks.transformer_v2 import (
    TransformerV2AttType,
    TransformerV2ConvNextBlock,
    TransformerV2CrossAttBlock,
    TransformerV2FeedForwardType,
    TransformerV2MLPBlock,
    TransformerV2SelfAttBlock,
)
from hyperion.torch.layers.attention_v2 import SDPBackendType
from hyperion.torch.layers.norm_layers import RMSNorm
from hyperion.torch.narchs.qformer_v2 import QFormerV2
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _norm_reference(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    """Evaluate a norm independently of the toolkit forward implementation.

    Args:
        x: Input tensor.
        norm: Module containing the normalization parameters.

    Returns:
        Normalized tensor, or the original tensor for Identity.
    """
    if isinstance(norm, nn.Identity):
        return x
    if isinstance(norm, RMSNorm):
        return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)
    return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


@pytest.mark.parametrize("ff_type", ["mlp", "convnext", "g4moe"])
@pytest.mark.parametrize("cross_attention", [False, True])
@pytest.mark.parametrize("pre_post_norm", [False, True])
@pytest.mark.parametrize("norm_layer", [RMSNorm, nn.LayerNorm])
def test_branch_norms_before_residual(
    ff_type: str,
    cross_attention: bool,
    pre_post_norm: bool,
    norm_layer: type,
) -> None:
    """Verify output norms precede each residual addition for all FF variants.

    Args:
        ff_type: Feed-forward variant.
        cross_attention: Whether to exercise a cross-attention wrapper.
        pre_post_norm: Whether to enable post-normalization.
        norm_layer: Normalization constructor selected by the wrapper.
    """
    torch.manual_seed(7)
    kwargs = dict(
        att_type=TransformerV2AttType.TORCH_SDP,
        ff_type=ff_type,
        num_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_intermediate_feats=16,
        ff_kernel_size=3,
        ff_dilation=1,
        ff_multiple_of=4,
        ff_num_experts=3,
        ff_top_k_experts=2,
        ff_moe_intermediate_dim=8,
        norm_layer=norm_layer,
        norm_eps=2e-6,
        pre_post_norm=pre_post_norm,
    )
    if cross_attention:
        block = TransformerV2CrossAttBlock(**kwargs, num_kv_feats=6)
    else:
        block = TransformerV2SelfAttBlock(**kwargs)
    captures = {}

    def capture_attention(
        module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        """Capture the unnormalized self-attention output.

        Args:
            module: Attention module.
            inputs: Input tuple.
            output: Attention output.
        """
        captures["self_attention"] = output

    def capture_cross(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Capture the unnormalized cross-attention output.

        Args:
            module: Cross-attention module.
            inputs: Input tuple.
            output: Attention output.
        """
        captures["cross_attention"] = output

    def capture_ff(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Capture feed-forward inputs and the full branch output.

        Args:
            module: Feed-forward module.
            inputs: Input tuple.
            output: Feed-forward output.
        """
        captures["ff_input"] = inputs[0]
        captures["ff_output"] = output

    handles = [block.attention.register_forward_hook(capture_attention)]
    handles.append(block.feed_forward.register_forward_hook(capture_ff))
    if cross_attention:
        handles.append(block.cross_attention.register_forward_hook(capture_cross))
    x = torch.randn(2, 4, 8, requires_grad=True)
    if cross_attention:
        output = block(x, x_kv=torch.randn(2, 5, 6))
    else:
        output = block(x)
    for handle in handles:
        handle.remove()
    norm_type = norm_layer if pre_post_norm else nn.Identity
    assert isinstance(block.att_post_norm, norm_type)
    h = x + _norm_reference(captures["self_attention"], block.att_post_norm)
    if cross_attention:
        assert isinstance(block.cross_att_post_norm, norm_type)
        h = h + _norm_reference(captures["cross_attention"], block.cross_att_post_norm)
    torch.testing.assert_close(captures["ff_input"], _norm_reference(h, block.ff_norm))
    torch.testing.assert_close(output, h + captures["ff_output"])
    if ff_type == "g4moe":
        assert isinstance(block.ff_norm, nn.Identity)
        assert block.feed_forward.pre_post_norm is pre_post_norm
        moe_norm_type = norm_layer if pre_post_norm else nn.Identity
        assert isinstance(block.feed_forward.out_norm, moe_norm_type)
        # Experts and the dense MLP must not acquire additional output norms.
        assert isinstance(block.feed_forward.dense_mlp.post_norm, nn.Identity)
        assert all(
            isinstance(e.post_norm, nn.Identity) for e in block.feed_forward.experts
        )
    else:
        assert isinstance(block.feed_forward.post_norm, norm_type)
    (output * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "ff_class", [TransformerV2MLPBlock, TransformerV2ConvNextBlock]
)
@pytest.mark.parametrize("pre_post_norm", [False, True])
def test_ff_output_norm(ff_class: type, pre_post_norm: bool) -> None:
    """Output normalization is applied after the final feed-forward projection.

    Args:
        ff_class: Feed-forward class.
        pre_post_norm: Whether to enable output normalization.
    """
    ff = ff_class(
        8,
        16,
        ff_multiple_of=4,
        norm_layer=RMSNorm,
        norm_eps=2e-6,
        pre_post_norm=pre_post_norm,
    )
    raw_outputs = []
    handle = ff.down_proj.register_forward_hook(
        lambda module, inputs, output: raw_outputs.append(output)
    )
    output = ff(torch.randn(2, 4, 8))
    handle.remove()
    torch.testing.assert_close(output, _norm_reference(raw_outputs[0], ff.post_norm))
    assert isinstance(ff.post_norm, RMSNorm if pre_post_norm else nn.Identity)
    if ff_class is TransformerV2ConvNextBlock:
        assert ff.norm.eps == 2e-6


@pytest.mark.parametrize("architecture", [TransformerEncoderV2, QFormerV2])
@pytest.mark.parametrize("ff_type", ["mlp", "convnext", "g4moe"])
@pytest.mark.parametrize("pre_post_norm", [False, True])
def test_architecture_norm_config(
    architecture: type, ff_type: str, pre_post_norm: bool
) -> None:
    """Check parser/config propagation to every block, including tied QFormer layers.

    Args:
        architecture: Architecture class.
        ff_type: Feed-forward variant.
        pre_post_norm: Whether to enable post-normalization.
    """
    parser = ArgumentParser()
    architecture.add_class_args(parser, prefix="arch")
    assert parser.parse_args([]).arch.pre_post_norm is False
    args = ["--arch.pre-post-norm"] if pre_post_norm else []
    parsed = parser.parse_args(args).arch.as_dict()
    assert architecture.filter_args(**parsed)["pre_post_norm"] is pre_post_norm
    skipped = ArgumentParser()
    architecture.add_class_args(skipped, skip={"pre_post_norm"})
    assert "pre_post_norm" not in skipped.parse_args([]).as_dict()
    kwargs = dict(
        in_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_multiple_of=4,
        ff_type=ff_type,
        ff_num_experts=3,
        ff_top_k_experts=2,
        ff_moe_intermediate_dim=8,
        pre_post_norm=pre_post_norm,
        sdp_backend=SDPBackendType.MATH,
        norm_layer="rms-norm",
        norm_eps=2e-6,
        rope_original_max_seq_length=32,
    )
    if architecture is TransformerEncoderV2:
        kwargs.update(
            stem_type="conv1d",
            stem_hidden_channels=[8],
            stem_kernel_sizes=[3],
            stem_strides=[1],
            encb_repeats=[2],
            hidden_dims=[8],
            ff_kernel_sizes=[3],
        )
    else:
        kwargs.update(
            num_layers=4,
            hidden_dim=8,
            cross_att_freq=2,
            tied_layers=True,
            ff_kernel_size=3,
        )
    model = architecture(**kwargs)
    config = model.get_config(no_class_name=True)
    assert config["pre_post_norm"] is pre_post_norm
    assert config["sdp_backend"] == SDPBackendType.MATH
    restored = architecture(**config)
    restored.load_state_dict(model.state_dict())
    blocks = [
        m
        for m in restored.modules()
        if isinstance(m, (TransformerV2SelfAttBlock, TransformerV2CrossAttBlock))
    ]
    assert blocks
    for block in blocks:
        assert block.attention._sdp_backends == SDPBackendType.to_backend(
            SDPBackendType.MATH
        )
        assert block.pre_post_norm is pre_post_norm
        assert block.feed_forward.pre_post_norm is pre_post_norm
        assert isinstance(
            block.att_post_norm, RMSNorm if pre_post_norm else nn.Identity
        )
    feats = torch.randn(2, 6, 8)
    if architecture is TransformerEncoderV2:
        output, _ = restored(feats)
    else:
        output = restored(torch.randn(2, 3, 8), feats)
    assert torch.isfinite(output).all()


def test_cached_attention_keeps_post_norm() -> None:
    """Cached attention returns the same normalized branch as an uncached call."""
    block = TransformerV2SelfAttBlock(
        att_type=TransformerV2AttType.TORCH_SDP,
        ff_type=TransformerV2FeedForwardType.MLP,
        num_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_intermediate_feats=16,
        ff_kernel_size=3,
        ff_dilation=1,
        ff_multiple_of=4,
        pre_post_norm=True,
        norm_layer=RMSNorm,
    ).eval()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        expected = block(x)
        state = block.init_state(2, 4)
        output, updated_state = block(x, state=state)
    torch.testing.assert_close(output, expected)
    assert updated_state["cache_length"] == 4


@pytest.mark.parametrize("architecture", [TransformerEncoderV2, QFormerV2])
def test_moe_uses_default_rope_context(architecture: type) -> None:
    """Required MoE settings suffice without an explicit original RoPE context.

    Args:
        architecture: Architecture class to instantiate.
    """
    kwargs = dict(
        in_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_type="g4moe",
        ff_num_experts=3,
        ff_top_k_experts=2,
        ff_moe_intermediate_dim=8,
        ff_multiple_of=4,
    )
    if architecture is TransformerEncoderV2:
        kwargs.update(
            stem_type="conv1d",
            stem_hidden_channels=[8],
            stem_kernel_sizes=[3],
            stem_strides=[1],
            encb_repeats=[1],
            hidden_dims=[8],
        )
    else:
        kwargs.update(
            num_layers=1, hidden_dim=8, rope_in_self_att=True, rope_in_cross_att=True
        )
    model = architecture(**kwargs)
    assert model.rope_original_max_seq_length is None
    rope = model.global_rope[0] if architecture is TransformerEncoderV2 else model.rope
    assert rope.max_seq_length.item() > 0
    config = model.get_config(no_class_name=True)
    restored = architecture(**config)
    restored_rope = (
        restored.global_rope[0]
        if architecture is TransformerEncoderV2
        else restored.rope
    )
    assert restored_rope.max_seq_length.item() == rope.max_seq_length.item()
