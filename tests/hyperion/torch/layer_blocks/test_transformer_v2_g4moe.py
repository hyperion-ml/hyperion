"""Gemma 4 MoE routing and feed-forward integration regression tests."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ArgumentParser

from hyperion.torch.layer_blocks.transformer_v2 import (
    TransformerV2AttType,
    TransformerV2CrossAttBlock,
    TransformerV2FeedForwardType,
    TransformerV2G4MoEBlock,
    TransformerV2SelfAttBlock,
)
from hyperion.torch.narchs.qformer_v2 import QFormerV2
from hyperion.torch.narchs.transformer_encoder_v2 import TransformerEncoderV2


def _rms(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    """Independent RMSNorm reference.

    Args:
        x: Input tensor.
        norm: Module providing epsilon and learned scale.

    Returns:
        Normalized tensor.
    """
    return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)


@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("pre_post_norm", [False, True])
def test_sparse_moe_matches_dense_reference(top_k: int, pre_post_norm: bool) -> None:
    """Compare sparse dispatch against all-expert evaluation with masked weights.

    Args:
        top_k: Number of selected experts.
        pre_post_norm: Whether to enable branch and final post-normalization.
    """
    torch.manual_seed(42)
    block = TransformerV2G4MoEBlock(
        8, 12, 3, top_k, 10, ff_multiple_of=4, pre_post_norm=pre_post_norm
    )
    assert block.intermediate_dim == 12
    assert block.moe_intermediate_dim == 12
    assert block.dense_mlp.act.approximate == "tanh"
    with torch.no_grad():
        block.router_scale.copy_(torch.linspace(0.5, 1.5, 8))
        block.per_expert_scale.copy_(torch.tensor([0.5, 1.0, 2.0]))
    x = torch.randn(2, 5, 8, requires_grad=True)
    flat = x.reshape(-1, 8)
    normalized = F.rms_norm(flat, (8,), eps=block.norm_eps)
    probabilities = F.softmax(
        F.linear(normalized * block.router_scale / (8**0.5), block.router_proj.weight),
        dim=-1,
    )
    top_values, top_indices = probabilities.topk(top_k, dim=-1)
    top_weights = top_values / top_values.sum(-1, keepdim=True)
    top_weights = top_weights * block.per_expert_scale[top_indices]
    weights, indices = block._route(flat)
    torch.testing.assert_close(indices, top_indices)
    torch.testing.assert_close(weights, top_weights)
    full_weights = torch.zeros_like(probabilities).scatter(1, top_indices, top_weights)
    expert_input = _rms(flat, block.expert_pre_norm)
    all_outputs = torch.stack([expert(expert_input) for expert in block.experts], dim=1)
    expert_sum = (all_outputs * full_weights[..., None]).sum(1).reshape_as(x)
    dense = block.dense_mlp(_rms(x, block.dense_pre_norm))
    if pre_post_norm:
        dense = _rms(dense, block.dense_post_norm)
        expert_sum = _rms(expert_sum, block.expert_post_norm)
        expected = _rms(dense + expert_sum, block.out_norm)
    else:
        expected = dense + expert_sum
        assert isinstance(block.dense_post_norm, nn.Identity)
        assert isinstance(block.expert_post_norm, nn.Identity)
        assert isinstance(block.out_norm, nn.Identity)
    actual = block(x)
    torch.testing.assert_close(actual, expected)
    coefficients = torch.randn_like(actual)
    parameters = [x, *block.parameters()]
    actual_gradients = torch.autograd.grad(
        (actual * coefficients).sum(), parameters, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * coefficients).sum(),
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    for parameter, actual_grad, expected_grad in zip(
        parameters, actual_gradients, expected_gradients
    ):
        actual_grad = (
            torch.zeros_like(parameter) if actual_grad is None else actual_grad
        )
        expected_grad = (
            torch.zeros_like(parameter) if expected_grad is None else expected_grad
        )
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-4)
    (actual * coefficients).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert block.router_scale.grad is not None
    assert block.per_expert_scale.grad is not None
    if top_k > 1:
        assert block.router_proj.weight.grad.abs().sum() > 0


def test_moe_skips_unselected_experts() -> None:
    """Only selected experts execute, while the dense MLP always executes."""
    block = TransformerV2G4MoEBlock(4, 8, 3, 1, 8, ff_multiple_of=4)
    assert block.pre_post_norm is False
    assert isinstance(block.dense_post_norm, nn.Identity)
    assert isinstance(block.expert_post_norm, nn.Identity)
    assert isinstance(block.out_norm, nn.Identity)
    with torch.no_grad():
        block.router_proj.weight.copy_(torch.tensor([[2.0] * 4, [0.0] * 4, [-2.0] * 4]))
    calls = [0, 0, 0]
    handles = []
    for idx, expert in enumerate(block.experts):

        def record(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
            idx: int = idx,
        ) -> None:
            """Count expert executions.

            Args:
                module: Expert module.
                inputs: Expert input tuple.
                output: Expert output tensor.
                idx: Expert index captured by the hook.
            """
            calls[idx] += 1

        handles.append(expert.register_forward_hook(record))
    output = block(torch.ones(2, 3, 4))
    (output * torch.randn_like(output)).sum().backward()
    assert all(
        parameter.grad is None
        for expert in block.experts[1:]
        for parameter in expert.parameters()
    )
    for handle in handles:
        handle.remove()
    assert calls == [1, 0, 0]


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"num_experts": 0},
        {"top_k_experts": 0},
        {"top_k_experts": 4},
        {"moe_intermediate_dim": None},
        {"ff_multiple_of": 0},
        {"norm_eps": 0},
    ],
)
def test_invalid_moe_configuration(bad_kwargs: dict) -> None:
    """Reject invalid sizes, routing counts, and normalization settings.

    Args:
        bad_kwargs: Invalid constructor settings.
    """
    kwargs = dict(
        hidden_dim=8,
        intermediate_dim=16,
        num_experts=3,
        top_k_experts=2,
        moe_intermediate_dim=8,
        ff_multiple_of=4,
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        TransformerV2G4MoEBlock(**kwargs)


@pytest.mark.parametrize("cross_attention", [False, True])
def test_moe_block_receives_unnormalized_input(cross_attention: bool) -> None:
    """Wrapper pre-normalization is bypassed for the MoE feed-forward branch.

    Args:
        cross_attention: Whether to construct a cross-attention wrapper.
    """
    kwargs = dict(
        att_type=TransformerV2AttType.TORCH_SDP,
        ff_type=TransformerV2FeedForwardType.G4MoE,
        num_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_intermediate_feats=16,
        ff_kernel_size=3,
        ff_dilation=1,
        ff_activation="gelu-tanh",
        ff_multiple_of=4,
        ff_num_experts=3,
        ff_top_k_experts=2,
        ff_moe_intermediate_dim=8,
    )
    if cross_attention:
        block = TransformerV2CrossAttBlock(**kwargs, num_kv_feats=6)
    else:
        block = TransformerV2SelfAttBlock(**kwargs)
    assert isinstance(block.ff_norm, nn.Identity)
    with torch.no_grad():
        block.attention.o_proj.weight.zero_()
        if cross_attention:
            block.cross_attention.o_proj.weight.zero_()
    x = torch.randn(2, 4, 8)
    inputs = []
    handle = block.feed_forward.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    if cross_attention:
        output = block(x, x_kv=torch.randn(2, 5, 6))
    else:
        output = block(x)
    handle.remove()
    torch.testing.assert_close(inputs[0], x)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("architecture", [TransformerEncoderV2, QFormerV2])
@pytest.mark.parametrize("pre_post_norm", [False, True])
def test_moe_architecture_config_and_parser(
    architecture: type, pre_post_norm: bool
) -> None:
    """MoE options survive parsing, filtering, reconstruction, and forward passes.

    Args:
        architecture: Architecture class to exercise.
        pre_post_norm: Whether to enable branch post-normalization.
    """
    parser = ArgumentParser()
    architecture.add_class_args(parser, prefix="arch")
    parsed = parser.parse_args(
        [
            "--arch.ff-type=g4moe",
            "--arch.ff-act=gelu-tanh",
            "--arch.ff-num-experts=3",
            "--arch.ff-top-k-experts=2",
            "--arch.ff-moe-intermediate-dim=12",
        ]
    ).arch.as_dict()
    kwargs = dict(
        pre_post_norm=pre_post_norm,
        in_feats=8,
        num_heads=2,
        num_kv_heads=1,
        ff_multiple_of=4,
        norm_eps=2e-6,
        rope_original_max_seq_length=32,
    )
    for key in [
        "ff_type",
        "ff_act",
        "ff_num_experts",
        "ff_top_k_experts",
        "ff_moe_intermediate_dim",
    ]:
        kwargs[key] = architecture.filter_args(**parsed)[key]
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
        kwargs.update(num_layers=2, hidden_dim=8, cross_att_freq=2, tied_layers=True)
    model = architecture(**kwargs)
    config = model.get_config(no_class_name=True)
    restored = architecture(**config)
    restored.load_state_dict(model.state_dict())
    moe_blocks = [
        m for m in restored.modules() if isinstance(m, TransformerV2G4MoEBlock)
    ]
    assert moe_blocks
    for moe in moe_blocks:
        assert moe.pre_post_norm is pre_post_norm
        assert moe.num_experts == 3
        assert moe.top_k_experts == 2
        assert moe.norm_eps == 2e-6
        assert moe.moe_intermediate_dim == 12
    feats = torch.randn(2, 6, 8)
    if architecture is TransformerEncoderV2:
        output, _ = restored(feats)
    else:
        output = restored(torch.randn(2, 3, 8), feats)
    assert torch.isfinite(output).all()
    assert TransformerV2FeedForwardType.to_class("g4moe") is TransformerV2G4MoEBlock


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_dtype_and_tensor_parallel_single_rank(dtype: torch.dtype) -> None:
    """Check dtype preservation and the tensor-parallel path with one rank.

    Args:
        dtype: Input and parameter dtype.
    """
    block = TransformerV2G4MoEBlock(
        8, 16, 3, 2, 8, ff_multiple_of=4, model_parallel=True, ff_bias=True
    ).to(dtype=dtype)
    output = block(torch.randn(2, 3, 8, dtype=dtype))
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert block.dense_mlp.up_proj.bias is not None
    assert block.dense_mlp.down_proj.bias is not None
