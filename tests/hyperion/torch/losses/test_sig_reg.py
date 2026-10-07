"""Reference and distributed regression coverage for SIGReg."""

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from jsonargparse import ArgumentParser, namespace_to_dict
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from hyperion.torch.losses import SIGReg


def reference(z, directions, t):
    """Evaluate the full complex characteristic function and trapezoidal rule.

    Args:
        z: Samples [N, D].
        directions: Unit directions [D, M].
        t: Frequency nodes [T].

    Returns:
        Scalar Epps--Pulley projection mean.
    """
    phase = (z.float() @ directions).unsqueeze(-1) * t
    ecf = torch.exp(1j * phase).mean(0)
    phi = torch.exp(-0.5 * t.square())
    return z.shape[0] * torch.trapezoid((ecf - phi).abs().square() * phi, t).mean()


@pytest.mark.parametrize("chunk", [None, 1, 7, 32])
def test_reference_loss_and_gradients(chunk):
    z = torch.randn(19, 5, requires_grad=True)
    reg = SIGReg(num_slices=23, projection_chunk_size=chunk)
    directions = reg._sample_directions(5, z.device)
    expected = reference(z, directions, reg.t)
    actual = reg(z)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z)[0], torch.autograd.grad(expected, z)[0]
    )


@pytest.mark.parametrize("multi_view", [False, True])
@pytest.mark.parametrize("with_lengths", [False, True])
def test_sequences_and_views(multi_view, with_lengths):
    shape = (2, 3, 6, 4) if multi_view else (3, 6, 4)
    z = torch.randn(*shape, requires_grad=True)
    lengths = torch.tensor([[6, 3, 0], [2, 5, 4]] if multi_view else [6, 3, 0])
    reg = SIGReg(num_slices=13, multi_view=multi_view)
    directions = reg._sample_directions(4, z.device)
    views = z.unbind(0) if multi_view else (z,)
    view_lengths = lengths.unbind(0) if multi_view else (lengths,)
    valid = [
        view[torch.arange(6) < length[:, None]] if with_lengths else view.reshape(-1, 4)
        for view, length in zip(views, view_lengths)
    ]
    expected = torch.stack([reference(x, directions, reg.t) for x in valid]).mean()
    actual = reg(z, lengths if with_lengths else None)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z)[0], torch.autograd.grad(expected, z)[0]
    )


def test_sample_views_are_not_pooled():
    z = torch.randn(2, 15, 4)
    z[1] += 3
    reg = SIGReg(num_slices=13, multi_view=True)
    directions = reg._sample_directions(4, z.device)
    expected = torch.stack([reference(x, directions, reg.t) for x in z]).mean()
    actual = reg(z)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, reference(z.reshape(-1, 4), directions, reg.t))


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("multi_view", [False, True])
def test_view_reduction_reference(reduction, multi_view):
    z = torch.randn(3, 12, 4) if multi_view else torch.randn(12, 4)
    z.requires_grad_()
    reg = SIGReg(num_slices=11, multi_view=multi_view, reduction=reduction)
    directions = reg._sample_directions(4, z.device)
    if multi_view:
        expected = torch.stack([reference(view, directions, reg.t) for view in z])
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()
    else:
        expected = reference(z, directions, reg.t)
    actual = reg(z)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), z)[0],
        torch.autograd.grad(expected.sum(), z)[0],
    )
    assert SIGReg(**reg.get_config()).reduction == reduction


def test_counter_checkpoint_and_rng():
    reg = SIGReg(num_slices=11, seed=123)
    z = torch.ones(8, 3)
    rng = torch.random.get_rng_state().clone()
    first = reg(z)
    state = copy.deepcopy(reg.state_dict())
    second = reg(z)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert reg.counter.item() == 2
    assert not torch.isclose(first, second)
    restored = SIGReg(**reg.get_config())
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(z), second)


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)]
)
def test_mixed_precision(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = nn.Linear(5, 4).to(device)
    reg = SIGReg(num_slices=17, projection_chunk_size=5).to(device=device, dtype=dtype)
    with torch.autocast(device_type=device, dtype=dtype):
        loss = reg(model(torch.randn(13, 5, device=device)))
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert reg.phi.dtype == torch.float32
    if device == "cuda":
        assert reg._generator.device.type == "cuda"


def test_gaussian_has_lower_average_discrepancy():
    generator = torch.Generator().manual_seed(42)
    totals = torch.zeros(4)
    reg = SIGReg(num_slices=64)
    for _ in range(8):
        z = torch.randn(512, 6, generator=generator)
        directions = reg._sample_directions(6, z.device)
        for i, samples in enumerate((z, torch.zeros_like(z), z + 3, z * 4)):
            totals[i] += reference(samples, directions, reg.t).detach()
        reg.counter.add_(1)
    assert totals[0] > 0
    assert (totals[0] < totals[1:]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_slices": 0},
        {"num_points": 1},
        {"t_max": float("inf")},
        {"projection_chunk_size": 0},
        {"distributed_mode": "other"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SIGReg(**kwargs)


def test_invalid_inputs():
    reg = SIGReg()
    for z, lengths in (
        (torch.randn(4), None),
        (torch.ones(4, 3, dtype=torch.long), None),
        (torch.randn(4, 3), torch.ones(4, dtype=torch.long)),
        (torch.randn(4, 5, 3), torch.ones(4)),
        (torch.randn(4, 5, 3), torch.ones(4, dtype=torch.int32)),
        (torch.randn(4, 5, 3), torch.ones(3, dtype=torch.long)),
        (torch.empty(0, 3), None),
    ):
        with pytest.raises(ValueError):
            reg(z, lengths)

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="same device"):
            reg(torch.randn(4, 5, 3, device="cuda"), torch.ones(4, dtype=torch.long))


def test_parser_configuration():
    parser = ArgumentParser()
    SIGReg.add_class_args(parser, prefix="sigreg")
    args = namespace_to_dict(
        parser.parse_args(
            [
                "--sigreg.num-slices=1024",
                "--sigreg.multi-view",
                "--sigreg.reduction=none",
            ]
        )
    )
    reg = SIGReg(**SIGReg.filter_args(**args["sigreg"], unused=1))
    assert reg.num_slices == 1024 and reg.multi_view
    assert reg.reduction == "none"


def distributed_worker(rank, rendezvous, sizes, mode, sequence):
    """Compare replicated losses and DDP gradients against a global reference.

    Args:
        rank: Process rank.
        rendezvous: File-store path.
        sizes: Per-rank sample counts.
        mode: Aggregation mode.
        sequence: Whether to test masked multi-view sequence inputs.
    """
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(12)
        model = nn.Linear(3, 4)
        ref_model = copy.deepcopy(model)
        ddp = DDP(model)
        x = torch.randn(2, sum(sizes), 4, 3) if sequence else torch.randn(sum(sizes), 3)
        reg = SIGReg(
            num_slices=13,
            projection_chunk_size=5,
            distributed_mode=mode,
            multi_view=sequence,
        )
        lengths = None
        if sequence:
            lengths = torch.arange(2 * sum(sizes)).reshape(2, -1) % 5
        start, end = sum(sizes[:rank]), sum(sizes[: rank + 1])
        local = (
            x if mode == "local" else (x[:, start:end] if sequence else x[start:end])
        )
        local_lengths = lengths[:, start:end] if sequence else None
        actual = reg(ddp(local), local_lengths)
        # Explicit complex reference; all ranks start with identical model and data.
        full_z = ref_model(x)
        directions = SIGReg(num_slices=13)._sample_directions(4, x.device)
        t = torch.linspace(-5, 5, 17)
        if sequence:
            expected = torch.stack(
                [
                    reference(view[torch.arange(4) < length[:, None]], directions, t)
                    for view, length in zip(full_z, lengths)
                ]
            ).mean()
        else:
            expected = reference(full_z, directions, t)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        actual.backward()
        expected.backward()
        for parameter, ref_parameter in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=3e-5, atol=3e-5
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "sizes,mode,sequence",
    [
        ((4, 4), "global_data", False),
        ((3, 7), "global_data", False),
        ((0, 9), "global_data", False),
        ((3, 7), "global_data", True),
        ((4, 4), "local", False),
    ],
)
def test_ddp_loss_and_parameter_gradients(tmp_path: Path, sizes, mode, sequence):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(
        distributed_worker,
        args=(str(tmp_path / "store"), sizes, mode, sequence),
        nprocs=2,
        join=True,
    )
