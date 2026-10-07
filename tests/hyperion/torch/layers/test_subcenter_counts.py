"""Distributed main-subcenter selection regression tests."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyperion.torch.layers.margin_losses import SubCenterArcLossOutput


def _counts_worker(rank: int, rendezvous: str) -> None:
    """Compare synchronized updates with concatenated-batch updates.

    Args:
        rank: Distributed rank.
        rendezvous: File-store path.
    """
    torch.set_num_threads(1)
    layer = SubCenterArcLossOutput(3, 2, 2)
    reference = SubCenterArcLossOutput(3, 2, 2)
    with torch.no_grad():
        layer.kernel.copy_(torch.arange(1, 13).reshape(3, 4))
        reference.kernel.copy_(layer.kernel)
        layer.subcenter_counts.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
        reference.subcenter_counts.copy_(layer.subcenter_counts)
    batches = [
        [
            (torch.tensor([0]), torch.tensor([[0, 0]])),
            (
                torch.tensor([0, 0, 0, 1]),
                torch.tensor([[1, 0], [1, 0], [1, 0], [0, 0]]),
            ),
        ],
        [
            (torch.empty(0, dtype=torch.long), torch.empty(0, 2, dtype=torch.long)),
            (torch.tensor([1, 1, 1, 1]), torch.tensor([[0, 1]] * 4)),
        ],
    ]
    expected = []
    for batch in batches:
        reference._update_counts(
            torch.cat([x[0] for x in batch]), torch.cat([x[1] for x in batch])
        )
        expected.append(
            (reference.subcenter_counts.clone(), reference.raw_prototypes.clone())
        )
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        for batch, (counts, prototypes) in zip(batches, expected):
            layer._update_counts(*batch[rank])
            torch.testing.assert_close(layer.subcenter_counts, counts)
            torch.testing.assert_close(layer.raw_prototypes, prototypes)
    finally:
        dist.destroy_process_group()


def test_distributed_subcenter_counts(tmp_path: Path) -> None:
    """Unequal and empty local batches select the global main subcenter.

    Args:
        tmp_path: Temporary rendezvous directory.
    """
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_counts_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
