"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from hyperion.torch.lr_schedulers import ReduceLROnPlateau


def _make_scheduler() -> ReduceLROnPlateau:
    optimizer = torch.optim.SGD(
        [
            {"params": [nn.Parameter(torch.ones(()))], "lr": 0.2},
            {"params": [nn.Parameter(torch.ones(()))], "lr": 0.1},
        ]
    )
    return ReduceLROnPlateau(
        optimizer, monitor="val_score", patience=1, factor=0.5, cooldown=2
    )


def test_plateau_uses_local_metric_in_single_process() -> None:
    scheduler = _make_scheduler()
    scheduler.on_epoch_end({"val_score": 3.0})
    assert scheduler.best == 3.0
    assert scheduler.epoch == 1
    with pytest.raises(KeyError, match="val_score"):
        scheduler.on_epoch_end({})


def _plateau_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        scheduler = _make_scheduler()
        expected = [
            ([0.2, 0.1], 0, 0),
            ([0.2, 0.1], 1, 0),
            ([0.1, 0.05], 0, 2),
            ([0.1, 0.05], 0, 1),
        ]
        for epoch, (lrs, bad_epochs, cooldown) in enumerate(expected, start=1):
            logs = {"val_score": 3.0, "val_loss": 1.0} if rank == 0 else {}
            scheduler.on_epoch_end(logs)
            assert [group["lr"] for group in scheduler.optimizer.param_groups] == lrs
            assert scheduler.best == 3.0
            assert scheduler.epoch == epoch
            assert scheduler.num_bad_epochs == bad_epochs
            assert scheduler.cooldown_counter == cooldown

        with pytest.raises(KeyError, match="val_score"):
            scheduler.on_epoch_end({} if rank == 0 else {"val_score": 1.0})
        assert scheduler.epoch == len(expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
def test_plateau_shares_the_monitored_metric_from_rank_zero(tmp_path: Path) -> None:
    mp.spawn(_plateau_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
