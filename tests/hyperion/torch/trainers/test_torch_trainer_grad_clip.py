"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from hyperion.torch.trainers.torch_trainer_base import TorchTrainerBase
from hyperion.torch.utils.ddp import TorchDDP


def _check_gradient_update(
    model: nn.Module, rank: int, grad_clip: float, expected_cap: float
) -> None:
    trainer = TorchTrainerBase.__new__(TorchTrainerBase)
    trainer.rank = rank
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, 60000.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    grad_scaler = torch.amp.GradScaler(enabled=False)
    clip_grad_norm = nn.utils.clip_grad_norm_
    read_norm = Mock()

    def clip_and_record(*args: Any, **kwargs: Any) -> SimpleNamespace:
        norm = clip_grad_norm(*args, **kwargs)
        read_norm.side_effect = norm.item
        return SimpleNamespace(item=read_norm)

    with patch.object(nn.utils, "clip_grad_norm_", side_effect=clip_and_record):
        norm = trainer._update_model_by_optim(
            model, optimizer, grad_clip, 2, False, grad_scaler
        )

    torch.testing.assert_close(parameter, before - expected_cap)
    if rank == 0:
        read_norm.assert_called_once_with()
        assert norm == 60000.0
    else:
        read_norm.assert_not_called()
        assert norm is None


@pytest.mark.parametrize(
    "grad_clip,expected_cap", [(0, 30000), (-1, 30000), (123, 123)]
)
def test_gradient_clipping_cap(grad_clip: float, expected_cap: float) -> None:
    _check_gradient_update(nn.Linear(1, 1, bias=False), 0, grad_clip, expected_cap)


def _ddp_grad_clip_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = TorchDDP(nn.Linear(1, 1, bias=False))
        _check_gradient_update(model, rank, 0, 30000)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
def test_standard_ddp_reads_the_gradient_norm_on_rank_zero(tmp_path: Path) -> None:
    mp.spawn(
        _ddp_grad_clip_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )
