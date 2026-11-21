"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from hyperion.torch.trainers.single_model_trainer import SingleModelTrainer
from hyperion.torch.utils.ddp import TorchDDP


class _ToyModel(nn.Sequential):
    def forward(self, audio: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return super().forward(audio)


def _count_reductions(
    calls: list[int], bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    calls.append(bucket.index())
    return allreduce_hook(None, bucket)


def _run_batches(
    rank: int, device: torch.device, defer_sync: bool
) -> tuple[list[int], dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    norm = nn.SyncBatchNorm(4, momentum=1.0) if device.type == "cuda" else nn.Identity()
    model = _ToyModel(nn.Linear(4, 4), norm).to(device)
    ddp = TorchDDP(model, device_ids=[rank] if device.type == "cuda" else None)
    calls = []
    ddp.register_comm_hook(calls, _count_reductions)

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.model = ddp
    trainer.device = device
    trainer.input_key, trainer.target_key = "x", "y"
    trainer.grad_acc_steps = 2
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer.grad_scaler = torch.amp.GradScaler(enabled=False)
    trainer.loss = nn.MSELoss()
    if not defer_sync:
        trainer._ddp_sync_context = lambda model, sync_gradients: nullcontext()

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.03)
    base_audio = torch.linspace(-1, 1, 12, device=device).reshape(3, 4)
    reduction_counts = []
    for batch_idx in range(5):
        audio = base_audio * (batch_idx + 1)
        if isinstance(norm, nn.SyncBatchNorm):
            # The two ranks have different local means. Every forward should
            # use the mean of the current distributed batch.
            with torch.no_grad():
                global_mean = model[0](audio + 1).mean(dim=0)
        previous_count = len(calls)
        trainer.training_step(
            batch_idx, {"id": ["a", "b", "c"], "x": audio + 2 * rank, "y": audio}
        )
        reduction_counts.append(len(calls) - previous_count)
        if isinstance(norm, nn.SyncBatchNorm):
            torch.testing.assert_close(norm.running_mean, global_mean)
        if trainer._should_update_models(batch_idx):
            optimizer.step()
            optimizer.zero_grad()

    return reduction_counts, model.state_dict()


def _ddp_sync_worker(rank: int, rendezvous: str, backend: str) -> None:
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)

    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        baseline_counts, baseline = _run_batches(rank, device, defer_sync=False)
        deferred_counts, deferred = _run_batches(rank, device, defer_sync=True)
        assert baseline_counts == [1, 1, 1, 1, 1]
        assert deferred_counts == [0, 1, 0, 1, 0]
        torch.testing.assert_close(deferred, baseline, rtol=1e-4, atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gloo",
            marks=pytest.mark.skipif(
                not dist.is_gloo_available(), reason="Requires Gloo"
            ),
        ),
        pytest.param(
            "nccl",
            marks=pytest.mark.skipif(
                not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                reason="SyncBatchNorm verification requires two CUDA devices with NCCL",
            ),
        ),
    ],
)
def test_ddp_sync_matches_baseline(tmp_path: Path, backend: str) -> None:
    mp.spawn(
        _ddp_sync_worker,
        args=((tmp_path / "rendezvous").as_uri(), backend),
        nprocs=2,
    )
