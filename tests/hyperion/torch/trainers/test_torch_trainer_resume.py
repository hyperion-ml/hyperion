"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hyperion.torch.data import BucketingSegSampler
from hyperion.torch.hyper_torch_model import HyperTorchModel
from hyperion.torch.trainers import torch_trainer_base
from hyperion.torch.trainers.single_model_trainer import SingleModelTrainer
from hyperion.utils import SegmentSet


class _RegressionModel(HyperTorchModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.seen: list[str] = []

    def forward(
        self, audio: torch.Tensor, id: list[str], target: torch.Tensor
    ) -> torch.Tensor:
        self.seen.extend(id)
        return audio * self.weight


class _NumberedDataset(Dataset):
    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: str | int) -> dict[str, str | torch.Tensor]:
        audio = torch.tensor((int(index) + 1) / len(self))
        return {"id": str(index), "audio": audio, "target": 2 * audio}


def _make_loader() -> DataLoader:
    dataset = _NumberedDataset()
    segments = SegmentSet.from_dict(
        {
            "id": [str(i) for i in range(len(dataset))],
            "duration": [float(i + 1) for i in range(len(dataset))],
        }
    )
    sampler = BucketingSegSampler(
        segments,
        num_buckets=2,
        min_batch_size=1,
        max_batch_size=1,
        sample_all_segments=True,
    )
    return DataLoader(dataset, batch_sampler=sampler)


def _make_trainer(
    exp_path: Path,
    grad_acc_steps: int,
    *,
    max_steps: int | None = None,
    save_steps: int | None = None,
    save_hours: float | None = None,
    val_hours: float | None = None,
) -> SingleModelTrainer:
    model = _RegressionModel()
    return SingleModelTrainer(
        model=model,
        optim=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        loss=nn.MSELoss(),
        input_key="audio",
        target_key="target",
        exp_path=exp_path,
        num_epochs=1,
        max_steps=max_steps,
        save_steps=save_steps,
        save_hours=save_hours,
        val_hours=val_hours,
        grad_acc_steps=grad_acc_steps,
        device=torch.device("cpu"),
        loggers=[],
    )


@pytest.mark.parametrize("grad_acc_steps", [1, 2])
def test_two_resumes_match_an_uninterrupted_run(
    tmp_path: Path, grad_acc_steps: int
) -> None:
    expected = _make_trainer(tmp_path / "reference", grad_acc_steps)
    expected.fit(_make_loader())

    exp_path = tmp_path / "sweep" / "step900" / "[trial]"
    initial = _make_trainer(exp_path, grad_acc_steps, max_steps=2, save_steps=1)
    initial.fit(_make_loader())
    periodic = initial.load_model_checkpoint_from_file(epoch=0, step=1)
    assert periodic["next_batch"] == grad_acc_steps
    assert "batch" not in periodic
    assert (initial.cur_epoch, initial.next_batch) == (0, 2 * grad_acc_steps)

    resumed = _make_trainer(exp_path, grad_acc_steps, max_steps=3)
    resumed.load_last_checkpoint()
    resumed.fit(_make_loader())
    assert (resumed.cur_epoch, resumed.next_batch) == (0, 3 * grad_acc_steps)

    finished = _make_trainer(exp_path, grad_acc_steps)
    finished.load_last_checkpoint()
    finished.fit(_make_loader())

    seen = initial.model.seen + resumed.model.seen + finished.model.seen
    assert seen == expected.model.seen
    assert len(seen) == len(set(seen)) == len(_NumberedDataset())
    assert finished.cur_step == expected.cur_step == len(seen) // grad_acc_steps
    torch.testing.assert_close(finished.model.state_dict(), expected.model.state_dict())

    finished.load_checkpoint(epoch=1, step=finished.cur_step)
    assert (finished.cur_epoch, finished.next_batch) == (1, 0)


@pytest.mark.parametrize(
    "positions",
    [((9999, 2500), (10000, 3)), ((7, 9_999_999_999), (7, 10_000_000_000))],
)
def test_checkpoint_discovery_compares_epoch_and_step_numerically(
    tmp_path: Path, positions: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    exp_path = tmp_path / "ep500" / "step900"
    exp_path.mkdir(parents=True)
    trainer = _make_trainer(exp_path, 1)
    for epoch, step in positions:
        (exp_path / f"model_ep{epoch:04d}_step{step:010d}.pth").touch()

    epoch, step = positions[-1]
    expected = exp_path / f"model_ep{epoch:04d}_step{step:010d}.pth"
    assert trainer.find_last_checkpoint() == (str(expected), epoch, step)


@pytest.mark.parametrize("model_name", ["dac_model", "vc_model", "codec+.[1]"])
def test_checkpoint_discovery_matches_only_the_requested_prefix(
    tmp_path: Path, model_name: str
) -> None:
    trainer = _make_trainer(tmp_path, 1)
    checkpoint = tmp_path / f"{model_name}_ep0003_step0000000042.pth"
    checkpoint.touch()
    (tmp_path / "model_ep9999_step9999999999.pth").touch()
    (tmp_path / "discrim_model_ep9999_step9999999999.pth").touch()
    (tmp_path / f"swa_{model_name}_ep9999_step9999999999.pth").touch()
    (tmp_path / f"{model_name}_ep9999_step0000000bad.pth").touch()
    (tmp_path / f"{model_name}_ep09999_step9999999999.pth").touch()
    (tmp_path / f"{model_name}_ep9999_step9999999999.pth").mkdir()

    assert trainer.find_last_checkpoint(model_name) == (str(checkpoint), 3, 42)


def test_checkpoint_discovery_handles_missing_directory_and_no_match(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path / "new-run", 1)
    assert trainer.find_last_checkpoint() == (None, 0, 0)
    assert trainer.load_last_checkpoint() is None

    trainer.exp_path.mkdir()
    (trainer.exp_path / "model_ep1_step2.pth").touch()
    assert trainer.find_last_checkpoint() == (None, 0, 0)
    assert trainer.load_last_checkpoint() is None


def test_load_last_checkpoint_can_load_epoch_and_step_zero(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, 1)
    logs = {"train_loss": 1.0}
    trainer.save_checkpoint(logs)

    resumed = _make_trainer(tmp_path, 1)
    assert resumed.load_last_checkpoint() == logs


def test_limit_on_the_last_batch_resumes_at_loader_exhaustion(tmp_path: Path) -> None:
    num_batches = len(_NumberedDataset())
    trainer = _make_trainer(tmp_path, 1, max_steps=num_batches)
    trainer.fit(_make_loader())

    resumed = _make_trainer(tmp_path, 1, max_steps=num_batches)
    resumed.load_checkpoint(epoch=0, step=num_batches)
    resumed.fit(_make_loader())
    assert (resumed.cur_epoch, resumed.next_batch) == (0, num_batches)
    assert resumed.model.seen == []

    resumed.max_steps += 1
    resumed.fit(_make_loader())
    assert resumed.model.seen == []
    assert resumed.cur_step == num_batches
    assert (resumed.cur_epoch, resumed.next_batch) == (1, 0)


def test_timed_checkpoint_waits_for_an_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elapsed = 0.0
    monkeypatch.setattr(
        torch_trainer_base,
        "time",
        SimpleNamespace(monotonic=lambda: elapsed, time=lambda: 0.0),
    )
    trainer = _make_trainer(tmp_path, 2, save_hours=1.0)
    training_step = trainer.training_step

    def run_batch(
        batch_idx: int, batch_data: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        nonlocal elapsed
        elapsed = 3600.0
        return training_step(batch_idx, batch_data)

    monkeypatch.setattr(trainer, "training_step", run_batch)
    trainer.fit(_make_loader())

    assert sorted(path.name for path in tmp_path.glob("model_ep0000*.pth")) == [
        "model_ep0000_step0000000001.pth"
    ]
    assert trainer.load_model_checkpoint_from_file(epoch=0, step=1)["next_batch"] == 2


def test_mid_epoch_resume_requires_an_offset_aware_batch_sampler(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path, 1, max_steps=1)
    trainer.fit(_make_loader())

    resumed = _make_trainer(tmp_path, 1)
    resumed.load_checkpoint(epoch=0, step=1)
    with pytest.raises(ValueError, match="Mid-epoch resume requires a batch sampler"):
        resumed.fit(DataLoader(_NumberedDataset(), batch_size=1))


def test_validation_timer_restarts_after_epoch_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _make_trainer(tmp_path, 2, val_hours=1)
    trainer.num_epochs = 2
    elapsed = 0.0
    validation_steps = []
    training_step = trainer.training_step
    on_val_loop_begin = trainer.on_val_loop_begin

    def run_batch(
        batch_idx: int, batch_data: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        nonlocal elapsed
        elapsed += 400
        return training_step(batch_idx, batch_data)

    def begin_validation() -> None:
        nonlocal elapsed
        validation_steps.append(trainer.cur_step)
        elapsed += 600
        on_val_loop_begin()

    monkeypatch.setattr(
        torch_trainer_base, "time", SimpleNamespace(monotonic=lambda: elapsed)
    )
    monkeypatch.setattr(trainer, "training_step", run_batch)
    monkeypatch.setattr(trainer, "on_val_loop_begin", begin_validation)
    trainer.fit(_make_loader(), _make_loader())

    assert validation_steps == [4, 8]
    assert trainer.last_val_time == elapsed


def _timed_validation_worker(rank: int, rendezvous: str, exp_path: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = _make_trainer(exp_path, 2, val_hours=1)
        trainer.rank, trainer.world_size = rank, 2
        clock = 0.0 if rank == 0 else 10000.0
        clock_reads = 0
        validation_steps = []
        decisions = []
        broadcast = dist.broadcast
        training_step = trainer.training_step
        on_val_loop_begin = trainer.on_val_loop_begin

        def monotonic() -> float:
            nonlocal clock_reads
            clock_reads += 1
            return clock

        def share_decision(decision: torch.Tensor, src: int) -> None:
            broadcast(decision, src=src)
            decisions.append((trainer.cur_step, bool(decision.item())))

        def run_batch(
            batch_idx: int, batch_data: dict[str, Any]
        ) -> tuple[int, dict[str, Any]]:
            nonlocal clock
            if rank == 0:
                clock = 3600.0
            return training_step(batch_idx, batch_data)

        def begin_validation() -> None:
            validation_steps.append(trainer.cur_step)
            on_val_loop_begin()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                torch_trainer_base, "time", SimpleNamespace(monotonic=monotonic)
            )
            patch.setattr(dist, "broadcast", share_decision)
            patch.setattr(trainer, "training_step", run_batch)
            patch.setattr(trainer, "on_val_loop_begin", begin_validation)

            loader = DataLoader(_NumberedDataset(), batch_size=1)
            trainer.val_data = loader
            trainer.cur_step = 1000
            if rank == 0:
                trainer.last_val_time = 0.0
            assert not trainer.validate_now()

            trainer.cur_step = 1998
            trainer.fit(loader, loader)
            assert validation_steps == [2000, 2002]

            trainer.val_hours = None
            trainer.val_steps = 1
            trainer.cur_step = 3000
            assert trainer.validate_now()

        assert decisions == [(1000, False), (2000, True)]
        if rank != 0:
            assert clock_reads == 1  # Checkpoint timer initialization in fit.
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
def test_timed_validation_shares_rank_zero_decisions(tmp_path: Path) -> None:
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_timed_validation_worker, args=(rendezvous, tmp_path), nprocs=2)
