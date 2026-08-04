from pathlib import Path

import pytest

import hyperion.torch.trainers.torch_trainer_base as trainer_base
from hyperion.torch.trainers.torch_trainer_base import DDPType, TorchTrainerBase


def _trainer_with_exp_path(exp_path: Path) -> TorchTrainerBase:
    trainer = object.__new__(TorchTrainerBase)
    trainer.exp_path = exp_path
    return trainer


def test_checkpoint_directory_layout(tmp_path: Path) -> None:
    trainer = _trainer_with_exp_path(tmp_path)

    assert (
        trainer.checkpoint_dir_name(5, 1000)
        == "checkpoint_ep0005_step0000001000"
    )
    assert trainer.checkpoint_dir(5, 1000) == (
        tmp_path / "checkpoint_ep0005_step0000001000"
    )
    assert trainer.checkpoint_model_dir("model", 5, 1000) == (
        tmp_path / "checkpoint_ep0005_step0000001000" / "model"
    )
    assert trainer.checkpoint_trainer_state_path(5, 1000) == (
        tmp_path / "checkpoint_ep0005_step0000001000" / "trainer_state.json"
    )
    assert trainer.checkpoint_rng_state_path(5, 1000) == (
        tmp_path / "checkpoint_ep0005_step0000001000" / "rng_state.pth"
    )


def test_find_last_complete_checkpoint_directory(tmp_path: Path) -> None:
    trainer = _trainer_with_exp_path(tmp_path)
    incomplete_dir = trainer.checkpoint_dir(3, 500)
    incomplete_dir.mkdir()

    complete_dir = trainer.checkpoint_dir(3, 1000)
    complete_dir.mkdir()
    (complete_dir / trainer.trainer_state_file_name).touch()
    (complete_dir / trainer.rng_state_file_name).touch()

    newer_complete_dir = trainer.checkpoint_dir(4, 100)
    newer_complete_dir.mkdir()
    (newer_complete_dir / trainer.trainer_state_file_name).touch()
    (newer_complete_dir / trainer.rng_state_file_name).touch()

    assert trainer.is_complete_checkpoint_dir(incomplete_dir) is False
    assert trainer.is_complete_checkpoint_dir(complete_dir) is True
    assert trainer.find_last_checkpoint_dir() == (newer_complete_dir, 4, 100)


def test_complete_checkpoint_requires_all_model_artifacts(tmp_path: Path) -> None:
    trainer = _trainer_with_exp_path(tmp_path)
    trainer.checkpoint_model_names = ("model",)
    checkpoint_dir = trainer.checkpoint_dir(3, 100)
    checkpoint_dir.mkdir()
    (checkpoint_dir / trainer.trainer_state_file_name).touch()
    (checkpoint_dir / trainer.rng_state_file_name).touch()

    assert trainer.is_complete_checkpoint_dir(checkpoint_dir) is False

    model_dir = checkpoint_dir / "model"
    model_dir.mkdir()
    (model_dir / trainer.model_config_file_name).touch()
    (model_dir / trainer.model_weights_file_name).touch()

    assert trainer.is_complete_checkpoint_dir(checkpoint_dir) is False

    (model_dir / trainer.optimizer_state_file_name).touch()
    assert trainer.is_complete_checkpoint_dir(checkpoint_dir) is True


def test_checkpoint_save_dir_publishes_atomically(tmp_path: Path) -> None:
    trainer = _trainer_with_exp_path(tmp_path)
    checkpoint_dir = trainer.checkpoint_dir(5, 1000)

    with trainer.checkpoint_save_dir(5, 1000) as tmp_dir:
        assert tmp_dir.is_dir()
        assert checkpoint_dir.exists() is False
        (tmp_dir / trainer.trainer_state_file_name).touch()
        (tmp_dir / trainer.rng_state_file_name).touch()

    assert checkpoint_dir.is_dir()
    assert trainer.is_complete_checkpoint_dir(checkpoint_dir) is True


def test_checkpoint_save_dir_cleans_up_after_error(tmp_path: Path) -> None:
    trainer = _trainer_with_exp_path(tmp_path)
    checkpoint_dir = trainer.checkpoint_dir(5, 1000)

    with pytest.raises(RuntimeError):
        with trainer.checkpoint_save_dir(5, 1000) as tmp_dir:
            assert tmp_dir.is_dir()
            raise RuntimeError("failed checkpoint")

    assert checkpoint_dir.exists() is False
    assert list(tmp_path.iterdir()) == []


def test_fsdp_load_uses_distributed_state_dict_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer_with_exp_path(Path("unused"))
    trainer.ddp = True
    trainer.ddp_type = DDPType.FSDP
    trainer.do_swa = False

    model_state_calls = []
    optimizer_state_calls = []

    def set_model_state_dict(*args, **kwargs) -> None:
        model_state_calls.append((args, kwargs))

    def set_optimizer_state_dict(*args, **kwargs) -> None:
        optimizer_state_calls.append((args, kwargs))

    class Optimizer:
        def load_state_dict(self, state_dict) -> None:
            raise AssertionError("FSDP must not use Optimizer.load_state_dict")

    monkeypatch.setattr(trainer_base, "set_model_state_dict", set_model_state_dict)
    monkeypatch.setattr(
        trainer_base, "set_optimizer_state_dict", set_optimizer_state_dict
    )

    model_state = {"weight": object()}
    optimizer_state = {"state": {}, "param_groups": []}
    trainer._load_model_state_dicts_from_checkpoint(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
        },
        object(),
        Optimizer(),
    )

    assert model_state_calls[0][0][1] is model_state
    assert optimizer_state_calls[0][1]["optim_state_dict"] is optimizer_state
