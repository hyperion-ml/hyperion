from pathlib import Path
from typing import Type

import pytest
import torch
import torch.nn as nn

from hyperion.bin.to_safetensors import convert_checkpoint
from hyperion.torch import HyperTorchModel
from hyperion.torch.trainers.dac_trainer import DACTrainer
from hyperion.torch.trainers.freevc_trainer import FreeVCTrainer
from hyperion.torch.trainers.single_model_trainer import SingleModelTrainer
from hyperion.torch.trainers.torch_trainer_base import DDPType


class TinyModel(HyperTorchModel):
    """Small serializable model used to test checkpoint formats."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the test linear projection.

        Args:
            x: Input tensor with two features.

        Returns:
            Projected output tensor.
        """
        return self.linear(x)


class TiedTensorModel(HyperTorchModel):
    """Model with tied parameters and a non-contiguous buffer for serialization tests."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, 2)
        self.output_projection = nn.Linear(2, 2)
        self.output_projection.weight = self.input_projection.weight
        self.register_buffer(
            "non_contiguous_buffer",
            torch.arange(6, dtype=torch.float32).reshape(2, 3).T,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tied projections.

        Args:
            x: Input tensor with two features.

        Returns:
            Projected output tensor.
        """
        return self.output_projection(self.input_projection(x))


def _update_model(model: TinyModel, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def _assert_model_equal(model_a: TinyModel, model_b: TinyModel) -> None:
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)


def _base_trainer(trainer_class: Type[object], exp_path: Path) -> object:
    trainer = object.__new__(trainer_class)
    trainer.exp_path = exp_path
    trainer.rank = 0
    trainer.cur_epoch = 3
    trainer.cur_batch = 4
    trainer.cur_step = 50
    trainer.ddp = False
    trainer.do_swa = False
    trainer.in_swa = False
    return trainer


def _single_model_trainer(exp_path: Path) -> SingleModelTrainer:
    trainer = _base_trainer(SingleModelTrainer, exp_path)
    trainer.model = TinyModel()
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1
    )
    trainer.wd_scheduler = None
    trainer.swa_model = None
    trainer.swa_scheduler = None
    _update_model(trainer.model, trainer.optimizer)
    trainer.lr_scheduler.step()
    return trainer


def _multi_model_trainer(
    trainer_class: Type[DACTrainer] | Type[FreeVCTrainer], exp_path: Path
) -> DACTrainer | FreeVCTrainer:
    trainer = _base_trainer(trainer_class, exp_path)
    model_prefix = "dac" if trainer_class is DACTrainer else "vc"
    generator = TinyModel()
    discriminator = TinyModel()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.02)
    generator_scheduler = torch.optim.lr_scheduler.StepLR(
        generator_optimizer, step_size=1
    )
    discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
        discriminator_optimizer, step_size=1
    )
    _update_model(generator, generator_optimizer)
    _update_model(discriminator, discriminator_optimizer)
    generator_scheduler.step()
    discriminator_scheduler.step()

    setattr(trainer, f"{model_prefix}_model", generator)
    setattr(trainer, f"{model_prefix}_optimizer", generator_optimizer)
    setattr(trainer, f"{model_prefix}_lr_scheduler", generator_scheduler)
    setattr(trainer, f"{model_prefix}_wd_scheduler", None)
    setattr(trainer, f"swa_{model_prefix}_model", None)
    setattr(trainer, f"swa_{model_prefix}_scheduler", None)
    trainer.discrim_model = discriminator
    trainer.discrim_optimizer = discriminator_optimizer
    trainer.discrim_lr_scheduler = discriminator_scheduler
    trainer.discrim_wd_scheduler = None
    return trainer


def _legacy_checkpoint(model: TinyModel) -> dict:
    return {
        "model_cfg": model.get_config(),
        "model_state_dict": model.state_dict(),
    }


def test_hyper_torch_model_auto_loads_legacy_and_directory_formats(
    tmp_path: Path,
) -> None:
    model = TinyModel()
    legacy_path = tmp_path / "model.pth"
    directory_path = tmp_path / "model"
    model.save(legacy_path)
    model.save(directory_path)

    _assert_model_equal(model, HyperTorchModel.auto_load(legacy_path))
    _assert_model_equal(model, HyperTorchModel.auto_load(directory_path))


def test_model_directory_save_handles_tied_and_non_contiguous_tensors(
    tmp_path: Path,
) -> None:
    model = TiedTensorModel()
    model_dir = tmp_path / "model"

    model.save_to_dir(model_dir)
    loaded_model = HyperTorchModel.auto_load(model_dir)

    assert torch.equal(
        model.non_contiguous_buffer, loaded_model.non_contiguous_buffer
    )
    assert (
        loaded_model.input_projection.weight.data_ptr()
        == loaded_model.output_projection.weight.data_ptr()
    )


def test_single_model_checkpoint_resumes_and_uses_safe_state_loads(
    tmp_path: Path,
) -> None:
    trainer = _single_model_trainer(tmp_path)
    trainer.save_checkpoint({"loss": 1.25})
    checkpoint_dir = trainer.checkpoint_dir(trainer.cur_epoch, trainer.cur_step)
    model_dir = checkpoint_dir / "model"

    assert (model_dir / "config.json").is_file()
    assert (model_dir / "model.safetensors").is_file()
    assert (checkpoint_dir / "trainer_state.json").is_file()
    assert torch.load(model_dir / "optimizer.pt", weights_only=True)
    assert torch.load(model_dir / "lr_scheduler.pt", weights_only=True)

    resumed_trainer = _single_model_trainer(tmp_path)
    logs = resumed_trainer.load_checkpoint(3, 50)

    assert logs == {"loss": 1.25}
    assert resumed_trainer.cur_epoch == 3
    assert resumed_trainer.cur_batch == 4
    assert resumed_trainer.cur_step == 50
    _assert_model_equal(trainer.model, resumed_trainer.model)


@pytest.mark.parametrize(
    ("trainer_class", "model_name"),
    [(DACTrainer, "dac_model"), (FreeVCTrainer, "vc_model")],
)
def test_multi_model_trainers_resume_from_one_checkpoint_root(
    tmp_path: Path,
    trainer_class: Type[DACTrainer] | Type[FreeVCTrainer],
    model_name: str,
) -> None:
    trainer = _multi_model_trainer(trainer_class, tmp_path)
    trainer.save_checkpoint({"loss": 1.25})
    checkpoint_dir = trainer.checkpoint_dir(trainer.cur_epoch, trainer.cur_step)

    assert (checkpoint_dir / model_name / "model.safetensors").is_file()
    assert (checkpoint_dir / "discrim_model" / "model.safetensors").is_file()
    assert (checkpoint_dir / "trainer_state.json").is_file()

    resumed_trainer = _multi_model_trainer(trainer_class, tmp_path)
    logs = resumed_trainer.load_checkpoint(3, 50)

    assert logs == {"loss": 1.25}
    _assert_model_equal(
        getattr(trainer, f"{model_name[:-6]}_model"),
        getattr(resumed_trainer, f"{model_name[:-6]}_model"),
    )
    _assert_model_equal(trainer.discrim_model, resumed_trainer.discrim_model)


def test_converter_creates_loadable_model_directory(tmp_path: Path) -> None:
    model = TinyModel()
    legacy_path = tmp_path / "model.pth"
    torch.save(_legacy_checkpoint(model), legacy_path)
    output_dir = tmp_path / "checkpoint"

    convert_checkpoint(legacy_path, output_dir)

    _assert_model_equal(model, HyperTorchModel.auto_load(output_dir / "model"))


def test_converter_migrates_trainer_state_for_resume(tmp_path: Path) -> None:
    trainer = _single_model_trainer(tmp_path)
    legacy_path = tmp_path / "model.pth"
    torch.save(
        trainer.model_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.lr_scheduler,
            trainer.wd_scheduler,
            trainer.swa_model,
            trainer.swa_scheduler,
            logs={"loss": 1.25},
        ),
        legacy_path,
    )
    output_dir = trainer.checkpoint_dir(trainer.cur_epoch, trainer.cur_step)

    convert_checkpoint(legacy_path, output_dir, get_trainer_state=True)
    resumed_trainer = _single_model_trainer(tmp_path)
    logs = resumed_trainer.load_checkpoint(3, 50)

    assert logs == {"loss": 1.25}
    _assert_model_equal(trainer.model, resumed_trainer.model)


def test_converter_rejects_conflicting_multi_model_trainer_state(tmp_path: Path) -> None:
    trainer = _single_model_trainer(tmp_path)
    first_checkpoint = trainer.model_checkpoint(
        trainer.model,
        trainer.optimizer,
        trainer.lr_scheduler,
        trainer.wd_scheduler,
        trainer.swa_model,
        trainer.swa_scheduler,
        logs={"loss": 1.25},
    )
    second_checkpoint = dict(first_checkpoint)
    second_checkpoint["epoch"] = 4
    first_path = tmp_path / "first.pth"
    second_path = tmp_path / "second.pth"
    torch.save(first_checkpoint, first_path)
    torch.save(second_checkpoint, second_path)
    output_dir = tmp_path / "checkpoint"

    convert_checkpoint(
        first_path,
        output_dir,
        model_name="dac_model",
        get_trainer_state=True,
    )
    with pytest.raises(ValueError, match="conflicts"):
        convert_checkpoint(
            second_path,
            output_dir,
            model_name="discrim_model",
            get_trainer_state=True,
        )

    assert not (output_dir / "discrim_model").exists()


def test_fsdp_nonzero_rank_participates_in_single_model_state_collection() -> None:
    trainer = object.__new__(SingleModelTrainer)
    trainer.ddp = True
    trainer.ddp_type = DDPType.FSDP
    trainer.rank = 1
    trainer.model = object()
    trainer.optimizer = object()
    trainer.lr_scheduler = None
    trainer.wd_scheduler = None
    trainer.swa_model = None
    trainer.swa_scheduler = None

    state_collection_calls = []

    def model_checkpoint(*args, **kwargs) -> dict:
        state_collection_calls.append((args, kwargs))
        return {}

    trainer.model_checkpoint = model_checkpoint
    trainer.save_checkpoint()

    assert len(state_collection_calls) == 1


def test_fsdp_nonzero_rank_collects_standalone_swa_state_without_optimizer() -> None:
    trainer = object.__new__(SingleModelTrainer)
    trainer.ddp = True
    trainer.ddp_type = DDPType.FSDP
    trainer.rank = 1
    trainer.model = object()
    trainer.swa_model = object()

    swa_state_collection_calls = []

    def swa_model_checkpoint(*args, **kwargs) -> dict:
        swa_state_collection_calls.append((args, kwargs))
        return {}

    trainer.swa_model_checkpoint = swa_model_checkpoint
    trainer.save_swa_model()

    assert len(swa_state_collection_calls) == 1


@pytest.mark.parametrize(
    ("trainer_class", "model_prefix"),
    [(DACTrainer, "dac"), (FreeVCTrainer, "vc")],
)
def test_fsdp_nonzero_rank_collects_both_multi_model_states(
    trainer_class: Type[DACTrainer] | Type[FreeVCTrainer], model_prefix: str
) -> None:
    trainer = object.__new__(trainer_class)
    trainer.ddp = True
    trainer.ddp_type = DDPType.FSDP
    trainer.rank = 1
    setattr(trainer, f"{model_prefix}_model", object())
    setattr(trainer, f"{model_prefix}_optimizer", object())
    setattr(trainer, f"{model_prefix}_lr_scheduler", None)
    setattr(trainer, f"{model_prefix}_wd_scheduler", None)
    setattr(trainer, f"swa_{model_prefix}_model", None)
    setattr(trainer, f"swa_{model_prefix}_scheduler", None)
    trainer.discrim_model = object()
    trainer.discrim_optimizer = object()
    trainer.discrim_lr_scheduler = None
    trainer.discrim_wd_scheduler = None

    state_collection_calls = []

    def model_checkpoint(*args, **kwargs) -> dict:
        state_collection_calls.append((args, kwargs))
        return {}

    trainer.model_checkpoint = model_checkpoint
    trainer.save_checkpoint()

    assert len(state_collection_calls) == 2
