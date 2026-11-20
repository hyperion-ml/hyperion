from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyperion.torch.hyper_torch_model import HyperTorchModel
from hyperion.torch.layers import VectorQuantizerOutput
from hyperion.torch.models.audio_discriminator import AudioMultiDiscriminator
from hyperion.torch.models.dac import DAC
from hyperion.torch.models.dac.dac import DACOutput
from hyperion.torch.trainers import torch_trainer_base
from hyperion.torch.trainers.dac_trainer import DACTrainer
from hyperion.torch.trainers.freevc_trainer import FreeVCTrainer
from hyperion.torch.trainers.single_model_trainer import SingleModelTrainer
from hyperion.torch.trainers.torch_trainer_base import AMPDType, DDPType

pytestmark = pytest.mark.skipif(
    not torch_trainer_base.FSDP_AVAILABLE, reason="Requires FSDP2"
)


class _BlockModel(HyperTorchModel):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
        self.head = nn.ModuleDict({"out": nn.Linear(5, 1)})
        self.register_buffer("offset", torch.tensor([0.1]))

    def fsdp_shard_blocks(self) -> tuple[nn.Module, ...]:
        return tuple(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x).relu()
        return self.head["out"](x) + self.offset


def _make_trainer(
    exp_path: Path,
    device: torch.device,
    cpu_offload: bool,
    amp_dtype: AMPDType | None,
    optimizer: str = "adam",
) -> SingleModelTrainer:
    torch.manual_seed(0)
    return SingleModelTrainer(
        model=_BlockModel(),
        optim={"opt_type": optimizer, "lr": 0.01},
        loss=nn.MSELoss(),
        input_key="x",
        target_key="y",
        ddp=True,
        ddp_type=DDPType.FSDP,
        fsdp_cpu_offload=cpu_offload,
        use_amp=amp_dtype is not None,
        amp_dtype=amp_dtype or AMPDType.FLOAT16,
        grad_clip=0.5,
        save_hours=1.0,
        exp_path=exp_path,
        device=device,
        loggers=[],
    )


def _check_overflow_skips_optimizer_on_all_ranks(
    trainer: SingleModelTrainer, scaler: torch.amp.GradScaler, device: torch.device
) -> None:
    scaler.scale(torch.tensor(1.0, device=device))
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    parameters = list(trainer.model.parameters())
    if trainer.rank == 1:
        parameters[0].grad.to_local().fill_(float("inf"))

    weights = [p.to_local().clone() for p in parameters]
    previous_scale = scaler.get_scale()
    scaler.unscale_(trainer.optimizer)
    scaler.step(trainer.optimizer)
    scaler.update()
    trainer.optimizer.zero_grad()

    assert scaler.get_scale() < previous_scale
    for parameter, previous in zip(parameters, weights):
        torch.testing.assert_close(parameter.to_local(), previous)


def _check_gan_saves_use_both_models(
    trainer: SingleModelTrainer, resumed: SingleModelTrainer, tmp_path: Path
) -> None:
    for trainer_type, name in ((DACTrainer, "dac"), (FreeVCTrainer, "vc")):
        gan = object.__new__(trainer_type)
        gan.exp_path = tmp_path / name
        if trainer.rank == 0:
            gan.exp_path.mkdir()
        gan.rank = trainer.rank
        gan.cur_epoch = trainer.cur_epoch
        gan.cur_step = trainer.cur_step
        gan.next_batch = trainer.next_batch
        gan.in_swa = False
        gan.grad_scaler = trainer.grad_scaler
        for key, single in ((name, trainer), ("discrim", resumed)):
            setattr(gan, key + "_model", single.model)
            setattr(gan, key + "_optimizer", single.optimizer)
            setattr(gan, key + "_lr_scheduler", None)
            setattr(gan, key + "_wd_scheduler", None)
        setattr(gan, "swa_" + name + "_model", None)
        setattr(gan, "swa_" + name + "_scheduler", None)

        gan.save_checkpoint({"writer_rank": trainer.rank})
        dist.barrier()
        for checkpoint_name in (name + "_model", "discrim_model"):
            saved = gan.load_model_checkpoint_from_file(
                checkpoint_name, gan.cur_epoch, gan.cur_step
            )
            assert saved["logs"] == {"writer_rank": 0}
            assert saved["model_state_dict"]["blocks.0.weight"].shape == (5, 5)
            assert saved["optimizer_state_dict"]["state"]["blocks.0.weight"]


def _checkpoint_worker(
    rank: int,
    rendezvous: str,
    tmp_path: Path,
    use_cuda: bool,
    cpu_offload: bool,
    amp_dtype: AMPDType | None,
) -> None:
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule

    from hyperion.torch.trainers._fsdp2 import CPUOffloadGradScaler

    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    elif cpu_offload:
        # A CPU-only build cannot allocate CUDA pinned memory.
        torch_trainer_base.CPUOffloadPolicy = lambda: CPUOffloadPolicy(pin_memory=False)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        trainer = _make_trainer(tmp_path, device, cpu_offload, amp_dtype)
        assert isinstance(trainer.model, FSDPModule)
        assert all(isinstance(block, FSDPModule) for block in trainer.model.blocks)
        assert not isinstance(trainer.model.blocks, FSDPModule)
        assert not isinstance(trainer.model.head, FSDPModule)
        assert trainer.grad_scaler.is_enabled() == (amp_dtype == AMPDType.FLOAT16)

        if use_cuda:
            torch.manual_seed(1 + rank)
            x = torch.randn(3, 5, device=device)
            with torch.autocast("cuda", dtype=trainer.amp_dtype):
                pred = trainer.model(x)
                loss = trainer.loss(pred, torch.full_like(pred, 0.3))
            # Use a small scale so this step is finite in FP16.
            if trainer.grad_scaler.is_enabled():
                trainer.grad_scaler.load_state_dict(
                    {**trainer.grad_scaler.state_dict(), "scale": 128.0}
                )
            trainer.grad_scaler.scale(loss).backward()
        else:
            # On CPU, exercise the real shards and optimizer state without
            # invoking FSDP2 forward, which needs accelerator streams.
            for parameter in trainer.model.parameters():
                parameter.grad = torch.ones_like(parameter)

        before = [p.to_local().clone() for p in trainer.model.parameters()]
        grad_norm = trainer.update_models()["grad_norm"]
        assert torch.isfinite(torch.tensor(grad_norm))
        assert any(
            not torch.equal(p.to_local(), old)
            for p, old in zip(trainer.model.parameters(), before)
        )
        if not use_cuda:
            assert grad_norm == pytest.approx(66**0.5)
            for parameter in trainer.model.parameters():
                moment = trainer.optimizer.state[parameter]["exp_avg"].to_local()
                torch.testing.assert_close(
                    moment, torch.full_like(moment, 0.1 * 0.5 / grad_norm)
                )
        if trainer.grad_scaler.is_enabled():
            _check_overflow_skips_optimizer_on_all_ranks(
                trainer, trainer.grad_scaler, device
            )
        elif not use_cuda and cpu_offload:
            _check_overflow_skips_optimizer_on_all_ranks(
                trainer, CPUOffloadGradScaler(device="cpu"), device
            )
        trainer.last_save_time = 0 if rank == 0 else float("inf")
        trainer.last_save_step = 0
        trainer.cur_step = 999
        assert not trainer.save_now()
        trainer.cur_step = 1000
        original_time = torch_trainer_base.time
        torch_trainer_base.time = SimpleNamespace(monotonic=lambda: 3601)
        try:
            assert trainer.save_now()
        finally:
            torch_trainer_base.time = original_time

        trainer.cur_epoch = 2
        trainer.next_batch = 5
        trainer.save_checkpoint({"writer_rank": rank})
        dist.barrier()
        checkpoint_path = tmp_path / "model_ep0002_step0000001000.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["logs"] == {"writer_rank": 0}
        assert checkpoint["model_state_dict"]["blocks.0.weight"].shape == (5, 5)
        assert checkpoint["model_state_dict"]["blocks.0.weight"].device.type == "cpu"

        resumed = _make_trainer(tmp_path, device, cpu_offload, amp_dtype)
        assert resumed.load_last_checkpoint() == {"writer_rank": 0}
        assert (resumed.cur_epoch, resumed.next_batch, resumed.cur_step) == (2, 5, 1000)
        assert resumed.grad_scaler.state_dict() == trainer.grad_scaler.state_dict()
        for parameter, expected in zip(
            resumed.model.parameters(), trainer.model.parameters()
        ):
            torch.testing.assert_close(parameter.to_local(), expected.to_local())
            torch.testing.assert_close(
                resumed.optimizer.state[parameter]["exp_avg"].to_local(),
                trainer.optimizer.state[expected]["exp_avg"].to_local(),
            )
        if not use_cuda and cpu_offload:
            _check_gan_saves_use_both_models(trainer, resumed, tmp_path)
        if not use_cuda:
            stateless = _make_trainer(
                tmp_path / "sgd", device, cpu_offload, None, optimizer="sgd"
            )
            if rank == 0:
                stateless.exp_path.mkdir()
            stateless.optimizer.param_groups[0]["lr"] = 0.123
            stateless.save_checkpoint()
            dist.barrier()
            assert not stateless.optimizer.state
            stateless.optimizer.param_groups[0]["lr"] = 0.456
            stateless.load_last_checkpoint()
            assert stateless.optimizer.param_groups[0]["lr"] == 0.123
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp2_collective_checkpoint_and_resume(
    tmp_path: Path, cpu_offload: bool
) -> None:
    mp.spawn(
        _checkpoint_worker,
        args=((tmp_path / "gloo").as_uri(), tmp_path, False, cpu_offload, None),
        nprocs=2,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="Requires two CUDA GPUs with NCCL",
)
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("amp_dtype", [AMPDType.FLOAT16, AMPDType.BFLOAT16])
def test_fsdp2_cuda_step_checkpoint_and_resume(
    tmp_path: Path, cpu_offload: bool, amp_dtype: AMPDType
) -> None:
    if amp_dtype == AMPDType.BFLOAT16:
        for rank in range(2):
            with torch.cuda.device(rank):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    pytest.skip("BF16 needs native support on both GPUs")
    mp.spawn(
        _checkpoint_worker,
        args=((tmp_path / "nccl").as_uri(), tmp_path, True, cpu_offload, amp_dtype),
        nprocs=2,
    )


def test_fsdp2_can_traverse_nested_dac_outputs() -> None:
    audio = torch.rand(2, 1, 32, requires_grad=True)
    codebook_loss = torch.tensor(0.25, requires_grad=True)
    output = DACOutput(
        x_recons=audio, vq=VectorQuantizerOutput(codebook_loss=codebook_loss)
    )

    leaves, spec = tree_flatten(output)
    assert any(value is audio for value in leaves)
    assert any(value is codebook_loss for value in leaves)
    restored = tree_unflatten(leaves, spec)
    assert isinstance(restored, DACOutput)
    assert restored.x_recons is audio
    assert restored.x_recons_lengths is None
    assert restored.vq.codebook_loss is codebook_loss


def _make_dac_trainer(
    exp_path: Path, device: torch.device, cpu_offload: bool, *, ddp: bool
) -> DACTrainer:
    torch.manual_seed(0)
    return DACTrainer(
        dac_model=DAC(
            encoder={
                "in_feats": 1,
                "init_inner_channels": 4,
                "strides": [2],
                "dilations": [1],
            },
            quantizer={"num_quantizers": 1, "codebook_sizes": 8},
            decoder={
                "out_feats": 1,
                "init_inner_channels": 8,
                "strides": [2],
                "dilations": [1],
            },
        ),
        discrim_model=AudioMultiDiscriminator(use_mpd=False, use_mrsd=False),
        mrfb_loss={"frame_lengths": [2.0], "num_filters": 5, "dither": 0.0},
        dac_optim={"opt_type": "adam", "lr": 0.001},
        discrim_optim={"opt_type": "adam", "lr": 0.001},
        exp_path=exp_path,
        device=device,
        ddp=ddp,
        ddp_type=DDPType.FSDP,
        fsdp_cpu_offload=cpu_offload,
        fsdp_reshard_after_forward=True,
        use_amp=device.type == "cuda",
        amp_dtype=AMPDType.FLOAT16,
        loggers=[],
    )


def _run_dac_step(trainer: DACTrainer) -> None:
    if trainer.grad_scaler.is_enabled():
        scaler_state = trainer.grad_scaler.state_dict()
        scaler_state["scale"] = 8.0
        trainer.grad_scaler.load_state_dict(scaler_state)
    models = (trainer.dac_model, trainer.discrim_model)
    first_weights = [next(model.parameters()) for model in models]
    before = [p.detach().clone() for p in first_weights]
    batch_size, metrics = trainer.training_step(
        0,
        {
            "id": ["a", "b"],
            "audio": torch.rand(2, 256) * 2 - 1,
            "audio_lengths": torch.full((2,), 256),
        },
    )
    assert batch_size == 2
    assert torch.isfinite(torch.tensor(metrics["loss_gen/total"]))
    assert torch.isfinite(torch.tensor(metrics["loss_discrim/total"]))
    logs = trainer.update_models()
    assert torch.isfinite(torch.tensor(logs["grad_norm/dac"]))
    assert torch.isfinite(torch.tensor(logs["grad_norm/discrim"]))
    for weight, previous in zip(first_weights, before):
        if trainer.ddp:
            weight, previous = weight.to_local(), previous.to_local()
        assert not torch.equal(weight, previous)


def _dac_cuda_worker(
    rank: int, rendezvous: str, tmp_path: Path, cpu_offload: bool
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        trainer = _make_dac_trainer(tmp_path, device, cpu_offload, ddp=True)
        assert trainer.grad_scaler.is_enabled()
        _run_dac_step(trainer)
        trainer.cur_step = 1
        trainer.save_checkpoint()
        dist.barrier()

        resumed = _make_dac_trainer(tmp_path, device, cpu_offload, ddp=True)
        resumed.load_last_checkpoint()
        assert resumed.cur_step == 1
        assert resumed.grad_scaler.state_dict() == trainer.grad_scaler.state_dict()
        for model, original in (
            (resumed.dac_model, trainer.dac_model),
            (resumed.discrim_model, trainer.discrim_model),
        ):
            for p, p_original in zip(model.parameters(), original.parameters()):
                torch.testing.assert_close(p.to_local(), p_original.to_local())
        _run_dac_step(resumed)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="Requires two CUDA GPUs with NCCL",
)
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp2_dac_fp16_step_checkpoint_and_resume(
    tmp_path: Path, cpu_offload: bool
) -> None:
    mp.spawn(
        _dac_cuda_worker,
        args=((tmp_path / "nccl-dac").as_uri(), tmp_path, cpu_offload),
        nprocs=2,
    )


@pytest.mark.parametrize(
    ("trainer_cls", "prefix"), [(DACTrainer, "dac"), (FreeVCTrainer, "vc")]
)
def test_only_cpu_offload_is_forwarded_to_the_discriminator(
    trainer_cls: type, prefix: str
) -> None:
    trainer = object.__new__(trainer_cls)
    for name in (prefix, "discrim"):
        for attr in ("model", "optim", "lrsched", "wdsched"):
            setattr(trainer, f"{name}_{attr}", None)
    trainer.device = torch.device("cpu")
    trainer.ddp = trainer.use_amp = trainer.fsdp_cpu_offload = True
    trainer.ddp_type = DDPType.FSDP
    trainer.fsdp_reshard_after_forward = True
    trainer.fsdp_mp_param_dtype = torch.float16
    trainer.fsdp_mp_reduce_dtype = torch.float32
    trainer.fsdp_mp_output_dtype = torch.float16
    trainer.do_swa = False
    trainer.swa_lr = trainer.swa_anneal_steps = 0
    trainer.get_grad_scaler = Mock()
    trainer._prepare_model_for_training = Mock(return_value=(None,) * 6)
    trainer.prepare_models_for_training()

    gen, disc = trainer._prepare_model_for_training.call_args_list
    assert gen.kwargs["fsdp_mp_param_dtype"] == torch.float16
    assert disc.kwargs == {
        "device": trainer.device,
        "ddp": True,
        "ddp_type": DDPType.FSDP,
        "fsdp_cpu_offload": True,
    }
