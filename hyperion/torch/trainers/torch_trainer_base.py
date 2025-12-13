"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
import math
import re
import time
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.amp as amp
import torch.distributed as dist
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.elastic.multiprocessing.errors import record

try:
    from torch.distributed.fsdp import CPUOffloadPolicy
    from torch.distributed.fsdp import FSDPModule as FSDP
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    FSDP_AVAILABLE = True
except (ImportError, AttributeError):
    CPUOffloadPolicy = None  # type: ignore
    FSDP = None  # type: ignore
    MixedPrecisionPolicy = None  # type: ignore
    fully_shard = None  # type: ignore

    class ShardedGradScaler(amp.GradScaler):  # type: ignore
        """Fallback so the symbol remains a type and callable."""

        pass

    FSDP_AVAILABLE = False

from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader

from ...utils import PathLike
from ...utils.misc import filter_func_args
from ..loggers import CSVLogger, LoggerList, ProgLogger, TensorBoardLogger, WAndBLogger
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..optim import OptimizerFactory as OF
from ..torch_model import TorchModel
from ..utils import MetricAcc, TorchDDP
from ..utils.grad_tracker import GradNormTracker
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF


class DDPType(str, Enum):
    DDP = "ddp"
    FSDP = "fsdp"  # torch FSDP2 via fully_shard

    @staticmethod
    def choices() -> List[str]:
        """
        Lists the available distributed data parallel backends.

        Args:
            None.

        Returns:
            List[str]: String identifiers for each supported DDP type.
        """
        return [o.value for o in DDPType]


class AMPDType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @staticmethod
    def choices() -> List[str]:
        """
        Lists the supported automatic mixed precision data types.

        Args:
            None.

        Returns:
            List[str]: Names of the AMP dtypes (float16, bfloat16).
        """
        return [o.value for o in AMPDType]

    @staticmethod
    def default():
        return AMPDType.FLOAT16

    @staticmethod
    def to_dtype(dtype: "AMPDType") -> torch.dtype:
        """
        Converts an `AMPDType` enum value to the corresponding `torch.dtype`.

        Args:
            dtype (AMPDType): Requested automatic mixed precision type.

        Returns:
            torch.dtype: `torch.float16` or `torch.bfloat16` depending on `dtype`.
        """
        if dtype == AMPDType.FLOAT16:
            return torch.float16
        if dtype == AMPDType.BFLOAT16:
            return torch.bfloat16

        raise ValueError(f"Unsupported AMPDType: {dtype}")


class FSDPMPDType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    NONE = "none"

    @staticmethod
    def choices() -> List[str]:
        """
        Lists the supported automatic mixed precision data types.

        Args:
            None.

        Returns:
            List[str]: Names of the AMP dtypes (float16, bfloat16, float32, none).
        """
        return [o.value for o in FSDPMPDType]

    @staticmethod
    def default() -> None:
        return None

    @staticmethod
    def to_dtype(dtype: Union["FSDPMPDType", None]) -> Optional[torch.dtype]:
        """
        Converts an `FSDPMPDType` enum value to the corresponding `torch.dtype`.

        Args:
            dtype (FSDPMPDType): Requested mixed precision type.

        Returns:
            torch.dtype: `torch.float16`, `torch.bfloat16`, or `torch.float32` depending on `dtype`.
        """
        if dtype is None or dtype == FSDPMPDType.NONE:
            return None

        if dtype == FSDPMPDType.FLOAT16:
            return torch.float16
        if dtype == FSDPMPDType.BFLOAT16:
            return torch.bfloat16
        if dtype == FSDPMPDType.FLOAT32:
            return torch.float32

        raise ValueError(f"Unsupported FSDPMPDType: {dtype}")


class TorchTrainerBase:
    """Base class for training PyTorch models using various training utilities.

    This class supports advanced training features including:
    - Distributed training with multiple DDP backends (standard DDP, and torch FSDP2 via fully_shard)
    - Mixed precision training with AMP (float16, bfloat16)
    - Learning rate and weight decay schedulers
    - Gradient accumulation and clipping
    - Stochastic Weight Averaging (SWA)
    - Optional logger integrations (stdout, TensorBoard, W&B)

    Attributes:
        exp_path (Path): Path to save training logs, checkpoints, and logs.
        num_epochs (int): Total number of training epochs.
        cur_epoch (int): Current training epoch.
        max_steps (int, optional): Maximum number of training steps (overrides num_epochs if set).
        cur_step (int): Current training step (incremented each optimizer update).
        grad_acc_steps (int): Number of batches to accumulate gradients over before optimizer step.
        eff_batch_size (int, optional): Effective total batch size across GPUs and grad accumulation.
        val_steps (int, optional): Number of steps between validation evaluations.
        val_hours (float, optional): Number of hours between validation evaluations (alternative to val_steps).
        save_steps (int, optional): Number of steps between model checkpoint saves.
        save_hours (float, optional): Number of hours between checkpoint saves (alternative to save_steps).
        device (torch.device or int, optional): Device to run training on (e.g. 'cuda:0').
        loggers (LoggerList): Logger interface for training output (e.g., console, file, TensorBoard).
        ddp (bool): Whether to use Distributed Data Parallel (DDP) training.
        ddp_type (DDPType): Type of distributed backend to use (DDP or torch FSDP variants).
        fsdp_reshard_after_forward (bool|int|None): FSDP2 reshard policy after forward.
        fsdp_mp_param_dtype (torch.dtype|None): FSDP2 mixed-precision parameter dtype.
        fsdp_mp_reduce_dtype (torch.dtype|None): FSDP2 mixed-precision reduction dtype.
        fsdp_mp_output_dtype (torch.dtype|None): FSDP2 mixed-precision output dtype.
        fsdp_cpu_offload (bool): Whether to offload parameters/gradients to CPU in FSDP.
        use_amp (bool): Enables mixed-precision training using AMP (Automatic Mixed Precision).
        amp_dtype (AMPDType): Data type for AMP (float16 or bfloat16).
        bf16_grad_scaler (bool): Enable GradScaler when using bfloat16.
        log_interval (int): Number of steps between log output (for loggers).
        log_gpu_usage (bool): Whether to log GPU usage (memory, utilization) during training.
        use_tensorboard (bool): Enables TensorBoard logger.
        use_wandb (bool): Enables Weights & Biases logger.
        wandb (dict): Configuration for W&B logging (project, name, etc.).
        grad_clip (float): Maximum gradient norm to clip (0 disables clipping).
        grad_clip_norm (str or int): Norm type for gradient clipping (e.g., 1, 2, or 'inf').
        swa_start (int): Step at which to begin Stochastic Weight Averaging (SWA).
        swa_lr (float): Learning rate to use during SWA phase.
        swa_anneal_steps (int): Number of steps over which to anneal SWA LR.
        swa_update_steps (int): Number of steps between averaging model weights in SWA.
        bn_update_steps (int): Max number of batches to use for batchnorm statistics update after SWA.
    """

    def __init__(
        self,
        exp_path: PathLike = "./train",
        num_epochs: int = 100,
        cur_epoch: int = 0,
        max_steps: Optional[int] = None,
        cur_step: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        val_steps: Optional[int] = None,
        val_hours: Optional[float] = None,
        save_steps: Optional[int] = None,
        save_hours: Optional[float] = None,
        device: Union[torch.device, int, None] = None,
        loggers: Optional[LoggerList] = None,
        ddp: bool = False,
        ddp_type: DDPType = DDPType.DDP,
        fsdp_reshard_after_forward: Optional[Union[bool, int]] = None,
        fsdp_mp_param_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_reduce_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_output_dtype: Optional[FSDPMPDType] = None,
        fsdp_cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        bf16_grad_scaler: bool = False,
        log_interval: int = 1000,
        log_gpu_usage: bool = False,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Optional[Dict[str, Any]] = None,
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        swa_update_steps: int = 5000,
        bn_update_steps: int = 5000,
    ) -> None:
        """
        Initializes trainer-wide state, logging, distributed context, and optional
        advanced training utilities (AMP, SWA, schedulers, etc.).

        Args:
            exp_path: Directory where checkpoints/logs are stored.
            num_epochs: Maximum number of epochs to run (unless `max_steps` stops earlier).
            cur_epoch: Epoch to resume from when continuing training.
            max_steps: Optional global-step limit overriding `num_epochs`.
            cur_step: Global step to resume from.
            grad_acc_steps: Number of minibatches to accumulate before each optimizer step.
            eff_batch_size: Optional effective batch size for logging/reference.
            val_steps: Run validation every N optimizer steps (mutually exclusive with `val_hours`).
            val_hours: Run validation every N hours (wall clock).
            save_steps: Save checkpoints every N optimizer steps (mutually exclusive with `save_hours`).
            save_hours: Save checkpoints every N hours.
            device: Target device (CUDA device index or torch.device) for model/data.
            loggers: Optional logger collection; defaults to console/TensorBoard/W&B selection.
            ddp: Whether to enable DistributedDataParallel of any flavor.
            ddp_type: Specific DDP implementation (standard DDP, torch FSDP sharded/full).
            fsdp_cpu_offload: Enable FSDP parameter CPU offload.
            fsdp_reshard_after_forward: Control post-forward resharding (None follows FSDP2 default: shard children, keep root unsharded; bool or int per FSDP2 API).
            fsdp_mp_param_dtype: Optional FSDP param dtype override for mixed precision.
            fsdp_mp_reduce_dtype: Optional FSDP reduce dtype override for mixed precision.
            fsdp_mp_output_dtype: Optional FSDP output dtype override for mixed precision.
            use_amp: Toggle automatic mixed precision for training/eval loops.
            amp_dtype: Precision to use when AMP is enabled (fp16 or bf16).
            bf16_grad_scaler: Use a GradScaler variant safe for bf16.
            log_interval: Steps between logger progress updates.
            log_gpu_usage: Whether to collect/report GPU utilization stats.
            use_tensorboard: Enable TensorBoard logging backend.
            use_wandb: Enable Weights & Biases logging backend.
            wandb: Extra configuration dict for W&B when enabled.
            grad_clip: Max gradient norm (<=0 disables clipping).
            grad_clip_norm: Norm type used when clipping gradients.
            swa_start: Global step at which to begin SWA (0 disables).
            swa_lr: Learning rate to use during SWA averaging.
            swa_anneal_steps: Steps used to anneal LR before SWA updates.
            swa_update_steps: Steps between SWA weight averaging operations.
            bn_update_steps: Max number of batches for BatchNorm stats refresh post-SWA.
        """
        self.exp_path = Path(exp_path)
        self.num_epochs = num_epochs
        self.cur_epoch = cur_epoch
        self.max_steps = max_steps
        self.cur_step = cur_step
        self.cur_batch = 0
        self.grad_acc_steps = grad_acc_steps
        self.eff_batch_size = eff_batch_size
        self.save_steps = save_steps
        self.val_steps = val_steps
        self.save_hours = save_hours
        self.val_hours = val_hours

        self.device = device

        if wandb is None:
            wandb = {}

        if loggers is None:
            resolved_loggers: LoggerList = self._default_loggers(
                log_interval,
                use_tensorboard,
                use_wandb,
                wandb,
                log_gpu_usage=log_gpu_usage,
            )
        elif isinstance(loggers, list):
            resolved_loggers = LoggerList(loggers)
        else:
            resolved_loggers = loggers

        self.loggers: LoggerList = resolved_loggers

        self.ddp = ddp
        self.ddp_type = ddp_type
        if ddp and ddp_type == DDPType.FSDP and not FSDP_AVAILABLE:
            raise RuntimeError(
                "FSDP2 requires torch>=2.6; current torch version does not provide it."
            )
        self.fsdp_cpu_offload = fsdp_cpu_offload
        self.fsdp_reshard_after_forward = fsdp_reshard_after_forward
        self.fsdp_mp_param_dtype = FSDPMPDType.to_dtype(fsdp_mp_param_dtype)
        self.fsdp_mp_reduce_dtype = FSDPMPDType.to_dtype(fsdp_mp_reduce_dtype)
        self.fsdp_mp_output_dtype = FSDPMPDType.to_dtype(fsdp_mp_output_dtype)

        self.use_amp = use_amp
        self.amp_dtype = AMPDType.to_dtype(amp_dtype)
        self.bf16_grad_scaler = bf16_grad_scaler

        self.grad_clip = grad_clip
        self.grad_clip_norm = grad_clip_norm

        self.swa_start = swa_start
        self.do_swa = swa_start > 0
        self.swa_lr = swa_lr
        self.swa_anneal_steps = swa_anneal_steps
        self.swa_update_steps = swa_update_steps
        self.bn_update_steps = bn_update_steps
        self.in_swa = False

        self.rank: int = 0
        self.world_size: int = 1

        self.ckpt_search_name: str = "model"

        self.train_data: Optional[DataLoader[Any]] = None
        self.val_data: Optional[DataLoader[Any]] = None

        if ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.grad_tracker: Optional[GradNormTracker] = None

    def _prepare_model_for_training(
        self,
        model: TorchModel,
        optim: Union[torch.optim.Optimizer, Dict[str, Any]],
        lrsched: Union[LRS, Dict[str, Any], None],
        wdsched: Union[WDS, Dict[str, Any], None],
        device: Optional[Union[torch.device, int]] = None,
        ddp: bool = False,
        ddp_type: DDPType = DDPType.DDP,
        fsdp_cpu_offload: bool = False,
        fsdp_reshard_after_forward: Optional[Union[bool, int]] = None,
        fsdp_mp_param_dtype: Optional[torch.dtype] = None,
        fsdp_mp_reduce_dtype: Optional[torch.dtype] = None,
        fsdp_mp_output_dtype: Optional[torch.dtype] = None,
        do_swa: bool = False,
        swa_lr: float = 1e-3,
        swa_anneal_steps: int = 50000,
    ) -> Tuple[
        nn.Module,
        torch.optim.Optimizer,
        Optional[LRS],
        Optional[WDS],
        Optional[AveragedModel],
        Optional[SWALR],
    ]:
        """
        Prepares the model, optimizer, and schedulers for training.

        Handles device placement, mixed-precision setup, DDP wrapping, and optionally
        initializes a Stochastic Weight Averaging (SWA) model.

        Args:
            model (TorchModel): The model to train.
            optim (torch.optim.Optimizer): Optimizer or dict of optimizer config.
            lrsched (LRS | Dict | None): Learning rate scheduler or config.
            wdsched (WDS | Dict | None): Weight decay scheduler or config.
            device (torch.device): Device to place the model on.
            ddp (bool): Enable DistributedDataParallel wrapping.
            ddp_type (DDPType): Distributed strategy (DDP or torch FSDP sharded/full).
            fsdp_cpu_offload (bool): If True, uses CPU offload in FSDP.
            fsdp_reshard_after_forward (bool|int|None): Reshard policy to pass into FSDP2.
            fsdp_mp_param_dtype (torch.dtype|None): Mixed-precision param dtype for FSDP2.
            fsdp_mp_reduce_dtype (torch.dtype|None): Mixed-precision reduce dtype for FSDP2.
            fsdp_mp_output_dtype (torch.dtype|None): Mixed-precision output dtype for FSDP2.
            do_swa (bool): Whether to initialize SWA model/scheduler.
            swa_lr (float): Learning rate for SWA.
            swa_anneal_steps (int): Annealing steps for SWA LR scheduler.

        Returns:
            Tuple:
                model (nn.Module),
                optimizer (Optimizer),
                lr_scheduler (Optional[LRS]),
                wd_scheduler (Optional[WDS]),
                swa_model (Optional[AveragedModel]),
                swa_scheduler (Optional[SWALR])
        """
        if device is not None:
            model.to(device)

        if ddp:
            if ddp_type == DDPType.DDP:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with distributed-data-parallel"
                    )
                optimizer = self._make_optimizer(optim, model)
                model = TorchDDP(
                    model,
                    device_ids=[device],
                    output_device=device,
                )
            elif ddp_type == DDPType.FSDP:
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with torch FSDP (sharded/fully-sharded)"
                    )
                fsdp_kwargs = self._build_fsdp_kwargs(
                    reshard_after_forward=fsdp_reshard_after_forward,
                    mp_param_dtype=fsdp_mp_param_dtype,
                    mp_reduce_dtype=fsdp_mp_reduce_dtype,
                    mp_output_dtype=fsdp_mp_output_dtype,
                    cpu_offload=fsdp_cpu_offload,
                )
                if self.rank == 0:
                    logging.info(
                        "fsdp settings: reshard_after_fwd=%s, mixed_precision=%s, fsdp_cpu_offload=%s",
                        fsdp_reshard_after_forward,
                        "enabled" if "mp_policy" in fsdp_kwargs else "disabled",
                        fsdp_cpu_offload,
                    )
                model = self._wrap_with_fully_shard(model, fsdp_kwargs)
                optimizer = self._make_optimizer(optim, model)
            else:
                raise ValueError(f"Unsupported ddp_type: {ddp_type}")

        else:
            optimizer = self._make_optimizer(optim, model)

        # make the learning rate scheduler
        lr_scheduler = self._make_lr_sched(lrsched, optimizer)

        # make weight decay scheduler if needed
        wd_scheduler = self._make_wd_sched(wdsched, optimizer)

        swa_model = None
        swa_scheduler = None
        if do_swa:
            if self.rank == 0:
                logging.info("init SWA model")
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(
                optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_steps
            )

        return (
            model,
            optimizer,
            lr_scheduler,
            wd_scheduler,
            swa_model,
            swa_scheduler,
        )

    def _build_fsdp_kwargs(
        self,
        reshard_after_forward: Optional[Union[bool, int]],
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        mp_output_dtype: Optional[torch.dtype],
        cpu_offload: bool,
    ) -> Dict[str, Any]:
        """
        Builds the kwargs dictionary for torch.distributed.fsdp FullyShardedDataParallel (FSDP2).

        Args:
            reshard_after_forward: Reshard policy after forward (None uses FSDP2 defaults).
            mp_param_dtype: Optional mixed-precision param dtype (None disables mp policy).
            mp_reduce_dtype: Optional mixed-precision reduce dtype.
            mp_output_dtype: Optional mixed-precision output dtype.
            cpu_offload: Whether to offload parameters to CPU between compute steps.

        Returns:
            Dict[str, Any]: Filtered kwargs to pass into `fully_shard`.
        """
        fsdp_kwargs: Dict[str, Any] = {
            "reshard_after_forward": reshard_after_forward,
        }

        if cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

        if mp_param_dtype is not None:
            mixed_precision = MixedPrecisionPolicy(
                param_dtype=mp_param_dtype,
                reduce_dtype=mp_reduce_dtype,
                output_dtype=mp_output_dtype,
            )
            fsdp_kwargs["mp_policy"] = mixed_precision

        return {k: v for k, v in fsdp_kwargs.items() if v is not None}

    @staticmethod
    def _is_fsdp_module(module: nn.Module) -> bool:
        if FSDP is None:
            return False
        return isinstance(module, FSDP)

    def _wrap_with_fully_shard(
        self, model: nn.Module, fsdp_kwargs: Dict[str, Any]
    ) -> nn.Module:
        """
        Recursively applies fully_shard to all submodules and then to the root,
        matching the FSDP2 tutorial guidance.
        """

        def _shard_recursive(module: nn.Module):
            for child in module.children():
                if not self._is_fsdp_module(child):
                    _shard_recursive(child)
                    fully_shard(child, **fsdp_kwargs)

        _shard_recursive(model)
        if not self._is_fsdp_module(model):
            model = fully_shard(model, **fsdp_kwargs)
        return model

    def get_grad_scaler(
        self,
        use_amp: Optional[bool] = None,
        ddp: Optional[bool] = None,
        ddp_type: Optional[DDPType] = None,
        amp_dtype: Optional[torch.dtype] = None,
        bf16_grad_scaler: Optional[bool] = None,
        fsdp_mp_param_dtype: Optional[torch.dtype] = None,
    ) -> Union[amp.GradScaler, ShardedGradScaler]:
        """
        Initializes the appropriate gradient scaler for AMP (automatic mixed precision).

        Uses the torch FSDP ShardedGradScaler for FSDP sharded/full modes, and native GradScaler otherwise.

        Args:
            use_amp (bool): Whether AMP is enabled.
            ddp (bool): Whether DDP is being used.
            ddp_type (DDPType): DDP backend type.
            amp_dtype (torch.dtype): Data type for AMP (float16 or bfloat16).
            bf16_grad_scaler (bool): If True, enables grad scaler for bfloat16 (default is False).

        Returns:
            GradScaler: AMP gradient scaler (native or sharded).
        """
        use_amp = self.use_amp if use_amp is None else use_amp
        ddp = self.ddp if ddp is None else ddp
        ddp_type = self.ddp_type if ddp_type is None else ddp_type
        amp_dtype = self.amp_dtype if amp_dtype is None else amp_dtype
        bf16_grad_scaler = (
            self.bf16_grad_scaler if bf16_grad_scaler is None else bf16_grad_scaler
        )
        fsdp_mp_param_dtype = (
            self.fsdp_mp_param_dtype
            if fsdp_mp_param_dtype is None
            else fsdp_mp_param_dtype
        )

        if ddp and ddp_type == DDPType.FSDP:
            use_grad_scaler = fsdp_mp_param_dtype == torch.float16 or bf16_grad_scaler
            if self.rank == 0:
                if use_grad_scaler:
                    logging.info(
                        "using mixed precision training with FSDP sharded-grad-scaler"
                    )
                else:
                    logging.info("not using grad scaler with FSDP")
            return ShardedGradScaler(enabled=use_grad_scaler)

        use_grad_scaler = use_amp and (amp_dtype == torch.float16 or bf16_grad_scaler)
        if self.rank == 0:
            if use_grad_scaler:
                logging.info(
                    "using automatic mixed precision training with grad-scaler"
                )
            else:
                logging.info("not using grad scaler")
        return amp.GradScaler(enabled=use_grad_scaler)

    def set_data_epoch(
        self, data_loader: DataLoader[Any], cur_epoch: int, cur_batch: int = 0
    ) -> None:
        """
        Sets the epoch index for the data loader and its batch sampler.

        This is used to ensure shuffling is seeded consistently across processes in DDP.

        Args:
            data_loader (DataLoader): The data loader to modify.
            cur_epoch (int): The current epoch index.
            cur_batch (int): Current batch index (for samplers that support batch-wise resumption).
        """
        try:
            data_loader.dataset.set_epoch(cur_epoch)
        except AttributeError:
            logging.warning("dataset doesn't have set_epoch member function")

        try:
            data_loader.batch_sampler.set_epoch(cur_epoch, cur_batch)
        except AttributeError:
            logging.warning("sampler doesn't have set_epoch member function")

    def on_train_begin(self) -> None:
        """
        Called at the beginning of training.

        Creates the experiment output directory, configures gradient accumulation,
        prepares loggers, and initializes gradient tracking.
        Also checks if SWA should be activated from the start.
        """
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self._compute_grad_acc_steps(self.train_data)
        if self.do_swa and self.cur_step >= self.swa_start:
            self.in_swa = True

        self.loggers.on_train_begin(
            epochs=self.num_epochs, epoch=self.cur_epoch, step=self.cur_step
        )
        self.grad_tracker = GradNormTracker()

    def on_epoch_begin(self) -> None:
        """Callback executed at the beginning of an epoch to trigger logger updates."""
        self.loggers.on_epoch_begin(self.cur_epoch, batches=len(self.train_data))

    def on_epoch_end(self, logs: Dict[str, Any]) -> None:
        """
        Callback executed at the end of an epoch to finalize logging.

        Args:
            logs (Dict[str, Any]): Dictionary of metrics and states collected during the epoch.
        """
        self.loggers.on_epoch_end(logs)

    def on_swa_epoch_begin(self) -> None:
        """Callback executed at the beginning of a SWA (Stochastic Weight Averaging) epoch."""
        self.loggers.on_epoch_begin(self.cur_epoch, batches=len(self.train_data))

    def on_swa_epoch_end(self, logs: Dict[str, Any]) -> None:
        """
        Callback executed at the end of a SWA epoch.

        Args:
            logs (Dict[str, Any]): Dictionary of metrics and states collected during the SWA epoch.
        """
        self.loggers.on_epoch_end(logs)

    def on_training_loop_begin(self) -> None:
        """
        Called at the beginning of the training loop.
        Must be overridden by child class to set models to training mode and configure any loop-specific settings.
        """
        raise NotImplementedError()

    def on_training_loop_resume(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called after validation to resume the training loop.

        Args:
            logs (Optional[Dict[str, Any]]): Optional logs from previous validation.
        """
        self.on_training_loop_begin()

    def on_val_loop_begin(self) -> None:
        """
        Called at the beginning of the validation loop.
        Must be overridden by child class to set models to evaluation mode and freeze behaviors.
        """
        raise NotImplementedError()

    def on_bn_update_loop_begin(self) -> None:
        """Called at the beginning of batch normalization update loop, defaults to using training behavior."""

        self.on_training_loop_begin()

    def preprocess_data(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Preprocess the training/validation batch before model input.
        Must be implemented by subclass.

        Args:
            batch_data (Dict[str, Any]): Raw batch from the DataLoader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch.
        """
        raise NotImplementedError()

    def preprocess_train_data(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Preprocess the training batch before model input.
        Default implementation delegates to :meth:`preprocess_data`.

        Args:
            batch_data (Dict[str, Any]): Raw batch from the DataLoader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch.
        """
        return self.preprocess_data(batch_data)

    def preprocess_val_data(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Preprocess the validation batch before model input.
        Default implementation delegates to :meth:`preprocess_data`.

        Args:
            batch_data (Dict[str, Any]): Raw batch from the DataLoader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch.
        """
        return self.preprocess_data(batch_data)

    def compute_forward(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass for training data.
        Must be implemented by subclass.

        Args:
            batch_data (Dict[str, Any]): Preprocessed training batch.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and model outputs.
        """
        raise NotImplementedError()

    def compute_train_forward(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass for training data.
        Default implementation defers to ``compute_forward`` so subclasses only
        need to override one method.

        Args:
            batch_data (Dict[str, Any]): Preprocessed training batch.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and model outputs.
        """
        return self.compute_forward(batch_data)

    def compute_val_forward(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass for validation data.
        Default implementation reuses ``compute_forward`` (same as training).

        Args:
            batch_data (Dict[str, Any]): Preprocessed validation batch.

        Returns:
            Tuple[torch.Tensor, Any]: Loss tensor and model outputs.
        """
        return self.compute_forward(batch_data)

    def compute_backward(self, loss: torch.Tensor) -> None:
        """
        Computes the backward pass for a given loss.
        Must be implemented by subclass.

        Args:
            loss (torch.Tensor): Scalar loss tensor to backpropagate.
        """
        raise NotImplementedError()

    def compute_metrics(
        self, batch_output: Any, batch_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Computes metrics for a training batch.

        Args:
            batch_output (Any): Model output.
            batch_data (Dict[str, Any]): Original batch.

        Returns:
            Dict[str, float]: Dictionary of training metrics.
        """
        metrics = ODict()
        return metrics

    def compute_train_metrics(
        self, batch_output: Any, batch_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Computes metrics for a training batch.

        Args:
            batch_output (Any): Model output.
            batch_data (Dict[str, Any]): Original batch.

        Returns:
            Dict[str, float]: Dictionary of training metrics.
        """
        return self.compute_metrics(batch_output, batch_data)

    def compute_val_metrics(
        self, batch_output: Any, batch_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Computes metrics for a validation batch.
        Defaults to reusing compute_train_metrics.

        Args:
            batch_output (Any): Model output.
            batch_data (Dict[str, Any]): Original batch.

        Returns:
            Dict[str, float]: Dictionary of validation metrics.
        """
        return self.compute_train_metrics(batch_output, batch_data)

    def compute_bn_update_metrics(
        self, batch_output: Any, batch_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Computes metrics during batch normalization update step.
        Defaults to training metrics.

        Args:
            batch_output (Any): Model output.
            batch_data (Dict[str, Any]): Original batch.

        Returns:
            Dict[str, float]: Dictionary of update metrics.
        """
        return self.compute_train_metrics(batch_output, batch_data)

    def update_swa_model(self) -> None:
        """
        Updates the SWA (Stochastic Weight Averaging) model using current model parameters.
        Must be implemented in subclass.
        """
        raise NotImplementedError()

    # def checkpoint(
    #     self,
    #     model: TorchModel,
    #     optimizer: Optional[torch.optim.Optimizer] = None,
    #     lr_scheduler: Optional[LRS] = None,
    #     wd_scheduler: Optional[WDS] = None,
    #     swa_model: Optional[TorchModel] = None,
    #     swa_scheduler: Optional[SWALR] = None,
    #     logs: Optional[Dict[str, Any]] = None,
    # ):
    #     """Creates a checkpoint of the training, to save and posterior recovery

    #     Args:
    #       logs: logs containing the current value of the metrics.
    #     """
    #     model.train()
    #     checkpoint = {
    #         "epoch": self.cur_epoch,
    #         "batch": self.cur_batch,
    #         "global_step": self.global_step,
    #         "rng_state": torch.get_rng_state(),
    #         "model_cfg": model.get_config(),
    #         "model_state_dict": model.state_dict(),
    #     }
    #     if optimizer is not None:
    #         checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    #     if lr_scheduler is not None:
    #         checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    #     if wd_scheduler is not None:
    #         checkpoint["wd_scheduler_state_dict"] = wd_scheduler.state_dict()

    #     if logs is not None:
    #         checkpoint["logs"] = logs

    #     if self.in_swa and swa_model is not None:
    #         checkpoint["swa_model_state_dict"] = swa_model.state_dict()
    #         checkpoint["swa_scheduler_state_dict"] = swa_scheduler.state_dict()

    #     return checkpoint

    def save_checkpoint(self, logs: Dict[str, Any]) -> None:
        """
        Saves the current model and training state.

        Args:
            logs (Dict[str, Any]): Optional logs to store with the checkpoint.
        """
        raise NotImplementedError()

    def zero_grad_optimizers(self) -> None:
        """
        Clears gradients in all optimizers used in training.
        Should be implemented by subclass to manage all used optimizers.
        """
        raise NotImplementedError

    def get_lrs(self) -> Dict[str, float]:
        """
        Gets learning rates for all optimizer parameter groups.

        Returns:
            Dict[str, float]: Mapping of group names to learning rates.
        """
        raise NotImplementedError()

    def get_wds(self) -> Dict[str, float]:
        """
        Gets weight decays for all optimizer parameter groups.

        Returns:
            Dict[str, float]: Mapping of group names to weight decays.
        """
        raise NotImplementedError()

    def get_grad_scales(self) -> Dict[str, float]:
        """
        Gets gradient scales for all optimizer parameter groups.

        Returns:
            Dict[str, float]: Mapping of group names to gradient scales.
        """

        try:
            return self._get_grad_scale(self.grad_scaler)
        except AttributeError:
            return {}

    def models_have_bn(self) -> bool:
        """
        Returns True if model(s) contain BatchNorm layers.

        Returns:
            bool: Whether BatchNorm exists and should be updated during SWA.
        """
        raise NotImplementedError()

    def make_train_logs(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a dictionary of training logs to be consumed by loggers.

        This method adds a "train_" prefix to all user-supplied log keys, then
        appends learning rate, weight decay values, gradient statistics, and
        gradient scaling factor (if AMP is used).

        Args:
            logs (Dict[str, Any]): Dictionary of base training metrics (e.g., loss, accuracy).

        Returns:
            Dict[str, Any]: Augmented training logs including metrics, learning rates,
                            weight decays, gradient EMA, and grad scaler scale (if applicable).
        """
        train_logs = ODict(("train_" + k, v) for k, v in logs.items())
        train_logs.update(self.get_lrs())
        train_logs.update(self.get_wds())
        train_logs.update(self.grad_tracker.grad_ema)
        train_logs.update(self.get_grad_scales())
        return train_logs

    def make_val_logs(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a dictionary of validation logs to be consumed by loggers.

        This method adds a "val_" prefix to all user-supplied log keys.

        Args:
            logs (Dict[str, Any]): Dictionary of base validation metrics (e.g., loss, accuracy).

        Returns:
            Dict[str, Any]: Prefixed validation logs (keys prefixed with 'val_').
        """
        val_logs = ODict(("val_" + k, v) for k, v in logs.items())
        return val_logs

    def send_data_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfers tensor data in a batch to the configured training device (e.g., GPU).

        This function only moves values that are torch.Tensors and skips non-tensor types.
        It also ensures tensors are pinned before transfer for non-blocking copy.

        Args:
            batch_data (Dict[str, Any]): Dictionary of minibatch data from DataLoader.

        Returns:
            Dict[str, Any]: Same dictionary, but all tensor values moved to the target device.
        """
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                assert v.is_pinned()

        return {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) and v.device != self.device
                else v
            )
            for k, v in batch_data.items()
        }

    def update_models(self) -> Dict[str, float]:
        """
        Applies optimizer updates and gradient clipping.
        Must be implemented in the subclass to handle model-specific updates.
        """
        raise NotImplementedError()

    def save_swa_model(self) -> None:
        """
        Saves the current SWA model checkpoint to disk.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def fit(
        self, train_data: DataLoader[Any], val_data: Optional[DataLoader[Any]] = None
    ) -> None:
        """
        Runs the full training loop over all epochs, including optional validation
        and SWA (Stochastic Weight Averaging) updates.

        This is the main entry point for training a model, handling:
          - epoch-level iteration
          - periodic validation
          - checkpoint saving
          - optional SWA and BN updates

        Args:
            train_data: A PyTorch DataLoader providing training batches.
            val_data: Optional PyTorch DataLoader for validation. If provided,
                      validation will be run at the end of each epoch or at intervals.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.last_save_time = time.time()
        self.last_val_time = time.time()
        self.last_save_step = self.cur_step
        self.last_val_step = self.cur_step
        self.on_train_begin()
        val_logs = {}
        for epoch in range(self.cur_epoch, self.num_epochs):
            self.set_data_epoch(train_data, self.cur_epoch, self.cur_batch)
            self.on_epoch_begin()
            logs = self.training_loop()
            self.cur_batch = 0
            if val_data is not None:
                self.set_data_epoch(val_data, self.cur_epoch)
                val_logs = self.validation_loop()
                logs.update(val_logs)

            self.cur_epoch += 1
            self.on_epoch_end(logs)
            self.save_checkpoint(logs)
            if self.finish_now():
                break

        if self.in_swa:
            self.on_swa_epoch_begin()
            if self.models_have_bn():
                logs = self.bn_update_loop()
            else:
                logs = ODict()

            if val_data is not None:
                val_logs = self.validation_loop()
                logs.update(val_logs)

            self.cur_epoch += 1
            self.on_swa_epoch_end(logs)
            self.save_swa_model(logs)

    def training_loop(self) -> Dict[str, Any]:
        """
        Runs one epoch of training, managing gradient accumulation, logging,
        checkpointing, and periodic validation.

        Returns:
            Dict[str, Any]: A dictionary of training metrics logged throughout the epoch.
        """
        metric_acc = MetricAcc(device=self.device)
        self.on_training_loop_begin()
        self.zero_grad_optimizers()
        for batch_idx, batch_data in enumerate(self.train_data):
            self.loggers.on_batch_begin(batch_idx)

            batch_size, batch_metrics = self.training_step(batch_idx, batch_data)
            self.model_update_step(batch_idx)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs.update(self.get_lrs())
            logs.update(self.get_wds())
            logs.update(self.grad_tracker.grad_ema)
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

            if self.finish_now():
                break

            if self.save_now():
                self.save_checkpoint()

            if self.validate_now():
                logs = self.make_train_logs(metric_acc.metrics)
                val_logs = self.validation_loop()
                logs.update(val_logs)
                self.loggers.on_val_end(logs=logs)
                self.on_training_loop_resume()
                metric_acc.reset()

        logs = self.make_train_logs(metric_acc.metrics)
        return logs

    # def train_forward_backward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Executes a single forward and backward pass for the given training batch.

    #     Handles AMP autocasting and loss scaling (if enabled), computes gradients,
    #     and collects training metrics.

    #     Args:
    #         batch_data (Dict[str, Any]): Dictionary of training input and targets.

    #     Returns:
    #         Dict[str, Any]: Dictionary of computed metrics including scaled loss.
    #     """
    #     with amp.autocast(
    #         enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
    #     ):
    #         loss, output = self.compute_train_forward(batch_data)

    #     loss = loss / self.grad_acc_steps

    #     self.compute_backward(loss)
    #     batch_metrics = self.compute_train_metrics(output, batch_data)
    #     batch_metrics["loss"] = loss.item() * self.grad_acc_steps
    #     return batch_metrics

    def training_step(
        self, batch_idx: int, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Performs a single training step including:
            - data preprocessing
            - sending data to device
            - forward/backward computation
            - deferred optimizer update handled by model_update_step()

        Args:
            batch_idx (int): Index of the current batch.
            batch_data (Dict[str, Any]): Batch data from the DataLoader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and computed training metrics.
        """
        batch_size, batch_data = self.preprocess_train_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)
        with amp.autocast(
            enabled=self.use_amp,
            dtype=self.amp_dtype,
            device_type="cuda",
        ):
            loss, batch_output = self.compute_train_forward(batch_data)

        loss = loss.float() / self.grad_acc_steps

        self.compute_backward(loss)
        batch_metrics = ODict()
        batch_metrics = self.compute_train_metrics(batch_output, batch_data)
        batch_metrics["loss"] = loss.item() * self.grad_acc_steps
        batch_metrics.move_to_end("loss", last=False)

        # if (batch_idx + 1) % self.grad_acc_steps == 0:
        #     self.cur_batch = batch_idx
        #     self.cur_step += 1
        #     grad_norms = self.update_models()
        #     self.grad_tracker.update(grad_norms)
        #     grad_logs = self.grad_tracker.grad_spikes
        #     self.zero_grad_optimizers()
        #     self.loggers.on_model_update(self.cur_step, log=grad_logs)

        return batch_size, batch_metrics

    def model_update_step(self, batch_idx: int) -> None:
        """
        Applies optimizer/scheduler updates when gradient accumulation is satisfied.

        Args:
            batch_idx: Zero-based index of the current batch inside the epoch.

        Returns:
            None
        """
        if (batch_idx + 1) % self.grad_acc_steps == 0:
            self.cur_batch = batch_idx
            self.cur_step += 1
            grad_norms = self.update_models()
            self.grad_tracker.update(grad_norms)
            grad_logs = self.grad_tracker.grad_spikes
            self.zero_grad_optimizers()
            self.loggers.on_model_update(self.cur_step, log=grad_logs)

    @torch.no_grad()
    def validation_loop(self) -> Dict[str, Any]:
        """
        Runs the validation loop over the entire validation set.

        Returns:
            Dict[str, Any]: Dictionary of averaged validation metrics for the epoch.
        """
        metric_acc = MetricAcc(self.device)
        self.on_val_loop_begin()
        for batch_idx, batch_data in enumerate(self.val_data):
            batch_size, batch_metrics = self.validation_step(batch_idx, batch_data)
            metric_acc.update(batch_metrics, batch_size)

        logs = self.make_val_logs(metric_acc.metrics)
        return logs

    def validation_step(
        self, batch_idx: int, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Performs a single validation step:
            - data preprocessing
            - device transfer
            - forward pass
            - metric computation

        Args:
            batch_idx (int): Index of the current validation batch.
            batch_data (Dict[str, Any]): Batch data from the validation DataLoader.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and computed validation metrics.
        """
        batch_size, batch_data = self.preprocess_val_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)

        with amp.autocast(
            enabled=self.use_amp,
            dtype=self.amp_dtype,
            device_type="cuda",
        ):
            loss, batch_output = self.compute_val_forward(batch_data)

        batch_metrics = self.compute_val_metrics(batch_output, batch_data)
        batch_metrics["loss"] = loss.item()
        batch_metrics.move_to_end("loss", last=False)

        return batch_size, batch_metrics

    @torch.no_grad()
    def bn_update_loop(self) -> Dict[str, Any]:
        """Batch normalization update loop"""
        metric_acc = MetricAcc(self.device)
        self.on_bn_update_loop_begin()
        for batch_idx, batch_data in enumerate(self.val_data):
            batch_size, batch_metrics = self.bn_update_step(batch_data)
            metric_acc.update(batch_metrics, batch_size)
            if batch_idx > self.bn_update_steps:
                break

        logs = self.make_train_logs(metric_acc.metrics)
        return logs

    def bn_update_step(self, batch_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Performs a single update step for batch normalization statistics.
        This uses the same forward logic as training but does not backpropagate.

        Args:
            batch_data (Dict[str, Any]): Batch data for updating BN statistics.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and metrics from this step.
        """
        batch_size, batch_data = self.preprocess_train_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)

        with amp.autocast(
            enabled=self.use_amp,
            dtype=self.amp_dtype,
            device_type="cuda",
        ):
            loss, batch_output = self.compute_train_forward(batch_data)

        batch_metrics = self.compute_train_metrics(batch_output, batch_data)
        batch_metrics["loss"] = loss.item()
        batch_metrics.move_to_end("loss", last=False)

        return batch_size, batch_metrics

    def _clip_grad_norm(
        self,
        model: nn.Module,
        grad_clip: float,
        grad_clip_norm: Union[int, float, str],
    ) -> float:
        """
        Clips the gradients of the model parameters to prevent exploding gradients.

        Args:
            model (nn.Module): The model whose gradients will be clipped.
            grad_clip (float): Maximum gradient norm. If <= 0, clipping is skipped.
            grad_clip_norm (float or str): Type of norm (e.g., 1, 2, 'inf').

        Returns:
            float: The total norm of the parameters (before clipping).
        """
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, norm_type=grad_clip_norm
        )
        return grad_norm.item()

    def _update_model_by_optim(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        grad_clip_norm: Union[int, float, str],
        use_amp: bool,
        grad_scaler: Union[amp.GradScaler, ShardedGradScaler],
    ) -> float:
        """
        Updates the model parameters by stepping the optimizer, with optional
        gradient clipping and AMP unscaling.

        This method:
            - Unscales gradients (if AMP is used)
            - Applies gradient clipping
            - Steps the optimizer
            - Zeros gradients afterward

        Args:
            model (nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer.
            grad_clip (float): Maximum norm for gradient clipping.
            grad_clip_norm (float or str): Norm type for gradient clipping.
            use_amp (bool): Whether AMP (Automatic Mixed Precision) is enabled.
            grad_scaler (torch.amp.GradScaler): AMP gradient scaler.

        Returns:
            float: The norm of the gradients after clipping.
        """
        grad_scaler.unscale_(optimizer)
        grad_clip = 30000 if grad_clip <= 0 else grad_clip
        grad_norm = self._clip_grad_norm(model, grad_clip, grad_clip_norm)
        grad_scaler.step(optimizer)
        optimizer.zero_grad()
        return grad_norm

        # if use_amp:
        #     # is_ok = self._check_for_grad_nans(model, optimizer)
        #     # if not is_ok:
        #     #     return
        #     if grad_clip > 0:
        #         grad_scaler.unscale_(optimizer)
        #         grad_norm = self._clip_grad_norm(
        #             model, optimizer, grad_clip, grad_clip_norm
        #         )

        #     grad_scaler.step(optimizer)
        # else:
        #     if grad_clip > 0:
        #         grad_norm = self._clip_grad_norm(
        #             model, optimizer, grad_clip, grad_clip_norm
        #         )

        #     optimizer.step()

        # optimizer.zero_grad()
        # return grad_norm

    def _make_optimizer(
        self,
        optim: Union[torch.optim.Optimizer, Dict[str, Any]],
        model: TorchModel,
    ) -> torch.optim.Optimizer:
        """
        Creates an optimizer instance for the given model.

        Args:
            optim (Union[torch.optim.Optimizer, dict]): Either an instantiated optimizer or a config dictionary.
            model (nn.Module): The model whose parameters will be optimized.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        if isinstance(optim, torch.optim.Optimizer):
            return optim

        assert isinstance(optim, dict)
        opt_args = OF.filter_args(**optim)
        if self.rank == 0:
            logging.info("optimizer args={}".format(opt_args))

        optimizer = OF.create(model.trainable_param_groups(), **opt_args)
        if self.rank == 0:
            for i, pg in enumerate(optimizer.param_groups):
                pg_keys = list(pg.keys())
                logging.info(f"optimizer param_group {i} keys={pg_keys}")

        return optimizer

    def _make_lr_sched(
        self,
        lr_sched: Union[LRS, Dict[str, Any], None],
        optim: torch.optim.Optimizer,
    ) -> Optional[LRS]:
        """
        Instantiates a learning rate scheduler from a config dictionary.

        Args:
            lr_sched (Union[LRS, dict, None]): The learning rate scheduler object or config.
            optim (torch.optim.Optimizer): Optimizer for which the scheduler is used.

        Returns:
            LRScheduler or None: The learning rate scheduler instance or None.
        """
        if lr_sched is None or isinstance(lr_sched, LRS):
            return lr_sched

        assert isinstance(lr_sched, dict)
        args = LRSF.filter_args(**lr_sched)
        if self.rank == 0:
            logging.info(f"lr scheduler args={args}")
        lr_sched = LRSF.create(optim, **args)
        return lr_sched

    def _make_wd_sched(
        self,
        wd_sched: Union[WDS, Dict[str, Any], None],
        optim: torch.optim.Optimizer,
    ) -> Optional[WDS]:
        """
        Instantiates a weight decay scheduler from a config dictionary.

        Args:
            wd_sched (Union[WDS, dict, None]): The weight decay scheduler object or config.
            optim (torch.optim.Optimizer): Optimizer for which the scheduler is used.

        Returns:
            WDScheduler or None: The weight decay scheduler instance or None.
        """
        if wd_sched is None or isinstance(wd_sched, WDS):
            return wd_sched

        assert isinstance(wd_sched, dict)
        args = WDSF.filter_args(**wd_sched)
        if self.rank == 0:
            logging.info(f"wd scheduler args={args}")
        wd_sched = WDSF.create(optim, **args)
        return wd_sched

    def _default_loggers(
        self,
        log_interval: int,
        use_tensorboard: bool,
        use_wandb: bool,
        wandb: Dict[str, Any],
        log_gpu_usage: bool = False,
    ) -> LoggerList:
        """
        Creates a default list of logger instances.

        Args:
            log_interval (int): Number of steps between log outputs.
            use_tensorboard (bool): Whether to include a TensorBoard logger.
            use_wandb (bool): Whether to include a Weights & Biases logger.
            wandb (dict): W&B configuration dictionary.
            log_gpu_usage (bool): Whether to track GPU memory usage in logs.

        Returns:
            LoggerList: A list of active loggers.
        """
        prog_log = ProgLogger(interval=log_interval)
        csv_log = CSVLogger(self.exp_path / "train.log", append=True)
        loggers = [prog_log, csv_log]
        if use_tensorboard:
            loggers.append(
                TensorBoardLogger(
                    self.exp_path / "tb", interval=log_interval, gpu_usage=log_gpu_usage
                )
            )
        if use_wandb:
            loggers.append(
                WAndBLogger(
                    **wandb,
                    path=self.exp_path / "wandb",
                    interval=log_interval,
                    gpu_usage=log_gpu_usage,
                )
            )
        return LoggerList(loggers)

    def _get_lrs(self, optim: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Extracts the current learning rates from all parameter groups in the optimizer.

        Args:
            optim (torch.optim.Optimizer): The optimizer instance.

        Returns:
            Dict[str, float]: Dictionary with keys as 'lr_0', 'lr_1', ..., or 'lr' if single group.
        """
        lrs = {
            f"lr_{i}": param_group["lr"]
            for i, param_group in enumerate(optim.param_groups)
        }
        if len(lrs) == 1:
            lrs["lr"] = lrs.pop("lr_0")

        return lrs

    def _get_wds(
        self, optim: torch.optim.Optimizer, wd_scheduler: Optional[WDS] = None
    ) -> Dict[str, float]:
        """
        Extracts the current weight decay values from all parameter groups in the optimizer.

        Args:
            optim (torch.optim.Optimizer): The optimizer instance.
            wd_scheduler (Optional[WDS]): Weight decay scheduler, must be non-None to report values.

        Returns:
            Dict[str, float]: Dictionary with keys as 'wd_0', 'wd_1', ..., or 'wd' if single group.
        """
        if wd_scheduler is None:
            return {}

        wds = {
            f"wd_{i}": param_group["weight_decay"]
            for i, param_group in enumerate(optim.param_groups)
        }
        if len(wds) == 1:
            wds["wd"] = wds.pop("wd_0")

        return wds

    def _get_grad_scale(
        self, grad_scaler: Union[amp.GradScaler, ShardedGradScaler]
    ) -> Dict[str, float]:
        """
        Extracts the current gradient scaling factor from the AMP GradScaler.

        Args:
            grad_scaler (amp.GradScaler): The AMP gradient scaler instance.

        Returns:
            Dict[str, float]: Dictionary with key 'grad_scale' and its current scaling factor.
        """
        if grad_scaler.is_enabled():
            return {"grad_scale": grad_scaler._scale.item()}

        return {}

    def _compute_grad_acc_steps(self, data_loader: DataLoader[Any]) -> None:
        """
        Computes gradient accumulation steps automatically based on effective batch size and dataset loader.

        Sets self.grad_acc_steps to ensure the total batch size matches `eff_batch_size` across GPUs.

        Args:
            data_loader (DataLoader): The training DataLoader.
        """
        if self.eff_batch_size is None:
            return

        if data_loader.batch_sampler is not None:
            try:
                batch_size = data_loader.batch_sampler.avg_batch_size
            except:
                logging.warning(
                    "batch sampler doesn't have avg_batch_size property, "
                    "we cannot estimate grad_acc_steps, using grad_acc_steps=%d",
                    self.grad_acc_steps,
                )
                return

            self.grad_acc_steps = int(
                math.ceil(self.eff_batch_size / batch_size / self.world_size)
            )
            logging.info(
                "Setting grad_acc_steps=%d for "
                "eff_batch_size=%d, avg_batch_size=%d, world_size=%d",
                self.grad_acc_steps,
                self.eff_batch_size,
                batch_size,
                self.world_size,
            )
            return

        logging.warning(
            "We cannot determine the batch_size, "
            "we cannot estimate grad_acc_steps, using grad_acc_steps=%d",
            self.grad_acc_steps,
        )

    def model_checkpoint(
        self,
        model: TorchModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRS] = None,
        wd_scheduler: Optional[WDS] = None,
        swa_model: Optional[AveragedModel] = None,
        swa_scheduler: Optional[SWALR] = None,
        logs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Creates a full model checkpoint dictionary including model state, optimizer,
        schedulers, RNG state, and optionally SWA components and logs.

        Args:
            model (TorchModel): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer instance.
            lr_scheduler (Optional[LRS]): Learning rate scheduler, if used.
            wd_scheduler (Optional[WDS]): Weight decay scheduler, if used.
            swa_model (Optional[TorchModel]): SWA-averaged model, if using SWA.
            swa_scheduler (Optional[SWALR]): SWA learning rate scheduler.
            logs (Optional[Dict[str, Any]]): Additional logs to store in checkpoint.

        Returns:
            Dict[str, Any]: The full checkpoint dictionary.
        """
        model.train()
        if self._is_fsdp_module(model):
            dict_options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
            model_state_dict = get_model_state_dict(
                model,
                options=dict_options,
            )
            optimizer_state_dict = get_optimizer_state_dict(
                model=model, optimizers=optimizer, options=dict_options
            )
        else:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

        checkpoint = {
            "epoch": self.cur_epoch,
            "batch": self.cur_batch,
            "step": self.cur_step,
            "rng_state": torch.get_rng_state(),
            "model_cfg": model.get_config(),
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        if lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        if wd_scheduler is not None:
            checkpoint["wd_scheduler_state_dict"] = wd_scheduler.state_dict()

        if logs is not None:
            checkpoint["logs"] = logs

        if self.in_swa:
            checkpoint["swa_model_state_dict"] = swa_model.state_dict()
            checkpoint["swa_scheduler_state_dict"] = swa_scheduler.state_dict()

        return checkpoint

    def save_now(self) -> bool:
        """
        Determines whether a model checkpoint should be saved based on elapsed time
        or training step count.

        Returns:
            bool: True if saving conditions are met, False otherwise.
        """
        if self.save_hours is not None:
            t = time.time() / 3600
            dt = t - self.last_save_time
            if dt > self.save_hours:
                self.last_save_time = t
                self.last_save_step = self.cur_step
                return True

        if self.save_steps is not None:
            dstep = self.cur_step - self.last_save_step
            if self.cur_step > 0 and dstep >= self.save_steps:
                self.last_save_time = time.time() / 3600
                self.last_save_step = self.cur_step
                return True

        return False

    def validate_now(self) -> bool:
        """
        Determines whether validation should be run based on elapsed time
        or training step count.

        Returns:
            bool: True if validation conditions are met, False otherwise.
        """
        if self.val_hours is not None:
            t = time.time() / 3600
            dt = t - self.last_val_time
            if dt > self.val_hours:
                self.last_val_time = t
                self.last_val_step = self.cur_step
                return True

        if self.val_steps is not None:
            dstep = self.cur_step - self.last_val_step
            if self.cur_step > 0 and dstep >= self.val_steps:
                self.last_val_step = self.cur_step
                self.last_val_time = time.time() / 3600
                return True

        return False

    def finish_now(self) -> bool:
        """
        Determines whether the training process should stop based on max_steps.

        Returns:
            bool: True if current step exceeds max_steps, False otherwise.
        """
        return self.max_steps is not None and self.cur_step > self.max_steps

    def save_model_checkpoint_to_file(
        self,
        model_name: str,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Saves a model checkpoint to disk using epoch and step as part of the filename.

        Args:
            model_name (str): Identifier for the model (used in filename).
            checkpoint (Dict[str, Any]): Checkpoint dictionary to save.
        """
        file_path = "%s/%s_ep%04d_step%010d.pth" % (
            self.exp_path,
            model_name,
            self.cur_epoch,
            self.cur_step,
        )

        logging.info("saving %s to %s", model_name, file_path)
        torch.save(checkpoint, file_path)

    def save_swa_model_checkpoint_to_file(
        self, model_name: str, checkpoint: Dict[str, Any]
    ) -> None:
        """
        Saves an SWA model checkpoint to disk by replacing the model state
        with the SWA model state and updating the filename.

        Args:
            model_name (str): Identifier for the model (used in filename).
            checkpoint (Dict[str, Any]): Checkpoint dictionary to save.
        """
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_%s_ep%04d_step%010d.pth" % (
            self.exp_path,
            model_name,
            self.cur_epoch,
            self.cur_step,
        )

        torch.save(checkpoint, file_path)

    def _load_vars_from_checkpoint(
        self, checkpoint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Loads trainer state variables (epoch, step, RNG) from a checkpoint.

        Args:
            checkpoint (Dict[str, Any]): Loaded checkpoint dictionary.

        Returns:
            Optional[Dict[str, Any]]: Logs from checkpoint if present, else None.
        """
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state)
        if self.rank > 0:
            # this will make sure that each process produces different data
            # when using ddp
            dummy = torch.rand(1000 * self.rank)
            del dummy

        self.cur_epoch = checkpoint["epoch"]
        if "batch" in checkpoint:
            self.cur_batch = checkpoint["batch"]
        else:
            self.cur_batch = 0

        self.cur_step = checkpoint["step"]

        logs = None
        if "logs" in checkpoint:
            logs = checkpoint["logs"]

        return logs

    def _load_model_state_dicts_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        model: TorchModel,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[LRS] = None,
        wd_scheduler: Optional[WDS] = None,
        swa_model: Optional[nn.Module] = None,
        swa_scheduler: Optional[SWALR] = None,
    ) -> None:
        """
        Loads the model, optimizer, and scheduler states from a checkpoint dictionary.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint dictionary.
            model (TorchModel): The model instance to load state into.
            optimizer (torch.optim.Optimizer): The optimizer to load state into.
            lr_scheduler (Optional[LRS]): Learning rate scheduler to load state into.
            wd_scheduler (Optional[WDS]): Weight decay scheduler to load state into.
            swa_model (Optional[TorchModel]): SWA model instance, if applicable.
            swa_scheduler (Optional[SWALR]): SWA scheduler instance, if applicable.
        """
        if not self.ddp:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except (RuntimeError, AttributeError):
                state_dict = {
                    k.replace("module.", "", 1) if k.startswith("module.") else k: v
                    for k, v in checkpoint["model_state_dict"].items()
                }
                model.load_state_dict(state_dict)
        elif self.ddp_type == DDPType.DDP:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except (RuntimeError, AttributeError):
                model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            set_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )

        if optimizer is not None:
            if self.ddp and self.ddp_type == DDPType.FSDP:
                set_optimizer_state_dict(
                    model=model,
                    optimizers=optimizer,
                    optim_state_dict=checkpoint["optimizer_state_dict"],
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                    ),
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if wd_scheduler is not None:
            wd_scheduler.load_state_dict(checkpoint["wd_scheduler_state_dict"])

        if self.do_swa:
            if "swa_model_state_dict" in checkpoint:
                swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
                swa_scheduler.load_state_dict(checkpoint["swa_scheduler_state_dict"])

    def find_last_checkpoint(
        self, model_name: str = "model"
    ) -> Tuple[Optional[str], int, int]:
        """
        Finds the most recent checkpoint file for a given model based on epoch and step.

        Args:
            model_name (str): Name prefix used in checkpoint filenames.

        Returns:
            Tuple[str, int, int]: Path to the checkpoint file, last epoch, and last step.
        """
        file_path = None
        last_epoch = 0
        last_step = 0
        file_pattern = "%s/%s_ep[0-9]*_step[0-9]*.pth" % (self.exp_path, model_name)
        file_paths = sorted(glob.glob(file_pattern))
        if len(file_paths) > 0:
            file_path = file_paths[-1]
            last_epoch = int(re.search(r"ep[0-9]*", file_path).group()[2:])
            last_step = int(re.search(r"step[0-9]*", file_paths[-1]).group()[4:])

        return file_path, last_epoch, last_step

    def load_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Loads the most recent checkpoint if one exists.

        Returns:
            Optional[Dict[str, Any]]: Logs from the last checkpoint if found, otherwise None.
        """
        _, last_epoch, last_step = self.find_last_checkpoint(self.ckpt_search_name)
        if last_epoch > 0 or last_step > 0:
            return self.load_checkpoint(last_epoch, last_step)

        return None

    def load_model_checkpoint_from_file(
        self, model_name: str = "model", epoch: int = 0, step: int = 0
    ) -> Dict[str, Any]:
        """
        Loads a checkpoint file from disk for a specific model at a given epoch and step.

        Args:
            model_name (str): Identifier for the model.
            epoch (int): Epoch number in checkpoint filename.
            step (int): Step number in checkpoint filename.

        Returns:
            Dict[str, Any]: Loaded checkpoint.
        """
        file_path = "%s/%s_ep%04d_step%010d.pth" % (
            self.exp_path,
            model_name,
            epoch,
            step,
        )
        logging.info("loading %s from %s", model_name, file_path)
        return torch.load(
            file_path, mmap=True, map_location=torch.device("cpu"), weights_only=False
        )

    def load_checkpoint(self, epoch: int, step: int) -> Optional[Dict[str, Any]]:
        """
        Placeholder to be implemented by subclasses to load checkpoint state.

        Args:
            epoch (int): Epoch number.
            step (int): Step number.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @staticmethod
    def get_augs_keys(
        batch: Dict[str, Any], base_key: str, skip: Set[str] = set()
    ) -> List[str]:
        """
        Retrieves a list of all augmentation keys (e.g., base_key, base_key_aug_0_0, etc.)
        found in a batch, excluding keys in the skip set.

        Args:
            batch (dict): A batch of input data.
            base_key (str): The key to use as base for finding augmentations.
            skip (set): A set of keys to exclude.

        Returns:
            List[str]: List of keys corresponding to base and its augmentations.
        """
        keys = []
        if base_key in batch and base_key not in skip:
            keys.append(base_key)

        aug_idx_1 = 0
        while True:
            aug_idx_2 = 0
            while True:
                aug_key = f"{base_key}_aug_{aug_idx_1}_{aug_idx_2}"
                if aug_key in batch:
                    if aug_key not in skip:
                        keys.append(aug_key)
                    aug_idx_2 += 1
                else:
                    break

            if aug_idx_2 == 0:
                break

            aug_idx_1 += 1

        return keys

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters a dictionary of keyword arguments to retain only those
        relevant for initializing the TorchTrainerBase class.

        This is useful when passing a larger config dictionary containing
        parameters for multiple components (e.g., model, optimizer, trainer, etc.).

        Returns:
            dict: A dictionary containing only the arguments that match the
                  TorchTrainerBase.__init__ parameters.
        """
        args = filter_func_args(TorchTrainerBase.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds command-line arguments for configuring a TorchTrainerBase instance.

        Args:
            parser (ArgumentParser): The parser to add arguments to.
            prefix (Optional[str]): If provided, arguments are added under a namespaced block.
            skip (set): Argument names to skip when adding.
        """
        if skip is None:
            skip = set()

        normalized_skip = set(skip)
        stripped_skip = {item.lstrip("-") for item in skip}
        underscored_skip = {item.replace("-", "_") for item in stripped_skip}
        normalized_skip.update(stripped_skip)
        normalized_skip.update(underscored_skip)

        def is_skipped(name: str) -> bool:
            bare = name.lstrip("-")
            return (
                name in normalized_skip
                or bare in normalized_skip
                or bare.replace("-", "_") in normalized_skip
            )

        def add_argument(name: str, *args: Any, **kwargs: Any) -> None:
            if not is_skipped(name):
                parser.add_argument(name, *args, **kwargs)

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        add_argument(
            "--exp-path",
            help="Path to the experiment directory for logs and checkpoints.",
        )
        add_argument(
            "--grad-acc-steps",
            type=int,
            default=1,
            help="Number of batches to accumulate gradients before optimizer step.",
        )
        add_argument(
            "--eff-batch-size",
            type=int,
            default=None,
            help="Target effective batch size. Overrides grad-acc-steps based on dataset and world size.",
        )
        add_argument(
            "--num-epochs",
            type=int,
            default=200,
            help="Total number of training epochs.",
        )
        add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="Maximum number of optimization steps. Overrides num-epochs if set.",
        )
        add_argument(
            "--log-interval",
            type=int,
            default=1000,
            help="Number of steps between logging to stdout or loggers.",
        )
        add_argument(
            "--log-gpu-usage",
            action=ActionYesNo,
            default=False,
            help="Enable GPU memory and utilization logging (e.g., in TensorBoard).",
        )
        add_argument(
            "--save-steps",
            default=None,
            type=int,
            help="Interval (in steps) between saving model checkpoints. If None, saves at epoch end only.",
        )
        add_argument(
            "--val-steps",
            default=None,
            type=int,
            help="Interval (in steps) between validation passes. If None, validates only at epoch end.",
        )
        add_argument(
            "--save-hours",
            default=None,
            type=float,
            help="Minimum hours between saving checkpoints based on wall-clock time.",
        )
        add_argument(
            "--val-hours",
            default=None,
            type=float,
            help="Minimum hours between validation runs based on wall-clock time.",
        )
        add_argument(
            "--use-tensorboard",
            action=ActionYesNo,
            default=False,
            help="Enable TensorBoard logging.",
        )
        add_argument(
            "--use-wandb",
            action=ActionYesNo,
            default=False,
            help="Enable Weights & Biases (W&B) experiment tracking.",
        )
        add_argument("--wandb.project", default=None, help="Name of the W&B project.")
        add_argument(
            "--wandb.group", default=None, help="W&B group name for multiple runs."
        )
        add_argument(
            "--wandb.name",
            default=None,
            help="Run name to appear in the W&B dashboard.",
        )
        add_argument(
            "--wandb.mode",
            default="online",
            choices=["online", "offline"],
            help="W&B logging mode: 'online' to sync or 'offline' to log locally.",
        )

        add_argument(
            "--ddp-type",
            default="ddp",
            choices=DDPType.choices(),
            help="Distributed backend: standard DDP or torch FSDP sharded/full.",
        )
        add_argument(
            "--fsdp-cpu-offload",
            action=ActionYesNo,
            default=False,
            help="Whether to offload parameters to CPU during training (used in FSDP).",
        )
        add_argument(
            "--fsdp-reshard-after-forward",
            default=None,
            help=(
                "Control FSDP2 resharding after forward (None=default child=True/root=False, "
                "True/False override, int=reshard to smaller world size)."
            ),
        )
        add_argument(
            "--fsdp-state-dict-type",
            default="full",
            choices=["full", "sharded", "local"],
            help="State dict format to use for saving/loading FSDP models.",
        )
        add_argument(
            "--fsdp-state-dict-cpu-offload",
            action=ActionYesNo,
            default=True,
            help="Offload FSDP state dicts to CPU when saving/loading.",
        )
        add_argument(
            "--fsdp-state-dict-rank0-only",
            action=ActionYesNo,
            default=True,
            help="Gather full FSDP state dict only on rank 0.",
        )
        add_argument(
            "--fsdp-sync-module-states",
            action=ActionYesNo,
            default=False,
            help="Synchronize module states before the first FSDP forward pass.",
        )
        add_argument(
            "--fsdp-mp-param-dtype",
            default=None,
            choices=FSDPMPDType.choices(),
            help="Override FSDP mixed-precision param dtype (default uses amp-dtype when AMP is enabled). Use 'none' to disable.",
        )
        add_argument(
            "--fsdp-mp-reduce-dtype",
            default=None,
            choices=FSDPMPDType.choices(),
            help="Override FSDP mixed-precision reduce dtype (default uses amp-dtype when AMP is enabled). Use 'none' to disable.",
        )
        add_argument(
            "--fsdp-mp-output-dtype",
            default=None,
            choices=FSDPMPDType.choices(),
            help="Override FSDP mixed-precision output dtype (default uses amp-dtype when AMP is enabled). Use 'none' to disable.",
        )
        add_argument(
            "--use-amp",
            action=ActionYesNo,
            default=False,
            help="Enable automatic mixed precision (AMP) training.",
        )
        add_argument(
            "--amp-dtype",
            default=AMPDType.FLOAT16.value,
            choices=AMPDType.choices(),
            help="AMP data type. Choose 'float16' or 'bfloat16'.",
        )
        add_argument(
            "--bf16-grad-scaler",
            action=ActionYesNo,
            default=False,
            help="Enable gradient scaling for bfloat16 (BF16) training.",
        )
        add_argument(
            "--grad-clip",
            type=float,
            default=0,
            help="Maximum norm for gradient clipping. Set to 0 to disable clipping.",
        )
        add_argument(
            "--grad-clip-norm",
            default=2,
            choices=["inf", 1, 2],
            help="Norm type used for gradient clipping (L1, L2, or L∞).",
        )

        add_argument(
            "--swa-start",
            type=int,
            default=0,
            help="Step at which to start Stochastic Weight Averaging (SWA). Disabled if 0.",
        )
        add_argument(
            "--swa-lr",
            type=float,
            default=1e-3,
            help="Learning rate for SWA optimization phase.",
        )
        add_argument(
            "--swa-anneal-steps",
            type=int,
            default=50000,
            help="Number of steps to anneal SWA learning rate.",
        )
        add_argument(
            "--swa-update-steps",
            type=int,
            default=5000,
            help="How frequently (in steps) to update SWA weights.",
        )
        add_argument(
            "--bn-update-steps",
            type=int,
            default=5000,
            help="Steps used to update BatchNorm statistics after SWA finalization.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
