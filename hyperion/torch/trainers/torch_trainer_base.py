"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import glob
import logging
import math
import re
import time
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.optim.oss import OSS
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.swa_utils import SWALR, AveragedModel

from ...utils import PathLike
from ...utils.misc import filter_func_args
from ..loggers import CSVLogger, LoggerList, ProgLogger, TensorBoardLogger, WAndBLogger
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..optim import OptimizerFactory as OF
from ..torch_model import TorchModel
from ..utils import (
    FairFullyShardedDDP,
    FairShardedDDP,
    MetricAcc,
    TorchDDP,
    tensors_subset,
)
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF


class DDPType(str, Enum):
    DDP = "ddp"
    OSS_DDP = "oss_ddp"
    OSS_SHARDED_DDP = "oss_sharded_ddp"
    FULLY_SHARDED_DDP = "fully_sharded_ddp"

    @staticmethod
    def choices():
        return [o.value for o in DDPType]


class AMPDType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @staticmethod
    def choices():
        return [o.value for o in AMPDType]

    @staticmethod
    def to_dtype(dtype):
        return torch.float16 if dtype == AMPDType.FLOAT16 else torch.bfloat16


class TorchTrainerBase:
    """Base Trainer class to train basic neural network models

    Attributes:
        exp_path: experiment output path
        num_epochs: max. number of epochs
        cur_epoch: current epoch
        max_steps: max. number of steps
        cur_step:  current step
        grad_acc_steps: gradient accumulation steps to simulate larger batch size.
        eff_batch_size: effective batch size
        val_steps: steps between validation loops
        val_hours: max. number of hours between validation loops
        save_steps: steps between model saves
        save_hours: max. number of hours between model saves.
        device: gpu device
        loggers: LoggerList object, loggers write training progress to std. output and file.
        ddp: if True use distributed data parallel training
        ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
        cpu_offload: CPU offload of gradients when using fully sharded ddp
        use_amp: uses mixed precision training.
        amp_dtype: "float16" | "bfloat16"
        log_interval: number of optim. steps between log outputs
        use_tensorboard: use tensorboard logger
        use_wandb: use wandb logger
        wandb: wandb dictionary of options
        grad_clip: norm to clip gradients, if 0 there is no clipping
        grad_clip_norm: norm type to clip gradients
        swa_start: step to start doing SWA
        swa_lr: SWA learning rate
        swa_anneal_steps: SWA learning rate annealing steps
        swa_update_steps: steps between SWA model averagings
        bn_update_steps:  max. number of steps for updating BatchNorm after SWA
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
        cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        swa_update_steps: int = 5000,
        bn_update_steps: int = 5000,
    ):
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

        if loggers is None:
            self.loggers = self._default_loggers(
                log_interval, use_tensorboard, use_wandb, wandb
            )
        elif isinstance(loggers, list):
            self.loggers = LoggerList(loggers)
        else:
            self.loggers = loggers

        self.ddp = ddp
        self.ddp_type = ddp_type
        self.cpu_offload = cpu_offload

        self.use_amp = use_amp
        self.amp_dtype = AMPDType.to_dtype(amp_dtype)

        self.grad_clip = grad_clip
        self.grad_clip_norm = grad_clip_norm

        self.swa_start = swa_start
        self.do_swa = swa_start > 0
        self.swa_lr = swa_lr
        self.swa_anneal_steps = swa_anneal_steps
        self.swa_update_steps = swa_update_steps
        self.bn_update_steps = bn_update_steps
        self.in_swa = False

        self.rank = 0
        self.world_size = 1
        self.global_step = 0

        if ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _prepare_model_for_training(
        self,
        model: Union[TorchModel, List[TorchModel], Dict[str, TorchModel]],
        optim,
        lrsched,
        wdsched,
        device,
        use_amp,
        ddp,
        ddp_type,
        cpu_offload,
        do_swa,
        swa_lr,
        swa_anneal_steps,
    ):
        if device is not None:
            model.to(device)

        if ddp:
            if ddp_type == DDPType.DDP or ddp_type == DDPType.OSS_DDP:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with distributed-data-parallel"
                    )
                oss = False if ddp_type == DDPType.DDP else True
                optimizer = self._make_optimizer(optim, model, oss=oss)
                model = TorchDDP(
                    model,
                    device_ids=[device],
                    output_device=device,
                )
            elif ddp_type == DDPType.OSS_SHARDED_DDP:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with fair sharded-distributed-data-parallel"
                    )
                optimizer = self._make_optimizer(optim, model, oss=True)
                model = FairShardedDDP(model, optimizer)
            else:
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with fair fully-sharded-distributed-data-parallel"
                    )
                # syncbathcnorm is not supported here, it raises exception
                model = FairFullyShardedDDP(
                    model,
                    mixed_precision=use_amp,
                    move_params_to_cpu=cpu_offload,
                )
                optimizer = self._make_optimizer(optim, model, oss=False)

        else:
            optimizer = self._make_optimizer(optim, model)

        # make the learning rate scheduler
        lr_scheduler = self._make_lr_sched(lrsched, optimizer)

        # make weight decay scheduler if needed
        wd_scheduler = self._make_wd_sched(wdsched, optimizer)

        grad_scaler = None
        if use_amp:
            if ddp and ddp_type != DDPType.DDP:
                if self.rank == 0:
                    logging.info(
                        "using automatic mixed precision training with sharded-grad-scaler"
                    )
                grad_scaler = ShardedGradScaler()
            else:
                if self.rank == 0:
                    logging.info(
                        "using automatic mixed precision training with grad-scaler"
                    )
                grad_scaler = amp.GradScaler()

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
            grad_scaler,
            swa_model,
            swa_scheduler,
        )

    def set_data_epoch(self, data_loader, cur_epoch: int, cur_batch: int = 0):
        try:
            data_loader.dataset.set_epoch(cur_epoch)
        except AttributeError:
            logging.warning("dataset doesn't have set_epoch member function")

        try:
            data_loader.batch_sampler.set_epoch(cur_epoch, cur_batch)
        except AttributeError:
            logging.warning("sampler doesn't have set_epoch member function")

    def on_train_begin(self):
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self._compute_grad_acc_steps(self.train_data)
        if self.do_swa and self.cur_step >= self.swa_start:
            self.in_swa = True

        self.loggers.on_train_begin(epochs=self.num_epochs)

    def on_epoch_begin(self):
        self.loggers.on_epoch_begin(self.cur_epoch, batches=len(self.train_data))

    def on_epoch_end(self, logs):
        self.loggers.on_epoch_end(logs)

    def on_swa_epoch_begin(self):
        self.loggers.on_epoch_begin(self.cur_epoch, batches=len(self.train_data))

    def on_swa_epoch_end(self, logs):
        self.loggers.on_epoch_end(logs)

    def on_train_loop_begin(self):
        raise NotImplementedError()

    def on_val_loop_begin(self):
        raise NotImplementedError()

    def on_bn_update_loop_begin(self):
        self.on_train_loop_begin()

    def preprocess_train_data(self, batch_data):
        raise NotImplementedError()

    def preprocess_val_data(self, batch_data):
        return self.preprocess_val_data(batch_data)

    def compute_train_forward(self, batch_data):
        raise NotImplementedError()

    def compute_val_forward(self, batch_data):
        return self.compute_train_forward(self, batch_data)

    def compute_backward(self, loss):
        raise NotImplementedError()

    def compute_train_metrics(self, batch_output, batch_data):
        metrics = ODict()
        return metrics

    def compute_val_metrics(self, batch_output, batch_data):
        return self.compute_train_metrics(batch_output, batch_data)

    def update_swa_model(self):
        pass
        if (
            self.do_swa
            and self.cur_step >= self.swa_start
            and self.cur_step % self.swa_steps == 0
        ):
            self.in_swa = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

    def save_checkpoint(self, logs):
        raise NotImplementedError()

    def zero_grad_optimizers(self):
        raise NotImplementedError

    def get_lrs(self):
        raise NotImplementedError()

    def get_wds(self):
        raise NotImplementedError()

    def models_have_bn(self):
        raise NotImplementedError()

    def send_data_to_device(self, batch_data):
        return {k: v.to(self.device) for k, v in batch_data.items()}

    def update_models(self):
        raise NotImplementedError()

    def save_swa_model(self):
        raise NotImplementedError()

    def fit(self, train_data, val_data=None):
        """Training function, it performs the training and validation epochs

        Args:
          train_data: PyTorch data loader for the training loop
          val_data: PyTorch data loader for the validation loop
        """
        self.train_data = train_data
        self.val_data = val_data
        self.last_save_time = time.time()
        self.last_val_time = time.time()
        self.on_train_begin()
        val_logs = {}
        for epoch in range(self.cur_epoch, self.epochs):
            self.set_data_epoch(train_data, self.epoch, self.cur_batch)
            self.on_epoch_begin()
            logs = self.train_loop()
            self.cur_batch = 0
            if val_data is not None:
                self.set_data_epoch(val_data)
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

    def train_loop(self):
        """Training epoch loop"""
        metric_acc = MetricAcc(device=self.device)
        self.on_train_loop_begin()
        for batch_idx, batch_data in enumerate(self.train_data):
            self.loggers.on_batch_begin(batch_idx)
            if batch_idx % self.grad_acc_steps == 0:
                self.zero_grad_optimizers()

            batch_size, batch_data = self.preprocess_train_data(batch_data)
            batch_data = self.send_data_to_device(batch_data)

            with amp.autocast(enabled=self.use_amp):
                loss, batch_output = self.compute_train_forward(batch_data)
                loss = loss / self.grad_acc_steps

            self.compute_backward(loss)

            batch_metrics = self.compute_train_metrics(batch_output, batch_data)
            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            metric_acc.update(batch_metrics, batch_size)

            if (batch_idx + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch_idx
                self.cur_step += 1
                self.update_models()

            logs = metric_acc.metrics
            logs["step"] = self.cur_step
            logs.update(self.get_lrs())
            logs.update(self.get_wds())
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

            if self.finish_now():
                break

            if self.save_now():
                self.save_checkpoint()

            if self.validate_now():
                self.validation_loop()
                self.on_train_loop_begin()

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        logs.update(self.get_lrs())
        logs.update(self.get_wds())
        return logs

    @torch.no_grad
    def validation_loop(self):
        """Validation epoch loop"""
        metric_acc = MetricAcc(self.device)
        self.on_val_loop_begin()
        for batch_idx, batch_data in enumerate(self.val_data):
            batch_size, batch_data = self.preprocess_val_data(batch_data)
            batch_data = self.send_data_to_device(batch_data)
            with amp.autocast(enabled=self.use_amp):
                loss, batch_output = self.compute_val_forward(batch_data)

            batch_metrics = self.compute_val_metrics(batch_output, batch_data)
            batch_metrics["loss"] = loss.item()
            metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict(("val_" + k, v) for k, v in logs.items())
        return logs

    @torch.no_grad
    def bn_update_loop(self):
        """Batch normalization update loop"""
        metric_acc = MetricAcc(self.device)
        self.on_bn_update_loop_begin()
        for batch_idx, batch_data in enumerate(self.val_data):
            batch_size, batch_data = self.preprocess_train_data(batch_data)
            batch_data = self.send_data_to_device(batch_data)
            with amp.autocast(enabled=self.use_amp):
                loss, batch_output = self.compute_train_forward(batch_data)

            batch_metrics = self.compute_train_metrics(batch_output, batch_data)
            batch_metrics["loss"] = loss.item()
            metric_acc.update(batch_metrics, batch_size)
            if batch_idx > self.bn_update_steps:
                break

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        logs.update(self.get_lrs())
        logs.update(self.get_wds())
        return logs

    def _check_for_grad_nans(self, model, optim):
        """Checks for NaN in gradients when using fp16

        Args:
          model: model nn.Module
          optim: optimizer

        Returns:
          True if ok, False if NaNs found
        """
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                logging.warning(
                    f"Detected NaN values in gradients of parameter {n} / skip update"
                )
                optim.zero_grad()
                return False

        return True

    def _clip_grad_norm(self, model, optim, grad_clip, grad_clip_norm):
        if self.ddp:
            if self.ddp_type == DDPType.DDP:
                nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, norm_type=grad_clip_norm
                )
                return
            if self.ddp_type == DDPType.FULLY_SHARDED_DDP:
                # we have to use the member function in FullyShardedDDP class
                model.clip_grad_norm_(grad_clip, norm_type=grad_clip_norm)
                return
            else:
                # not sure about this but it looks like
                # we have to use the member function in the OSS optimizer wrapper
                optim.clip_grad_norm(grad_clip, norm_type=grad_clip_norm)

        # if no DDP clip normally
        nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, norm_type=grad_clip_norm
        )

    def _update_model_by_optim(
        self, model, optimizer, grad_clip, grad_clip_norm, use_amp, grad_scaler
    ):
        """Updates the model and does gradding clipping."""
        if use_amp:
            # is_ok = self._check_for_grad_nans(model, optimizer)
            # if not is_ok:
            #     return
            if grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                self._clip_grad_norm(model, optimizer, grad_clip, grad_clip_norm)

            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            if grad_clip > 0:
                self._clip_grad_norm(model, optimizer, grad_clip, grad_clip_norm)

            optimizer.step()

    def _make_optimizer(self, optim, model, oss=False):
        """Makes an optimizer object."""
        if isinstance(optim, torch.optim.Optimizer):
            return optim

        assert isinstance(optim, dict)
        opt_args = OF.filter_args(**optim)
        opt_args["oss"] = oss
        if self.rank == 0:
            logging.info("optimizer args={}".format(opt_args))

        optimizer = OF.create(model.trainable_param_groups(), **opt_args)
        return optimizer

    def _make_lr_sched(self, lr_sched, optim):
        """Makes a Learning Rate scheduler object."""
        if lr_sched is None or isinstance(lr_sched, LRS):
            return lr_sched

        assert isinstance(lr_sched, dict)
        args = LRSF.filter_args(**lr_sched)
        if self.rank == 0:
            logging.info(f"lr scheduler args={args}")
        lr_sched = LRSF.create(optim, **args)
        return lr_sched

    def _make_wd_sched(self, wd_sched, optim):
        """Makes a Learning Rate scheduler object."""
        if wd_sched is None or isinstance(wd_sched, WDS):
            return wd_sched

        assert isinstance(wd_sched, dict)
        args = WDSF.filter_args(**wd_sched)
        if self.rank == 0:
            logging.info(f"wd scheduler args={args}")
        wd_sched = WDSF.create(optim, **args)
        return wd_sched

    def _default_loggers(self, log_interval, use_tensorboard, use_wandb, wandb):
        """Creates the default data loaders"""
        prog_log = ProgLogger(interval=log_interval)
        csv_log = CSVLogger(self.exp_path / "train.log", append=True)
        loggers = [prog_log, csv_log]
        if use_tensorboard:
            loggers.append(
                TensorBoardLogger(self.exp_path / "tb", interval=log_interval)
            )
        if use_wandb:
            loggers.append(
                WAndBLogger(
                    **wandb, path=self.exp_path / "wandb", interval=log_interval
                )
            )
        return LoggerList(loggers)

    def _get_lrs(self, optim: torch.optim.Optimizer):
        """Returns the current learning rates of all param groups to show in the loggers"""
        lrs = {
            f"lr_{i}": param_group["lr"]
            for i, param_group in enumerate(optim.param_groups)
        }
        if len(lrs) == 1:
            lrs["lr"] = lrs.pop("lr_0")

        return lrs

    def _get_wds(
        self, optim: torch.optim.Optimizer, wd_scheduler: Optional[WDS] = None
    ):
        """Returns the current learning rates of all param groups to show in the loggers"""
        if wd_scheduler is None:
            return {}

        wds = {
            f"wd_{i}": param_group["weight_decay"]
            for i, param_group in enumerate(optim.param_groups)
        }
        if len(wds) == 1:
            wds["wd"] = wds.pop("wd_0")

        return wds

    def _compute_grad_acc_steps(self, data_loader):
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
        swa_model: Optional[TorchModel] = None,
        swa_scheduler: Optional[SWALR] = None,
        logs: Dict[str, Any] = None,
    ):
        """Creates a checkpoint of the training, to save and posterior recovery

        Args:
          logs: logs containing the current value of the metrics.
        """
        model.train()
        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            optimizer = cast(OSS, optimizer)
            optimizer.consolidate_state_dict()

        checkpoint = {
            "epoch": self.cur_epoch,
            "batch": self.cur_batch,
            "step": self.cur_step,
            "rng_state": torch.get_rng_state(),
            "model_cfg": model.get_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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

    def save_now(self):
        if self.save_hours is not None:
            t = time.time() / 3600
            dt = t - self.last_save_time
            if dt > self.save_hours:
                self.last_save_time = t
                return True

        if self.save_steps is not None:
            if self.cur_step % self.save_steps == 0:
                self.last_save_time = time.time() / 3600
                return True

    def validate_now(self):
        if self.val_hours is not None:
            t = time.time() / 3600
            dt = t - self.last_val_time
            if dt > self.val_hours:
                self.last_val_time = t
                return True

        if self.val_steps is not None:
            if self.cur_step % self.val_steps == 0:
                self.last_val_time = time.time() / 3600
                return True

    def finish_now(self):
        return self.max_steps is not None and self.cur_step > self.max_steps

    def save_model_checkpoint(
        self,
        model_name: str,
        checkpoint: Dict[str, Any],
    ):

        file_path = "%s/%s_ep%04d_step%010d.pth" % (
            self.exp_path,
            model_name,
            self.cur_epoch,
            self.cur_step,
        )

        logging.info("saving %s to %s", model_name, file_path)
        torch.save(checkpoint, file_path)

    def save_swa_model_checkpoint(self, model_name: str, checkpoint: Dict[str, Any]):
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_%s_ep%04d_step%010d.pth" % (
            self.exp_path,
            model_name,
            self.cur_epoch,
            self.cur_step,
        )

        torch.save(checkpoint, file_path)

    def _load_vars_from_checkpoint(self, checkpoint: Dict[str, Any]):
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
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRS] = None,
        wd_scheduler: Optional[WDS] = None,
        swa_model: Optional[TorchModel] = None,
        swa_scheduler: Optional[SWALR] = None,
    ):

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            model.module.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if wd_scheduler is not None:
            wd_scheduler.load_state_dict(checkpoint["wd_scheduler_state_dict"])

        if self.do_swa:
            if "swa_model_state_dict" in checkpoint:
                swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
                swa_scheduler.load_state_dict(checkpoint["swa_scheduler_state_dict"])
            # else:
            #     swa_scheduler = SWALR(
            #         optimizer,
            #         swa_lr=self.swa_lr,
            #         anneal_epochs=self.swa_anneal_steps,
            #     )

    def find_last_checkpoint(self, model_name: str = "model"):
        """finds the last checkpoint epoch and step in the experiment dir"""
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

    def load_last_checkpoint(self):
        """Loads the last training checkpoint in the experiment dir."""
        last_epoch, last_step = self.find_last_checkpoint()
        if last_epoch > 0 or last_step > 0:
            return self.load_checkpoint(last_epoch, last_step)

        return None

    def load_model_checkpoint(self, model_name="model", epoch=0, step=0):
        file_path = "%s/%s_ep%04d_steps%10d.pth" % (
            self.exp_path,
            model_name,
            epoch,
            step,
        )
        logging.info("loading %s from %s", model_name, file_path)
        return torch.load(file_path, map_location=torch.device("cpu"))

    def load_checkpoint(self, epoch, step):
        raise NotImplementedError()

    @staticmethod
    def get_augs_keys(batch, base_key, skip=set()):
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
    def filter_args(**kwargs):
        args = filter_func_args(TorchTrainerBase.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--exp-path", help="experiment path")
        parser.add_argument(
            "--grad-acc-steps",
            type=int,
            default=1,
            help="gradient accumulation batches before weight update",
        )
        parser.add_argument(
            "--eff-batch-size",
            type=int,
            default=None,
            help="effective total batch size, if given, it overrides grad_acc_steps",
        )
        parser.add_argument(
            "--num-epochs", type=int, default=200, help="number of epochs"
        )
        parser.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="maximum number of optimization steps",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=1000,
            help="how many batches to wait before logging training status",
        )
        parser.add_argument(
            "--save-steps",
            default=None,
            type=int,
            help="number of steps between model saves, if None only saves at the end of the epoch",
        )
        parser.add_argument(
            "--val-steps",
            default=None,
            type=int,
            help="number of steps between model validations, if None only validates at the end of the epoch",
        )
        parser.add_argument(
            "--save-hours",
            default=None,
            type=float,
            help="number of hours between model saves, if None only saves at the end of the epoch",
        )
        parser.add_argument(
            "--val-hours",
            default=None,
            type=float,
            help="number of hours between model validations, if None only validates at the end of the epoch",
        )
        parser.add_argument(
            "--use-tensorboard",
            action=ActionYesNo,
            default=False,
            help="use tensorboard logger",
        )
        parser.add_argument(
            "--use-wandb", action=ActionYesNo, default=False, help="use wandb logger"
        )
        parser.add_argument("--wandb.project", default=None, help="wandb project name")
        parser.add_argument("--wandb.group", default=None, help="wandb group name")
        parser.add_argument("--wandb.name", default=None, help="wandb display name")
        parser.add_argument(
            "--wandb.mode",
            default="online",
            choices=["online", "offline"],
            help="wandb mode (online, offline)",
        )

        parser.add_argument(
            "--ddp-type",
            default="ddp",
            choices=DDPType.choices(),
            help=f"DDP type in {DDPType.choices()}",
        )
        parser.add_argument(
            "--cpu-offload",
            action=ActionYesNo,
            default=False,
            help="CPU offload of gradients when using fully_sharded_ddp",
        )
        parser.add_argument(
            "--use-amp",
            action=ActionYesNo,
            default=False,
            help="use mixed precision training",
        )
        parser.add_argument(
            "--amp-dtype", default=AMPDType.FLOAT16.value, choices=AMPDType.choices()
        )

        parser.add_argument(
            "--grad-clip", type=float, default=0, help="gradient clipping norm value"
        )
        parser.add_argument(
            "--grad-clip-norm",
            default=2,
            choices=["inf", 1, 2],
            help="gradient clipping norm type",
        )
        parser.add_argument(
            "--swa-start",
            type=int,
            default=0,
            help="start step for SWA, if 0 it does not use SWA",
        )
        parser.add_argument(
            "--swa-lr", type=float, default=1e-3, help="learning rate for SWA phase"
        )
        parser.add_argument(
            "--swa-anneal-steps",
            type=int,
            default=50000,
            help="SWA learning rate anneal steps",
        )
        parser.add_argument(
            "--swa-update-steps",
            type=int,
            default=5000,
            help="Average SWA model every this number of steps",
        )
        parser.add_argument(
            "--bn-update-steps",
            type=int,
            default=5000,
            help="Run Batchnorm updates on the SWA model for this number of steps",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
