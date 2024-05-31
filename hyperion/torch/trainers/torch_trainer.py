"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import glob
import logging
import math
import os
import re
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from fairscale.optim.grad_scaler import ShardedGradScaler
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.optim.swa_utils import SWALR, AveragedModel

from ...utils.misc import filter_func_args
from ..loggers import CSVLogger, LoggerList, ProgLogger, TensorBoardLogger, WAndBLogger
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..optim import OptimizerFactory as OF
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


ddp_choices = [o.value for o in DDPType]


class TorchTrainer(object):
    """Base Trainer class to train basic neural network models

    Attributes:
      model: model object.
      loss: nn.Module loss class
      optim: pytorch optimizer object or optimizer options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object
      loggers: LoggerList object, loggers write training progress to std. output and file.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      train_mode: training mode in ['full', 'frozen']
      use_amp: uses mixed precision training.
      amp_dtype: "float16" | "bfloat16"
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
      grad_clip_norm: norm type to clip gradients
      swa_start: epoch to start doing swa
      swa_lr: SWA learning rate
      swa_anneal_epochs: SWA learning rate anneal epochs
      save_interval_steps: number of steps between model saves, if None only saves at the end of the epoch
      cpu_offload: CPU offload of gradients when using fully sharded ddp
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        model,
        loss,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        eff_batch_size=None,
        device=None,
        metrics=None,
        lrsched=None,
        wdsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        train_mode="full",
        use_amp=False,
        amp_dtype=AMPDType.FLOAT16,
        log_interval=1000,
        use_tensorboard=False,
        use_wandb=False,
        wandb={},
        grad_clip=0,
        grad_clip_norm=2,
        swa_start=0,
        swa_lr=1e-3,
        swa_anneal_epochs=10,
        save_interval_steps=None,
        cpu_offload=False,
        input_key="x",
        target_key="class_id",
    ):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.cur_epoch = cur_epoch
        self.cur_batch = 0
        self.grad_acc_steps = grad_acc_steps
        self.eff_batch_size = eff_batch_size
        self.exp_path = Path(exp_path)
        self.optim = optim
        self.lrsched = lrsched
        self.wdsched = wdsched

        if loggers is None:
            self.loggers = self._default_loggers(
                log_interval, use_tensorboard, use_wandb, wandb
            )
        elif isinstance(loggers, list):
            self.loggers = LoggerList(loggers)
        else:
            self.loggers = loggers

        self.metrics = metrics
        self.device = device
        self.train_mode = train_mode
        self.use_amp = use_amp
        self.amp_dtype = AMPDType.to_dtype(amp_dtype)
        self.grad_clip = grad_clip
        self.grad_clip_norm = grad_clip_norm
        self.swa_start = swa_start
        self.do_swa = swa_start > 0
        self.swa_lr = swa_lr
        self.swa_anneal_epochs = swa_anneal_epochs
        self.input_key = input_key
        self.target_key = target_key
        self.ddp = ddp
        self.ddp_type = ddp_type
        self.cpu_offload = cpu_offload
        self.rank = 0
        self.world_size = 1
        self.in_swa = False
        self.global_step = 0
        self.save_interval_steps = save_interval_steps
        if ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.set_train_mode()
        self.prepare_models_for_training()

    def prepare_models_for_training(self):
        self.loss = self._prepare_loss_for_training(self.loss, self.device)
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.grad_scaler,
            self.swa_model,
            self.swa_scheduler,
        ) = self._prepare_model_for_training(
            self.model,
            self.optim,
            self.lrsched,
            self.wdsched,
            self.device,
            self.use_amp,
            self.ddp,
            self.ddp_type,
            self.cpu_offload,
            self.do_swa,
            self.swa_lr,
            self.swa_anneal_epochs,
        )

    def _prepare_loss_for_training(self, loss, device):
        if loss is not None:
            loss.to(device)

        return loss

    def _prepare_model_for_training(
        self,
        model,
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
        swa_anneal_epochs,
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
                optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs
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

    def set_epoch(self, data_loader, cur_batch: int = 0):
        try:
            data_loader.dataset.set_epoch(self.cur_epoch)
        except AttributeError:
            logging.warning("dataset doesn't have set_epoch member function")

        try:
            data_loader.batch_sampler.set_epoch(self.cur_epoch, cur_batch)
        except AttributeError:
            logging.warning("sampler doesn't have set_epoch member function")

    def fit(self, train_data, val_data=None):
        """Training function, it performs the training and validation epochs

        Args:
          train_data: PyTorch data loader for the training loop
          val_data: PyTorch data loader for the validation loop
        """
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self._compute_grad_acc_steps(train_data)

        if self.do_swa and self.cur_epoch >= self.swa_start:
            self.in_swa = True

        val_logs = {}
        self.loggers.on_train_begin(epochs=self.epochs)
        for epoch in range(self.cur_epoch, self.epochs):
            self.set_epoch(train_data, self.cur_batch)
            self.loggers.on_epoch_begin(epoch, batches=len(train_data))
            if self.lr_scheduler is not None:
                # this is needed by cosine scheduler
                epoch_updates = int(len(train_data) / self.grad_acc_steps)
                self.lr_scheduler.on_epoch_begin(epoch, epoch_updates=epoch_updates)

            if self.wd_scheduler is not None:
                self.wd_scheduler.on_epoch_begin(epoch)

            logs = self.train_epoch(train_data)
            self.cur_batch = 0
            if val_data is not None:
                self.set_epoch(val_data)
                val_logs = self.validation_epoch(val_data)
                logs.update(val_logs)

            self.cur_epoch += 1

            self.loggers.on_epoch_end(logs)
            if self.do_swa and self.cur_epoch >= self.swa_start:
                self.in_swa = True
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_epoch_end(logs)
                if self.wd_scheduler is not None:
                    self.wd_scheduler.on_epoch_end()

            self.save_checkpoint(logs)

        if self.in_swa:
            self.loggers.on_epoch_begin(self.cur_epoch, batches=len(train_data))
            self.model = self.swa_model.module
            logs = self.bn_update_epoch(train_data)

            if val_data is not None:
                val_logs = self.validation_epoch(val_data)
                logs.update(val_logs)

            self.cur_epoch += 1
            self.loggers.on_epoch_end(logs)
            self.save_swa_model(logs)

    def set_train_mode(self):
        self.model.set_train_mode(self.train_mode)
        if self.rank == 0:
            self.model.parameter_summary(verbose=True)
            self.model.print_parameter_list()

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """
        batch_keys = [self.input_key, self.target_key]
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()
        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            input_data, target = tensors_subset(data, batch_keys, self.device)
            batch_size = input_data.size(0)

            with amp.autocast(enabled=self.use_amp):
                output = self.model(input_data)
                loss = self.loss(output, target) / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                self.update_model()
                self.save_checkpoint(partial=True)

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            lrs = self._get_lrs()
            logs.update(lrs)
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        lrs = self._get_lrs()
        logs.update(lrs)
        logs.update(self._get_wds())
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs.
          sw_update_bn: wheter or not, update batch-norm layers in SWA.
        """
        batch_keys = [self.input_key, self.target_key]
        metric_acc = MetricAcc(self.device)
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.model.train()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, data in enumerate(data_loader):
                x, target = tensors_subset(data, batch_keys, self.device)
                batch_size = x.size(0)
                with amp.autocast(enabled=self.use_amp):
                    output = self.model(x)
                    loss = self.loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    def bn_update_epoch(self, data_loader):
        logs = self.validation_epoch(data_loader, swa_update_bn=True)
        logs.update(self._get_lrs())
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
                logging.warn(
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
            is_ok = self._check_for_grad_nans(model, optimizer)
            if not is_ok:
                return
            if grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                self._clip_grad_norm(model, optimizer, grad_clip, grad_clip_norm)

            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            if grad_clip > 0:
                self._clip_grad_norm(model, optimizer, grad_clip, grad_clip_norm)

            optimizer.step()

    def update_model(self):
        """Updates the model and does gradding clipping."""
        if self.lr_scheduler is not None and not self.in_swa:
            self.lr_scheduler.on_opt_step()

        self._update_model_by_optim(
            self.model,
            self.optimizer,
            self.grad_clip,
            self.grad_clip_norm,
            self.use_amp,
            self.grad_scaler,
        )
        self.global_step += 1

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

    def _get_lr(self):
        """Returns the current learning rate to show in the loggers"""
        lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
        return max(lrs)

    def _get_lrs(self):
        """Returns the current learning rates of all param groups to show in the loggers"""
        lrs = {
            f"lr_{i}": param_group["lr"]
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        if len(lrs) == 1:
            lrs["lr"] = lrs.pop("lr_0")

        return lrs

    def _get_wd(self):
        """Returns the current learning rate to show in the loggers"""
        wds = [
            param_group["weight_decay"] for param_group in self.optimizer.param_groups
        ]
        return max(wds)

    def _get_wds(self, if_scheduler=True):
        """Returns the current learning rates of all param groups to show in the loggers"""
        if if_scheduler and self.wd_scheduler is None:
            return {}

        wds = {
            f"wd_{i}": param_group["weight_decay"]
            for i, param_group in enumerate(self.optimizer.param_groups)
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

    def checkpoint(self, logs=None):
        """Creates a checkpoint of the training, to save and posterior recovery

        Args:
          logs: logs containing the current value of the metrics.
        """
        self.model.train()
        checkpoint = {
            "epoch": self.cur_epoch,
            "batch": self.cur_batch,
            "global_step": self.global_step,
            "rng_state": torch.get_rng_state(),
            "model_cfg": self.model.get_config(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_state_dict": (
                self.loss.state_dict() if self.loss is not None else None
            ),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.wd_scheduler is not None:
            checkpoint["wd_scheduler_state_dict"] = self.wd_scheduler.state_dict()

        if logs is not None:
            checkpoint["logs"] = logs

        if self.in_swa:
            checkpoint["swa_model_state_dict"] = self.swa_model.state_dict()
            checkpoint["swa_scheduler_state_dict"] = self.swa_scheduler.state_dict()

        return checkpoint

    def save_partial_checkpoint(self):
        return (
            self.save_interval_steps is not None
            and self.global_step % self.save_interval_steps == 0
        )

    def save_checkpoint(self, logs=None, partial: bool = False):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
          partial: if True, it is saving in the middle of the epoch
        """
        if partial and not self.save_partial_checkpoint():
            return

        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            # optimizer = cast(OSS, optimizer)
            self.optimizer.consolidate_state_dict()

        if self.rank != 0:
            return

        checkpoint = self.checkpoint(logs)
        self.save_model_checkpoint("model", checkpoint, partial=partial)

    def save_model_checkpoint(
        self, model_name: str, checkpoint: Dict[str, Any], partial: bool = False
    ):
        if partial:
            file_path = "%s/%s_ep%04d_step%010d.pth" % (
                self.exp_path,
                model_name,
                self.cur_epoch,
                self.global_step,
            )
        else:
            file_path = "%s/%s_ep%04d.pth" % (self.exp_path, model_name, self.cur_epoch)

        logging.info("saving %s to %s", model_name, file_path)
        torch.save(checkpoint, file_path)

    def old_save_checkpoint(self, logs=None, partial: bool = False):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
          partial: if True, it is saving in the middle of the epoch
        """
        if partial and (
            self.save_interval_steps is None
            or self.global_step % self.save_interval_steps != 0
        ):
            return

        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            # optimizer = cast(OSS, optimizer)
            self.optimizer.consolidate_state_dict()

        if self.rank != 0:
            return

        checkpoint = self.checkpoint(logs)
        if partial:
            file_path = "%s/model_ep%04d_step%010d.pth" % (
                self.exp_path,
                self.cur_epoch,
                self.global_step,
            )
        else:
            file_path = "%s/model_ep%04d.pth" % (self.exp_path, self.cur_epoch)

        torch.save(checkpoint, file_path)

    def save_swa_model(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
        """
        if self.rank != 0:
            return

        checkpoint = self.checkpoint(logs)
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_model_ep%04d.pth" % (self.exp_path, self.cur_epoch)

        torch.save(checkpoint, file_path)

    def _load_checkpoint(self, checkpoint):
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

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.loss is not None:
            self.loss.load_state_dict(checkpoint["loss_state_dict"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if self.wd_scheduler is not None:
            self.wd_scheduler.load_state_dict(checkpoint["wd_scheduler_state_dict"])

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        elif self.lr_scheduler is not None:
            # this for older models that didn't save the global step
            self.global_step = self.lr_scheduler.step

        # if self.use_amp:
        #    amp.load_state_dict(checkpoint['amp'])
        if self.do_swa:
            if "swa_model_state_dict" in checkpoint:
                self.swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
                self.swa_scheduler.load_state_dict(
                    checkpoint["swa_scheduler_state_dict"]
                )
            else:
                self.swa_scheduler = SWALR(
                    self.optimizer,
                    swa_lr=self.swa_lr,
                    anneal_epochs=self.swa_anneal_epochs,
                )

        logs = None
        if "logs" in checkpoint:
            logs = checkpoint["logs"]

        del checkpoint
        # this was added before to try to release as much GPU memory as possible
        # Recently has started to cause CUDA not available devices error
        # Commenting for now.
        # if self.device is not None:
        #    torch.cuda.empty_cache()

        return logs

    def find_last_checkpoint(self, model_name="model"):
        """finds the last checkpoint epoch and step in the experiment dir"""
        last_epoch = 0
        last_step = 0
        file_pattern = "%s/%s_ep[0-9]*.pth" % (self.exp_path, model_name)
        file_paths = sorted(glob.glob(file_pattern))
        if len(file_paths) > 0:
            last_epoch = int(re.search(r"ep[0-9]*", file_paths[-1]).group()[2:])

        file_pattern = "%s/%s_ep%04d_step[0-9]*.pth" % (
            self.exp_path,
            model_name,
            last_epoch,
        )
        file_paths = sorted(glob.glob(file_pattern))
        if len(file_paths) > 0:
            last_step = int(re.search(r"step[0-9]*", file_paths[-1]).group()[4:])

        return last_epoch, last_step

    def load_last_checkpoint(self):
        """Loads the last training checkpoint in the experiment dir."""
        last_epoch, last_step = self.find_last_checkpoint()
        if last_epoch > 0 or last_step > 0:
            return self.load_checkpoint(last_epoch, last_step)

        return None

    def load_model_checkpoint(self, model_name="model", epoch=0, step=0):
        if step == 0:
            file_path = "%s/%s_ep%04d.pth" % (self.exp_path, model_name, epoch)
        else:
            file_path = "%s/%s_ep%04d_steps%10d.pth" % (
                self.exp_path,
                model_name,
                epoch,
                step,
            )
        logging.info("loading %s from %s", model_name, file_path)
        return torch.load(file_path, map_location=torch.device("cpu"))

    def load_checkpoint(self, epoch, step):
        checkpoint = self.load_model_checkpoint("model", epoch, step)
        return self._load_checkpoint(checkpoint)

    def old_load_checkpoint(self, file_path):
        """Loads a training checkpoint from file.

        Args:
           file_path: checkpoint file path
        """
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        return self._load_checkpoint(checkpoint)

    def old_load_last_checkpoint(self):
        """Loads the last training checkpoint in the experiment dir."""
        for epoch in range(self.epochs, 0, -1):
            file_path = Path("%s/model_ep%04d.pth" % (self.exp_path, epoch))
            if file_path.is_file():
                steps_pattern = "%s/model_ep%04d_steps*.pth" % (self.exp_path, epoch)
                steps_file_paths = sorted(glob.glob(steps_pattern))
                if len(steps_file_paths) > 0:
                    file_path = steps_file_paths[-1]

                return self.load_checkpoint(file_path)

        return None

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
        args = filter_func_args(TorchTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "optim" not in skip:
            OF.add_class_args(parser, prefix="optim")

        if "lrsched" not in skip:
            LRSF.add_class_args(parser, prefix="lrsched")

        if "wdsched" not in skip:
            WDSF.add_class_args(parser, prefix="wdsched")

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
        parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
        if train_modes is not None:
            parser.add_argument(
                "--train-mode",
                default="full",
                choices=train_modes,
                help=f"Available train modes for the model in {train_modes}",
            )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=1000,
            help="how many batches to wait before logging training status",
        )
        parser.add_argument(
            "--save-interval-steps",
            default=None,
            type=int,
            help="number of steps between model saves, if None only saves at the end of the epoch",
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
        # parser.add_argument(
        #     '--wandb.path', default=None,
        #     help='wandb directory')
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
            help="DDP type in {}".format(ddp_choices),
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
            "--cpu-offload",
            action=ActionYesNo,
            default=False,
            help="CPU offload of gradients when using fully_sharded_ddp",
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
            help="start epoch for SWA, if 0 it does not use SWA",
        )
        parser.add_argument(
            "--swa-lr", type=float, default=1e-3, help="learning rate for SWA phase"
        )
        parser.add_argument(
            "--swa-anneal-epochs",
            type=int,
            default=10,
            help="SWA learning rate anneal epochs",
        )

        parser.add_argument("--exp-path", help="experiment path")
        if "input_key" not in skip:
            parser.add_argument(
                "--input-key", default="x", help="dict. key for nnet input"
            )
        if "target_key" not in skip:
            parser.add_argument(
                "--target-key", default="class_id", help="dict. key for nnet targets"
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
