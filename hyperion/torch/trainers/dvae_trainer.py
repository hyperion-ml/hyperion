"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import Logger, LoggerList
from ..utils import MetricAcc, tensors_subset
from .legacy_torch_trainer import AMPDType, LegacyTorchTrainer


class DVAETrainer(LegacyTorchTrainer):
    """Denoising VAE trainer class

    Attributes:
      model: model object.
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
      ddp_type: distributed data parallel backend (only standard PyTorch DDP)
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
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
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        optim: Dict[str, Any] = {},
        epochs: int = 100,
        exp_path: str = "./train",
        cur_epoch: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lrsched: Optional[Dict[str, Any]] = None,
        wdsched: Optional[Dict[str, Any]] = None,
        loggers: Optional[Union[List[Logger], LoggerList]] = None,
        ddp: bool = False,
        ddp_type: str = "ddp",
        train_mode: str = "full",
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, Any] = {},
        grad_clip: float = 0,
        grad_clip_norm: float = 2,
        swa_start: int = 0,
        swa_lr: float = 1e-3,
        swa_anneal_epochs: int = 10,
        save_interval_steps: Optional[int] = None,
        input_key: str = "x_aug",
        target_key: str = "x",
    ) -> None:
        """Initializes a denoising VAE trainer.

        Args:
          model: Model to train.
          optim: Optimizer instance or configuration dictionary.
          epochs: Number of epochs.
          exp_path: Output directory.
          cur_epoch: Starting epoch.
          grad_acc_steps: Gradient accumulation factor.
          eff_batch_size: Desired effective batch size.
          device: Training device.
          metrics: Additional metric callables.
          lrsched: Learning-rate scheduler or configuration.
          wdsched: Weight-decay scheduler or configuration.
          loggers: None, a list of loggers, or a LoggerList instance.
          ddp: Enables distributed training.
          ddp_type: Distributed backend selector.
          train_mode: Model train mode.
          use_amp: Enables automatic mixed precision.
          amp_dtype: AMP dtype name.
          log_interval: Batch interval between log writes.
          use_tensorboard: Enables TensorBoard logging.
          use_wandb: Enables Weights & Biases logging.
          wandb: Weights & Biases options.
          grad_clip: Gradient clip value.
          grad_clip_norm: Gradient clip norm type.
          swa_start: Epoch at which SWA starts.
          swa_lr: SWA learning rate.
          swa_anneal_epochs: SWA annealing epochs.
          save_interval_steps: Partial checkpoint interval.
          input_key: Input key for dict batches.
          target_key: Target key for dict batches.
        """
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        # super().__init__(
        #     model,
        #     None,
        #     optim,
        #     epochs,
        #     exp_path,
        #     cur_epoch=cur_epoch,
        #     grad_acc_steps=grad_acc_steps,
        #     eff_batch_size=eff_batch_size,
        #     device=device,
        #     metrics=metrics,
        #     lrsched=lrsched,
        #     loggers=loggers,
        #     ddp=ddp,
        #     ddp_type=ddp_type,
        #     train_mode=train_mode,
        #     use_amp=use_amp,
        #     log_interval=log_interval,
        #     use_tensorboard=use_tensorboard,
        #     use_wandb=use_wandb,
        #     wandb=wandb,
        #     grad_clip=grad_clip,
        #     grad_clip_norm=grad_clip_norm,
        #     swa_start=swa_start,
        #     swa_lr=swa_lr,
        #     swa_anneal_epochs=swa_anneal_epochs,
        # )

    def train_epoch(self, data_loader: Any) -> Dict[str, Any]:
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning noisy and clean features

        Returns:
          Dictionary with training metrics.
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
            with amp.autocast(
                enabled=self.use_amp,
                dtype=self.amp_dtype,
                device_type=input_data.device.type,
            ):
                output = self.model(input_data, x_target=target, return_x_mean=True)

                elbo = output["elbo"].mean()
                loss = -elbo / self.grad_acc_steps
            x_hat = output["x_mean"]

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                self.update_model()
                self.save_checkpoint(partial=True)

            batch_metrics["elbo"] = elbo.item()
            for metric in ["log_px", "kldiv_z"]:
                batch_metrics[metric] = output[metric].mean().item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(x_hat, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            lrs = self._get_lrs()
            logs.update(lrs)
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        lrs = self._get_lrs()
        logs.update(lrs)
        return logs

    def validation_epoch(
        self, data_loader: Any, swa_update_bn: bool = False
    ) -> Dict[str, Any]:
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs

        Returns:
          Dictionary with validation metrics.
        """
        batch_keys = [self.input_key, self.target_key]
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.model.train()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, data in enumerate(data_loader):
                input_data, target = tensors_subset(data, batch_keys, self.device)
                batch_size = input_data.size(0)
                with amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                    device_type=input_data.device.type,
                ):
                    output = self.model(input_data, x_target=target, return_x_mean=True)

                x_hat = output["x_mean"]
                for metric in ["elbo", "log_px", "kldiv_z"]:
                    batch_metrics[metric] = output[metric].mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(x_hat, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    @staticmethod
    def add_class_args(
        parser: Any,
        prefix: Optional[str] = None,
        train_modes: Optional[list] = None,
        skip: set = set(),
    ) -> None:
        """Registers denoising VAE trainer arguments on a parser.

        Args:
          parser: Parser instance to extend.
          prefix: Optional nested prefix.
          train_modes: Allowed train-mode values.
          skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        LegacyTorchTrainer.add_class_args(
            parser,
            train_modes=train_modes,
            skip=skip.union({"input_key", "target_key"}),
        )
        if "input_key" not in skip:
            parser.add_argument(
                "--input-key", default="x_aug", help="dict. key for nnet input"
            )

        if "target_key" not in skip:
            parser.add_argument(
                "--target-key", default="x", help="dict. key for nnet targets"
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
