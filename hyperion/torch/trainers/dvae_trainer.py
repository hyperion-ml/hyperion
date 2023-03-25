"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .torch_trainer import TorchTrainer


class DVAETrainer(TorchTrainer):
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
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
      use_amp: uses mixed precision training.
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
      grad_clip_norm: norm type to clip gradients
      swa_start: epoch to start doing swa
      swa_lr: SWA learning rate
      swa_anneal_epochs: SWA learning rate anneal epochs
      cpu_offload: CPU offload of gradients when using fully sharded ddp
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        model,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        eff_batch_size=None,
        device=None,
        metrics=None,
        lrsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        train_mode="full",
        use_amp=False,
        log_interval=10,
        use_tensorboard=False,
        use_wandb=False,
        wandb={},
        grad_clip=0,
        grad_clip_norm=2,
        swa_start=0,
        swa_lr=1e-3,
        swa_anneal_epochs=10,
        cpu_offload=False,
        input_key="x_aug",
        target_key="x",
    ):
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
        #     cpu_offload=cpu_offload,
        # )

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning noisy and clean features
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
                output = self.model(input_data, x_target=target, return_x_mean=True)

                elbo = output["elbo"].mean()
                loss = -elbo / self.grad_acc_steps
            x_hat = output["x_mean"]

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics["elbo"] = elbo.item()
            for metric in ["log_px", "kldiv_z"]:
                batch_metrics[metric] = output[metric].mean().item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(x_hat, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        logs["lr"] = self._get_lr()
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
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
                with amp.autocast(enabled=self.use_amp):
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
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        super().add_class_args(
            parser, train_modes, skip=skip.union({"input_key", "target_key"})
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
