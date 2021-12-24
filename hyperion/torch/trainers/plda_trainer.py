"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc
from ..utils.misc import get_selfsim_tarnon
from ..losses import BCEWithLLR
from .torch_trainer import TorchTrainer


class PLDATrainer(TorchTrainer):
    """Trainer to train PLDA back-end

    Attributes:
      model: PLDA model object.
      optim: pytorch optimizer object
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object
      loggers: LoggerList object, loggers write training progress to std. output and file.
               If None, it uses default loggers.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      loss: if None, it uses cross-entropy
      loss_weights: dictionary with weights for multiclass and binary cross-entropies

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
    """

    def __init__(
        self,
        model,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        device=None,
        metrics=None,
        lrsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        loss=None,
        loss_weights={"multi": 1, "bin": 0},
        p_tar=0.5,
        train_mode="train",
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
    ):

        if loss is None:
            loss = nn.CrossEntropyLoss()
        super().__init__(
            model,
            loss,
            optim,
            epochs,
            exp_path,
            cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps,
            device=device,
            metrics=metrics,
            lrsched=lrsched,
            loggers=loggers,
            ddp=ddp,
            ddp_type=ddp_type,
            train_mode=train_mode,
            use_amp=use_amp,
            log_interval=log_interval,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            wandb=wandb,
            grad_clip=grad_clip,
            grad_clip_norm=grad_clip_norm,
            swa_start=swa_start,
            swa_lr=swa_lr,
            swa_anneal_epochs=swa_anneal_epochs,
            cpu_offload=cpu_offload,
        )

        self.loss_bce = BCEWithLLR(p_tar)
        self.loss_weights = loss_weights

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """

        self.model.update_margin(self.cur_epoch)

        return_multi = self.loss_weights["multi"] > 0
        return_bin = self.loss_weights["bin"] > 0
        target_bin = None

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        self.set_train_mode()
        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            if return_bin:
                target_bin, mask_bin = get_selfsim_tarnon(target, return_mask=True)
            with self.amp_autocast():
                output = self.model(
                    data,
                    target,
                    return_multi=return_multi,
                    return_bin=return_bin,
                    y_bin=target_bin,
                )
                loss = 0
                if return_multi:
                    loss_multi = self.loss(output["multi"], target).mean()
                    loss = loss + self.loss_weights["multi"] * loss_multi
                if return_bin:
                    output_bin = output["bin"][mask_bin]
                    target_bin = target_bin[mask_bin]
                    loss_bin = self.loss_bce(output_bin, target_bin).mean()
                    loss = loss + self.loss_weights["bin"] * loss_bin

                loss = loss / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            if return_bin:
                batch_metrics["loss_bin"] = loss_bin.item()
            if return_multi:
                batch_metrics["loss_multi"] = loss_multi.item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output["multi"], target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs["lr"] = self._get_lr()
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        return_multi = self.loss_weights["multi"] > 0
        return_bin = self.loss_weights["bin"] > 0

        with torch.no_grad():
            if swa_update_bn:
                log_tag = ""
                self.set_train_mode()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                if return_bin:
                    target_bin, mask_bin = get_selfsim_tarnon(target, return_mask=True)
                with self.amp_autocast():
                    output = self.model(
                        data, return_multi=return_multi, return_bin=return_bin
                    )
                    loss = 0
                    if return_multi:
                        loss_multi = self.loss(output["multi"], target).mean()
                        loss = loss + self.loss_weights["multi"] * loss_multi
                    if return_bin:
                        output_bin = output["bin"][mask_bin]
                        target_bin = target_bin[mask_bin]
                        loss_bin = self.loss_bce(output_bin, target_bin).mean()
                        loss = loss + self.loss_weights["bin"] * loss_bin

                batch_metrics["loss"] = loss.item()
                if return_bin:
                    batch_metrics["loss_bin"] = loss_bin.item()
                if return_multi:
                    batch_metrics["loss_multi"] = loss_multi.item()
                    for k, metric in self.metrics.items():
                        batch_metrics[k] = metric(output["multi"], target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs
