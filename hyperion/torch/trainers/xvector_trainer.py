"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import os
from collections import OrderedDict as ODict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.distributed.elastic.multiprocessing.errors import record

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .torch_trainer import TorchTrainer


class XVectorTrainer(TorchTrainer):
    """Trainer to train x-vector style models.

    Attributes:
      model: x-Vector model object.
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object or options dict
      loggers: LoggerList object, loggers write training progress to std. output and file.
               If None, it uses default loggers.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      loss: if None, it uses cross-entropy
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
      save_interval_steps: number of steps between model saves, if None only saves at the end of the epoch
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
        loss=None,
        train_mode="full",
        use_amp=False,
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
        if loss is None:
            loss = nn.CrossEntropyLoss()

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

    @record
    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """
        batch_keys = [self.input_key, self.target_key]
        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            input_keys = self.get_augs_keys(data, self.input_key)
            loss_scale = self.grad_acc_steps * len(input_keys)
            for aug_key in input_keys:
                batch_keys = [aug_key, self.target_key]
                x, target = tensors_subset(data, batch_keys, self.device)
                batch_size = x.size(0)
                with amp.autocast(enabled=self.use_amp):
                    output = self.model(x, y=target)
                    loss = self.loss(output, target) / loss_scale

                if self.use_amp:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                self.update_model()
                self.save_checkpoint(partial=True)

            batch_metrics["loss"] = loss.item() * loss_scale
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
        return logs
