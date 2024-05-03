"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import time
from collections import OrderedDict as ODict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .torch_trainer import AMPDType
from .xvector_trainer import XVectorTrainer


class XVectorAdvTrainer(XVectorTrainer):
    """Adversarial Training of x-vectors with attack in feature domain

    Attributes:
      model: x-Vector model object.
      attack: adv. attack generator object
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      p_attack: attack probability
      p_val_attack: attack probability in validation
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lr_scheduler: learning rate scheduler object
      loggers: LoggerList object, loggers write training progress to std. output and file.
               If None, it uses default loggers.
      data_parallel: if True use nn.DataParallel
      loss: if None, it uses cross-entropy
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
      use_amp: uses mixed precision training.
      amp_dtype: "float16" | "bfloat16"
      log_interval: number of optim. steps between log outputs
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
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
        attack,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        eff_batch_size=None,
        p_attack=0.8,
        p_val_attack=0,
        device=None,
        metrics=None,
        lrsched=None,
        wdsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        loss=None,
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
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)
        self.attack = attack
        self.attack.to(device)
        self.p_attack = p_attack * self.grad_acc_steps
        self.p_val_attack = p_val_attack
        if self.p_attack > 1:
            logging.warning(
                (
                    "p-attack(%f) cannot be larger than 1./grad-acc-steps (%f)"
                    "because we can only create adv. signals in the "
                    "first step of the gradient acc. loop given that"
                    "adv optimization over-writes the gradients "
                    "stored in the model"
                )
                % (p_attack, 1.0 / self.grad_acc_steps)
            )

    def train_epoch(self, data_loader):
        batch_keys = [self.input_key, self.target_key]
        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            input_data, target = tensors_subset(data, batch_keys, self.device)
            batch_size = input_data.size(0)

            if batch % self.grad_acc_steps == 0:
                if torch.rand(1) < self.p_attack:
                    # generate adversarial attacks
                    logging.info("generating adv attack for batch=%d", batch)
                    self.model.eval()
                    data_adv = self.attack.generate(input_data, target)
                    max_delta = torch.max(torch.abs(data_adv - data)).item()
                    logging.info("adv attack max perturbation=%f", max_delta)
                    input_data = data_adv
                    self.model.train()

                self.optimizer.zero_grad()

            with amp.autocast(enabled=self.use_amp):
                output = self.model(input_data, target)
                loss = self.loss(output.logits, target) / self.grad_acc_steps

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
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        batch_keys = [self.input_key, self.target_key]
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()

        if swa_update_bn:
            log_tag = "train_"
            self.model.train()
        else:
            log_tag = "val_"
            self.model.eval()

        for batch, data in enumerate(data_loader):
            input_data, target = tensors_subset(data, batch_keys, self.device)
            batch_size = input_data.size(0)

            if torch.rand(1) < self.p_val_attack:
                # generate adversarial attacks
                self.model.eval()
                data = self.attack.generate(input_data, target)
                if swa_update_bn:
                    self.model.train()

            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp):
                    output = self.model(data, **self.amp_args)
                    loss = self.loss(output.logits, target)

            batch_metrics["loss"] = loss.item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    @staticmethod
    def filter_args(**kwargs):
        args = XVectorTrainer.filter_args(**kwargs)
        valid_args = ("p_attack", "p_val_attack")
        args_1 = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_1)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVectorTrainer.add_class_args(parser, skip=skip)
        parser.add_argument(
            "--p-attack",
            default=0.5,
            type=float,
            help="ratio of batches with adv attack",
        )
        parser.add_argument(
            "--p-val-attack",
            default=0.0,
            type=float,
            help="ratio of batches with adv attack in validation",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='trainer options')
