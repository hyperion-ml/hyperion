"""
 Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import os
from collections import OrderedDict as ODict

import torch
import torch.nn as nn
import torchaudio
from jsonargparse import ActionParser, ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .torch_trainer import TorchTrainer
# from ..losses.focal_loss import FocalLoss
# from torchvision.ops.focal_loss import sigmoid_focal_loss


class LanguageIDTrainer(TorchTrainer):
    """Trainer to train Language identification style models.

    Attributes:
      model: Language identification model object.
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
        input_key="x",
        target_key="language",
        loss_weight=None,
        loss_weight_exp=0.5,
    ):

        if loss == "CE" or loss is None:
            loss = nn.CrossEntropyLoss()
        elif loss == "weightedCE":
            loss = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight.values, dtype=torch.float).to(device)**(-loss_weight_exp))
            logging.info(torch.tensor(loss_weight.values).to(device)**(-loss_weight_exp))
        elif loss == "focal_loss":
            loss = FocalLoss(alpha=torch.tensor(focal_weight.values).to(device)**(-loss_weight_exp), gamma=2, size_average=True)
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

    @record
    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """
        batch_keys = [
            self.input_key, self.target_key
        ]

        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
            input_data, target = tensors_subset(
                data, batch_keys, self.device)
            # input_data, input_lengths, target = tensors_subset(
                # data, batch_keys, self.device)
            batch_size = input_data.shape[0]

            with self.amp_autocast():
                # TODO: Check and Modify output, loss from the model
                # output, loss = self.model(data,
                #                           x_lengths=audio_length,
                #                           y=target)
                # loss = loss.mean() / self.grad_acc_steps
                output = self.model(input_data, y=target)
                loss = self.loss(output, target).mean() / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

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
          data_loader: PyTorch data loader return input/output pairs.
          sw_update_bn: wheter or not, update batch-norm layers in SWA.
        """
        batch_keys = [
            self.input_key, self.target_key
        ]
        metric_acc = MetricAcc(self.device)
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.train()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, data in enumerate(data_loader):
                input_data, target = tensors_subset(
                    data, batch_keys, self.device)
                # input_data, input_lengths, target = tensors_subset(
                    # data, batch_keys, self.device)
                batch_size = input_data.shape[0]
                # data, target = data.to(self.device), target.to(self.device)
                # batch_size = data.shape[0]

                with self.amp_autocast():
                    output = self.model(input_data, y=target)
                    loss = self.loss(output, target).mean() / self.grad_acc_steps

                    # output, loss = self.model(data,
                    #                           x_lengths=audio_length,
                    #                           y=target)
                    # output = self.model(data)
                    # loss = self.loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(LanguageIDTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        super_skip = skip.copy()
        super_skip.add("target_key")
        TorchTrainer.add_class_args(parser,
                                    train_modes=train_modes,
                                    skip=super_skip)
        if "target_key" not in skip:
            parser.add_argument("--target-key",
                                default="language",
                                help="dict. key for nnet targets")
        if "loss" not in skip:
            parser.add_argument("--loss",
                                default=None,
                                choices=["CE", "weightedCE", "focal_loss"],
                                help="loss function")
        if "loss_weight_exp" not in skip:
            parser.add_argument("--loss-weight-exp",
                                default=0.5,
                                type=float,
                                help="focal loss weight exponent")
        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
