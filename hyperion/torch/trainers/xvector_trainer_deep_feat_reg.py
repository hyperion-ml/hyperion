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
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .torch_trainer import AMPDType
from .xvector_trainer import XVectorTrainer


class XVectorTrainerDeepFeatReg(XVectorTrainer):
    """Trainer to train x-vector style models.

    Attributes:
      model: x-Vector model object that we want to fine-tune
      prior_model: x-Vector model object that we use as regularizer
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      reg_layers_enc: list of encoder layer indexes that we use for regularization
      reg_layers_classif: list of classification head layer indexes that we use for regularization
      reg_weight_enc: weight of the regularization loss for encoder hidden activations
      reg_weight_classif: weight of the regularization loss for classification head hidden activations
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object or options dict.
      loggers: LoggerList object, loggers write training progress to std. output and file.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      loss: if None, it uses cross-entropy
      reg_loss: nn.Module loss used for regularization, if None it uses L1 loss.
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
      cpu_offload: CPU offload of gradients when using fully sharded ddp
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        model,
        prior_model,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        eff_batch_size=None,
        reg_layers_enc=None,
        reg_layers_classif=None,
        reg_weight_enc=0.1,
        reg_weight_classif=0.1,
        device=None,
        metrics=None,
        lrsched=None,
        wdsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        loss=None,
        reg_loss=None,
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

        self.prior_model = prior_model
        if reg_loss is None or reg_loss == "l1":
            reg_loss = nn.L1Loss()
        elif reg_loss == "mse":
            reg_loss = nn.MSELoss()
        self.reg_loss = reg_loss
        self.reg_layers_enc = reg_layers_enc
        self.reg_layers_classif = reg_layers_classif
        self.reg_weight_enc = reg_weight_enc
        self.reg_weight_classif = reg_weight_classif

        if device is not None:
            self.prior_model.to(device)

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
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

            input_data, target = tensors_subset(data, batch_keys, self.device)
            batch_size = input_data.size(0)
            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_data,
                    y=target,
                    return_enc_layers=self.reg_layers_enc,
                    return_classif_layers=self.reg_layers_classif,
                    return_output=True,
                )
                h_enc, h_classif, output = (
                    outputs["h_enc"],
                    outputs["h_classif"],
                    outputs["logits"],
                )

                loss = self.loss(output, target)
                batch_metrics["loss-classif"] = loss.item()

                prior_outputs = self.prior_model(
                    input_data,
                    return_enc_layers=self.reg_layers_enc,
                    return_classif_layers=self.reg_layers_classif,
                    return_output=False,
                )
                prior_h_enc, prior_h_classif = (
                    prior_outputs["h_enc"],
                    prior_outputs["h_classif"],
                )

                n_enc = len(h_enc)
                if n_enc > 0:
                    loss_scale = self.reg_weight_enc / n_enc
                for i in range(n_enc):
                    l = self.reg_layers_enc[i]
                    loss_i = self.reg_loss(h_enc[i], prior_h_enc[i]).mean()
                    loss_name = "reg-h-enc-%d" % l
                    batch_metrics[loss_name] = loss_i.item()
                    loss += loss_scale * loss_i

                n_classif = len(h_classif)
                if n_classif > 0:
                    loss_scale = self.reg_weight_classif / n_classif
                for i in range(n_classif):
                    l = self.reg_layers_classif[i]
                    loss_i = self.reg_loss(h_classif[i], prior_h_classif[i]).mean()
                    loss_name = "reg-h-classif-%d" % l
                    batch_metrics[loss_name] = loss_i.item()
                    loss += loss_scale * loss_i

                batch_metrics["loss"] = loss.item()
                loss = loss / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                self.update_model()
                self.save_checkpoint(partial=True)

            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs = ODict(("train_" + k, v) for k, v in logs.items())
            lrs = self._get_lrs()
            logs.update(lrs)
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        lrs = self._get_lrs()
        logs.update(lrs)
        return logs

    @staticmethod
    def filter_args(**kwargs):
        args = XVectorTrainer.filter_args(**kwargs)
        valid_args = (
            "reg_layers_enc",
            "reg_layers_classif",
            "reg_weight_enc",
            "reg_weight_classif",
            "reg_loss",
        )
        args_1 = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_1)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=[]):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        XVectorTrainer.add_class_args(parser, skip=skip)
        parser.add_argument(
            "--reg-layers-enc",
            type=int,
            default=None,
            nargs="+",
            help="list of layers from the encoder nnet to use for regularization ",
        )
        parser.add_argument(
            "--reg-layers-classif",
            type=int,
            default=None,
            nargs="+",
            help="list of layers from the classif nnet to use for regularization ",
        )
        parser.add_argument(
            "--reg-weight-enc",
            type=float,
            default=0.1,
            help="weight for regularization from enc layers",
        )
        parser.add_argument(
            "--reg-weight-classif",
            type=float,
            default=0.1,
            help="weight for regularization from classif layers",
        )
        parser.add_argument(
            "--reg-loss",
            default="l1",
            choices=["l1", "mse"],
            help=("type of regularization loss"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
