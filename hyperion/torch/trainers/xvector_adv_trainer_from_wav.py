"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import time
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
from .legacy_torch_trainer import AMPDType
from .xvector_trainer_from_wav import XVectorTrainerFromWav


class XVectorAdvTrainerFromWav(XVectorTrainerFromWav):
    """Adversarial Training of x-vectors with attack in feature domain

    Attributes:
      model: x-Vector model object.
      feat_extractor: feature extractor nn.Module
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
      lrsched: learning rate scheduler object or options dict
      loggers: LoggerList object, loggers write training progress to std. output and file.
               If None, it uses default loggers.
      ddp: if True use distributed data parallel training
      ddp_type: distributed data parallel backend (only standard PyTorch DDP)
      loss: if None, it uses cross-entropy
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
      use_amp: uses mixed precision training.
      amp_dtype: "float16" | "bfloat16"
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
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
        feat_extractor: Any,
        attack: Any,
        optim: Dict[str, Any] = {},
        epochs: int = 100,
        exp_path: str = "./train",
        cur_epoch: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        p_attack: float = 0.8,
        p_val_attack: float = 0,
        device: Optional[torch.device] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lrsched: Optional[Dict[str, Any]] = None,
        wdsched: Optional[Dict[str, Any]] = None,
        loggers: Optional[Union[List[Logger], LoggerList]] = None,
        ddp: bool = False,
        ddp_type: str = "ddp",
        loss: Optional[nn.Module] = None,
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
        input_key: str = "x",
        target_key: str = "class_id",
    ) -> None:
        """Initializes the adversarial wav-based x-vector trainer.

        Args:
          model: Model to train.
          feat_extractor: Feature extractor used before the model.
          attack: Adversarial attack generator.
          optim: Optimizer instance or configuration dictionary.
          epochs: Number of epochs.
          exp_path: Output directory.
          cur_epoch: Starting epoch.
          grad_acc_steps: Gradient accumulation factor.
          eff_batch_size: Desired effective batch size.
          p_attack: Probability of generating adversarial examples.
          p_val_attack: Validation-time adversarial probability.
          device: Training device.
          metrics: Additional metric callables.
          lrsched: Learning-rate scheduler or configuration.
          wdsched: Weight-decay scheduler or configuration.
          loggers: None, a list of loggers, or a LoggerList instance.
          ddp: Enables distributed training.
          ddp_type: Distributed backend selector.
          loss: Loss module, defaults to cross-entropy.
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
                ),
                p_attack,
                1.0 / self.grad_acc_steps,
            )

    def train_epoch(self, data_loader: Any) -> Dict[str, Any]:
        """Runs one adversarial training epoch.

        Args:
          data_loader: Training data loader.

        Returns:
          Dictionary with training metrics.
        """
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
                    # logging.info('generating adv attack for batch=%d' % (batch))
                    self.model.eval()
                    data_adv = self.attack.generate(input_data, target)
                    max_delta = torch.max(torch.abs(data_adv - input_data)).item()
                    input_data = data_adv
                    self.model.train()

                self.optimizer.zero_grad()

            with torch.no_grad():
                feats = self.feat_extractor(input_data)

            with amp.autocast(
                enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
            ):
                output = self.model(feats, y=target)
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
                batch_metrics[k] = metric(output.logits, target)

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
        """Runs one validation epoch with optional adversarial examples.

        Args:
          data_loader: Validation data loader.
          swa_update_bn: Whether to update batch-norm layers for SWA.

        Returns:
          Dictionary with validation metrics.
        """
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
                input_data = self.attack.generate(input_data, target)
                if swa_update_bn:
                    self.model.train()

            with torch.no_grad():
                feats = self.feat_extractor(input_data)
                with amp.autocast(
                    enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
                ):
                    output = self.model(feats)
                    loss = self.loss(output.logits, target)

            batch_metrics["loss"] = loss.item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output.logits, target)

            metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments for :class:`XVectorAdvTrainerFromWav`.

        Returns:
          Keyword arguments accepted by the trainer constructor.
        """
        args = XVectorTrainerFromWav.filter_args(**kwargs)
        valid_args = ("p_attack", "p_val_attack")
        args_1 = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(args_1)
        return args

    @staticmethod
    def add_class_args(
        parser: Any,
        prefix: Optional[str] = None,
        train_modes: Optional[List[str]] = None,
        skip: set = set(),
    ) -> None:
        """Registers adversarial wav-trainer arguments on a parser.

        Args:
          parser: Parser instance to extend.
          prefix: Optional nested prefix.
          train_modes: Allowed model train-mode values.
          skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVectorTrainerFromWav.add_class_args(parser, train_modes=train_modes, skip=skip)
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
