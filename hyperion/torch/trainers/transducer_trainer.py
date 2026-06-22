"""
Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict
from typing import Any, Dict, Optional

import torch
import torch.amp as amp
import torch.nn as nn
import torchaudio
from jsonargparse import ActionParser, ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

from ...utils.misc import filter_func_args
from ..utils import MetricAcc, tensors_subset
from .legacy_torch_trainer import AMPDType, LegacyTorchTrainer


class TransducerTrainer(LegacyTorchTrainer):
    """Trainer to train ASR style models.

    Attributes:
      model: ASR model object.
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
      grad_clip_norm: norm type to clip gradients
      swa_start: epoch to start doing swa
      swa_lr: SWA learning rate
      swa_anneal_epochs: SWA learning rate anneal epochs
      save_interval_steps: number of steps between model saves, if None only saves at the end of the epoch
    """

    def __init__(
        self,
        model: Any,
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
        loggers: Optional[Any] = None,
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
        grad_clip_norm: int = 2,
        swa_start: int = 0,
        swa_lr: float = 1e-3,
        swa_anneal_epochs: int = 10,
        save_interval_steps: Optional[int] = None,
        input_key: str = "x",
        target_key: str = "text",
    ) -> None:
        """Initializes the transducer trainer.

        Args:
          model: ASR model instance.
          optim: Optimizer instance or configuration dictionary.
          epochs: Number of epochs to train for.
          exp_path: Directory used for logs and checkpoints.
          cur_epoch: Epoch to resume from.
          grad_acc_steps: Number of batches to accumulate before optimizer steps.
          eff_batch_size: Optional effective batch size reference.
          device: Target device for training.
          metrics: Additional metric callables.
          lrsched: Learning-rate scheduler or configuration.
          wdsched: Weight-decay scheduler or configuration.
          loggers: Logger collection.
          ddp: Whether distributed training is enabled.
          ddp_type: Distributed backend name.
          loss: Loss module used to train the model.
          train_mode: Model train mode.
          use_amp: Whether automatic mixed precision is enabled.
          amp_dtype: AMP precision to use when ``use_amp`` is true.
          log_interval: Number of steps between logger updates.
          use_tensorboard: Whether to enable TensorBoard logging.
          use_wandb: Whether to enable Weights & Biases logging.
          wandb: W&B configuration dictionary.
          grad_clip: Gradient clipping threshold.
          grad_clip_norm: Gradient norm type.
          swa_start: Step at which SWA starts.
          swa_lr: SWA learning rate.
          swa_anneal_epochs: SWA learning-rate annealing epochs.
          save_interval_steps: Number of steps between checkpoint saves.
          input_key: Batch key for input features.
          target_key: Batch key for supervision targets.

        Returns:
          None.
        """
        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

    @record
    def train_epoch(self, data_loader: Any) -> Dict[str, Any]:
        """Runs one training epoch.

        Args:
          data_loader: pytorch data loader returning features and class labels.

        Returns:
          Dictionary with training metrics.
        """
        batch_keys = [self.input_key, f"{self.input_key}_lengths", self.target_key]
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            # # TODO: Check and Modify data, target
            # data, audio_length, target = data.to(self.device), audio_length.to(
            #     self.device), target.to(self.device)
            # print(data.keys(), batch_keys, flush=True)
            input_data, input_lengths, target = tensors_subset(
                data, batch_keys, self.device
            )
            batch_size = input_data.shape[0]

            with amp.autocast(
                enabled=self.use_amp,
                dtype=self.amp_dtype,
                device_type=input_data.device.type,
            ):
                output = self.model(input_data, x_lengths=input_lengths, y=target)
                loss = output.loss
                loss = loss.mean() / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                self.update_model()
                self.save_checkpoint(partial=True)

            for k, v in output.items():
                if "loss" in k and v is not None:
                    batch_metrics[k] = output[k].item()

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

    def validation_epoch(
        self, data_loader: Any, swa_update_bn: bool = False
    ) -> Dict[str, Any]:
        """Runs one validation epoch.

        Args:
          data_loader: PyTorch data loader return input/output pairs.
          swa_update_bn: Whether to update batch-norm layers in SWA mode.

        Returns:
          Dictionary with validation metrics.
        """
        batch_keys = [self.input_key, f"{self.input_key}_lengths", self.target_key]
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
                input_data, input_lengths, target = tensors_subset(
                    data, batch_keys, self.device
                )
                batch_size = input_data.shape[0]

                # data, audio_length, target = data.to(
                #     self.device), audio_length.to(self.device), target.to(
                #         self.device)
                # batch_size = data.shape[0]
                # data, target = data.to(self.device), target.to(self.device)
                # batch_size = data.shape[0]

                with amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                    device_type=input_data.device.type,
                ):
                    output = self.model(input_data, x_lengths=input_lengths, y=target)

                for k, v in output.items():
                    if "loss" in k and v is not None:
                        batch_metrics[k] = output[k].item()

                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        train_modes: Optional[list[str]] = None,
        skip: Optional[set[str]] = None,
    ) -> None:
        """Registers the CLI arguments required to build this trainer.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix.
          train_modes: Optional list of allowed train modes.
          skip: Optional set of argument names to omit.

        Returns:
          None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        super_skip = skip.copy()
        super_skip.add("target_key")
        LegacyTorchTrainer.add_class_args(
            parser, train_modes=train_modes, skip=super_skip
        )
        if "target_key" not in skip:
            parser.add_argument(
                "--target-key", default="text", help="dict. key for nnet targets"
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
