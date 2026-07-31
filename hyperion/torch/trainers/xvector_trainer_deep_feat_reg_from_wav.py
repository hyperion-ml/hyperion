"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn

from ...utils.misc import filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import Logger, LoggerList
from ..utils import MetricAcc, tensors_subset
from .legacy_torch_trainer import AMPDType
from .xvector_trainer_deep_feat_reg import XVectorTrainerDeepFeatReg


class XVectorTrainerDeepFeatRegFromWav(XVectorTrainerDeepFeatReg):
    """Trainer to train x-vector style models.

    Attributes:
      model: x-Vector model object that we want to fine-tune
      feat_extractor: feature extractor nn.Module
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
      lrsched: learning rate scheduler object or options dict.
      loggers: LoggerList object, loggers write training progress to std. output and file.
      ddp: if True use distributed data parallel training
      ddp_type: distributed data parallel backend (only standard PyTorch DDP)
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
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        model: HyperTorchModel,
        feat_extractor: Any,
        prior_model: HyperTorchModel,
        optim: Dict[str, Any] = {},
        epochs: int = 100,
        exp_path: str = "./train",
        cur_epoch: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        reg_layers_enc: Optional[list] = None,
        reg_layers_classif: Optional[list] = None,
        reg_weight_enc: float = 0.1,
        reg_weight_classif: float = 0.1,
        device: Optional[torch.device] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lrsched: Optional[Dict[str, Any]] = None,
        wdsched: Optional[Dict[str, Any]] = None,
        loggers: Optional[Union[List[Logger], LoggerList]] = None,
        ddp: bool = False,
        ddp_type: str = "ddp",
        loss: Optional[nn.Module] = None,
        reg_loss: Optional[nn.Module] = None,
        train_mode: str = "full",
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 10,
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
        """Initializes the wav-based deep-feature regularized trainer.

        Args:
          model: Model to train.
          feat_extractor: Feature extractor used before the model.
          prior_model: Frozen reference model used as regularizer.
          optim: Optimizer instance or configuration dictionary.
          epochs: Number of epochs.
          exp_path: Output directory.
          cur_epoch: Starting epoch.
          grad_acc_steps: Gradient accumulation factor.
          eff_batch_size: Desired effective batch size.
          reg_layers_enc: Encoder layer indices to regularize.
          reg_layers_classif: Classifier layer indices to regularize.
          reg_weight_enc: Encoder regularization weight.
          reg_weight_classif: Classifier regularization weight.
          device: Training device.
          metrics: Additional metric callables.
          lrsched: Learning-rate scheduler or configuration.
          wdsched: Weight-decay scheduler or configuration.
          loggers: None, a list of loggers, or a LoggerList instance.
          ddp: Enables distributed training.
          ddp_type: Distributed backend selector.
          loss: Classification loss module.
          reg_loss: Regularization loss module.
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

        self.feat_extractor = feat_extractor
        if device is not None:
            self.feat_extractor.to(device)

    def train_epoch(self, data_loader: Any) -> Dict[str, Any]:
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs

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
            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            input_data, target = tensors_subset(data, batch_keys, self.device)
            batch_size = input_data.size(0)
            with torch.no_grad():
                feats = self.feat_extractor(input_data)

            with amp.autocast(
                enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
            ):
                outputs = self.model(
                    feats,
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

                loss = self.loss(
                    output, target
                )  # you need to take the mean here because of the multi-gpu training
                batch_metrics["loss-classif"] = loss.item()

                prior_outputs = self.prior_model(
                    feats,
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
