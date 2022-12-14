"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import sys
from collections import OrderedDict as ODict

import logging
import copy
import math

import torch
import torch.nn as nn

from ..utils import MetricAcc, TorchDDP
from ..utils import cancel_gradients_last_layer
from .xvector_trainer_dinossl import DINOSSLXVectorTrainer


class DINOSSLXVectorTrainerFromWav(DINOSSLXVectorTrainer):
    """Trainer to train x-vector style models.

    Attributes:
      model: x-Vector model object.
      feat_extractor: feature extractor nn.Module
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object or options dict.
      loggers: LoggerList object, loggers write training progress to std. output and file.
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
        feat_extractor,
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
        niter_per_ep=0,
        batch_size=0
    ):

        super().__init__(
            model,
            optim,
            epochs,
            exp_path,
            cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps,
            eff_batch_size=eff_batch_size,
            device=device,
            metrics=metrics,
            lrsched=lrsched,
            loggers=loggers,
            ddp=ddp,
            ddp_type=ddp_type,
            loss=loss,
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
            niter_per_ep=niter_per_ep, 
            batch_size=batch_size
        )

        self.feat_extractor = feat_extractor
        if device is not None:
            self.feat_extractor.to(device)

        # if ddp:
        #     self.feat_extractor = TorchDDP(self.feat_extractor)

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """

        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.feat_extractor.train()
        self.model.train()
        for batch, (data, _) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            data = [i.to(self.device, non_blocking=True) for i in data]
            batch_size = data[0].shape[0]
            with torch.no_grad():
                feats = []
                for i in data:
                    feats.append(self.feat_extractor(i))

            with self.amp_autocast():
                output = self.model(feats)
                output_teacher = self.model_teacher(feats[:2]) # 2 (currently this number is fixed) global crops
                loss = self.loss(output, output_teacher, self.cur_epoch)/self.grad_acc_steps

            if not math.isfinite(loss.item()):
                logging.warning('Loss is {}, stopping training'.format(loss.item()))
                sys.exit(1)

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            freeze_last_layer=1
            cancel_gradients_last_layer(self.cur_epoch, self.model, freeze_last_layer)

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                # update learning rate and weight decay rate
                it = len(data_loader) * self.cur_epoch + int(batch/self.grad_acc_steps) # it: global batch index, batch: local batch index in the current epoch
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = self.lr_schedule[it]
                    if i == 0:  # only the first group is regularized
                        param_group["weight_decay"] = self.wd_schedule[it]
                self.update_model()

                # EMA update for the teacher
                with torch.no_grad():
                    m = self.momentum_schedule[it]  # momentum parameter
                    if hasattr(self.model,'module'): # train with ddp
                        for param_q, param_k in zip(self.model.module.parameters(), self.model_teacher_without_ddp.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    else: # train with a single gpu (w/o ddp), which I (JJ) used in debugging
                        for param_q, param_k in zip(self.model.parameters(), self.model_teacher_without_ddp.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            # (JJ: TODO - for now, I skipped the logging related parts (per iter and epoch) after the above from dino)

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr() # (JJ: TODO - this may need to change later (NOT now) if lrs are applied differerently over parameter groups)
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
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.feat_extractor.eval()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.model.train()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                feats = self.feat_extractor(data)
                with self.amp_autocast():
                    output = self.model(feats)
                    loss = self.loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs
