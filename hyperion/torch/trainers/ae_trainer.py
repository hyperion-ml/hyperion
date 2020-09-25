"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
from collections import OrderedDict as ODict

import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc
from .torch_trainer import TorchTrainer

class AETrainer(TorchTrainer):
    """Auto-encoder trainer class

       Attributes:
         model: x-Vector model object.
         optimizer: pytorch optimizer object
         loss: nn.Module loss class
         epochs: max. number of epochs
         exp_path: experiment output path
         cur_epoch: current epoch
         grad_acc_steps: gradient accumulation steps to simulate larger batch size.
         device: cpu/gpu device
         metrics: extra metrics to compute besides cxe.
         lr_scheduler: learning rate scheduler object
         loggers: LoggerList object, loggers write training progress to std. output and file.
         data_parallel: if True use nn.DataParallel
         train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
         use_amp: uses mixed precision training.
         log_interval: number of optim. steps between log outputs
         grad_clip: norm to clip gradients, if 0 there is no clipping
         swa_start: epoch to start doing swa
         swa_lr: SWA learning rate
         swa_anneal_epochs: SWA learning rate anneal epochs
    """

    def __init__(self, model, optimizer, loss=None, epochs=100, 
                 exp_path='./train', cur_epoch=0, grad_acc_steps=1,
                 device=None, metrics=None, lr_scheduler=None, 
                 loggers=None, data_parallel=False, 
                 train_mode='train', use_amp=False, log_interval=10,
                 grad_clip=0, swa_start=0, swa_lr=1e-3, swa_anneal_epochs=10):
        if loss is None:
            loss = nn.MSELoss()
            
        super().__init__(
            model, optimizer, loss, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel, 
            train_mode=train_mode, use_amp=use_amp, log_interval=log_interval,
            grad_clip=grad_clip,                  
            swa_start=swa_start, swa_lr=swa_lr, 
            swa_anneal_epochs=swa_anneal_epochs)
            
            
    def train_epoch(self, data_loader):
        """Training epoch loop

           Args:
             data_loader: pytorch data loader returning features and class labels.
        """

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        self.set_train_mode()
        for batch, data in enumerate(data_loader):

            if isinstance(data, (tuple, list)):
                data, _ = data

            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            data = data.to(self.device)
            batch_size = data.shape[0]

            with self.amp_autocast():
                output = self.model(data, **self.amp_args)
                loss = self.loss(output, data).mean()/self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics['loss'] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, data)
            
            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs['lr'] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)
            #total_batches += 1

        logs = metric_acc.metrics
        logs['lr'] = self._get_lr()
        return logs


    def validation_epoch(self, data_loader, swa_update_bn=False):

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = ''
                self.set_train_mode()
            else:
                log_tag = 'val_'
                self.model.eval()

            for batch, data in enumerate(data_loader):
                if isinstance(data, (tuple, list)):
                    data, _ = data

                data = data.to(self.device)
                batch_size = data.shape[0]
                with self.amp_autocast():
                    output = self.model(data, **self.amp_args)
                    loss = self.loss(output, data)

                batch_metrics['loss'] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, data)
            
                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k,v in logs.items())
        return logs


