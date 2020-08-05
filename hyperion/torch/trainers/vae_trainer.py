"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

import os
from collections import OrderedDict as ODict

import logging
#import numpy as np

import torch
import torch.nn as nn
from apex import amp

from ..utils import MetricAcc
from .torch_trainer import TorchTrainer

class VAETrainer(TorchTrainer):

    def __init__(self, model, optimizer, epochs, exp_path, cur_epoch=0, grad_acc_steps=1,
                 device=None, metrics=None, lr_scheduler=None, loggers=None, data_parallel=False, 
                 train_mode='train', use_amp=False, log_interval=10):
            
        super().__init__(
            model, optimizer, None, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel, 
            train_mode=train_mode, use_amp=use_amp, log_interval=log_interval)


            
            
    def train_epoch(self, data_loader):

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        if self.train_mode == 'train':
            self.model.train()
        else:
            self.model.train_mode(self.train_mode)

        for batch, data in enumerate(data_loader):

            if isinstance(data, (tuple, list)):
                data, _ = data

            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            data = data.to(self.device)
            batch_size = data.shape[0]
            
            elbo, log_px, kldiv_z, px, qz = self.model(data)
            loss = - elbo.mean()/self.grad_acc_steps
            x_hat = px.mean
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_opt_step()
                self.optimizer.step()

            batch_metrics['elbo'] = - loss.item() * self.grad_acc_steps
            batch_metrics['log_px'] = log_px.mean().item()
            batch_metrics['kldiv_z'] = kldiv_z.mean().item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(x_hat, data)
            
            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs['lr'] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)


        logs = metric_acc.metrics
        logs['lr'] = self._get_lr()
        return logs


    def validation_epoch(self, data_loader):

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        with torch.no_grad():
            self.model.eval()
            for batch, data in enumerate(data_loader):
                if isinstance(data, (tuple, list)):
                    data, _ = data

                data = data.to(self.device)
                batch_size = data.shape[0]

                elbo, log_px, kldiv_z, px, _ = self.model(data)
                x_hat = px.mean
                batch_metrics['elbo'] = elbo.mean().item() 
                batch_metrics['log_px'] = log_px.mean().item()
                batch_metrics['kldiv_z'] = kldiv_z.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(x_hat, data)
            
                metric_acc.update(batch_metrics, batch_size)
        
        logs = metric_acc.metrics
        logs = ODict(('val_' + k, v) for k,v in logs.items())
        return logs


