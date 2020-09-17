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
            
            # elbo, log_px, kldiv_z, x_hat = self.model(
            #     data, return_x_mean=True)
            # elbo = elbo.mean()
            # loss = - elbo/self.grad_acc_steps

            output = self.model(
                data, return_x_mean=True)
            elbo = output['elbo'].mean()
            loss = - elbo/self.grad_acc_steps
            x_hat = output['x_mean']

            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_opt_step()
                self.optimizer.step()

            batch_metrics['elbo'] = elbo.item()
            for metric in ['log_px', 'kldiv_z']:
                batch_metrics[metric] = output[metric].mean().item()
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

                #elbo, log_px, kldiv_z, x_hat = self.model(
                #    data, return_x_mean=True)
                output = self.model(
                    data, return_x_mean=True)
                x_hat = output['x_mean']
                for metric in ['elbo', 'log_px', 'kldiv_z']:
                    batch_metrics[metric] = output[metric].mean().item()

                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(x_hat, data)
            
                metric_acc.update(batch_metrics, batch_size)
        
        logs = metric_acc.metrics
        logs = ODict(('val_' + k, v) for k,v in logs.items())
        return logs


