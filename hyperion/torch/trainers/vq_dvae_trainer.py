"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging
import math

import torch
import torch.nn as nn
from apex import amp

from ..utils import MetricAcc
from .dvae_trainer import DVAETrainer

class VQDVAETrainer(DVAETrainer):

    def __init__(self, model, optimizer, epochs, exp_path, cur_epoch=0, grad_acc_steps=1,
                 device=None, metrics=None, lr_scheduler=None, loggers=None, data_parallel=False, 
                 train_mode='train', use_amp=False, log_interval=10):
            
        super().__init__(
            model, optimizer, epochs, exp_path, cur_epoch=cur_epoch,
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

            assert isinstance(data, (tuple, list))
            x = data[0]
            x_target = data[1]

            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            x = x.to(self.device)
            x_target = x_target.to(self.device)
            batch_size = x.shape[0]
            
            # loss, elbo, log_px, kldiv_z, vq_loss, perplexity, x_hat = self.model(
            #     x, x_target=x_target, return_x_mean=True)
            output = self.model(x, x_target=x_target, return_x_mean=True)

            loss = output['loss']
            x_hat = output['x_mean']

            loss = loss.mean()/self.grad_acc_steps

            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_opt_step()
                self.optimizer.step()

            batch_metrics['loss'] = loss.item() * self.grad_acc_steps
            for metric in ['elbo', 'log_px', 'kldiv_z', 'vq_loss']:
                batch_metrics[metric] = output[metric].mean().item()
            batch_metrics['perplexity'] = math.exp(output['log_perplexity'].mean().item())
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(x_hat, x_target)
            
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

                assert isinstance(data, (tuple, list))
                x = data[0]
                x_target = data[1]

                x = x.to(self.device)
                x_target = x_target.to(self.device)
                batch_size = x.shape[0]

                # loss, elbo, log_px, kldiv_z, vq_loss, perplexity, x_hat = self.model(
                #     x, x_target=x_target, return_x_mean=True)
                output = self.model(
                    x, x_target=x_target, return_x_mean=True)

                x_hat = output['x_mean']
                for metric in ['loss', 'elbo', 'log_px', 'kldiv_z', 'vq_loss']:
                    batch_metrics[metric] = output[metric].mean().item()
                batch_metrics['perplexity'] = math.exp(output['log_perplexity'].mean().item())
        
                # batch_metrics['loss'] = loss.mean().item()
                # batch_metrics['elbo'] = elbo.mean().item()
                # batch_metrics['log_px'] = log_px.mean().item()
                # batch_metrics['kldiv_z'] = kldiv_z.mean().item()
                # batch_metrics['vq_loss'] = vq_loss.mean().item()
                # batch_metrics['perplexity'] = perplexity.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(x_hat, x_target)
            
                metric_acc.update(batch_metrics, batch_size)
        
        logs = metric_acc.metrics
        logs = ODict(('val_' + k, v) for k,v in logs.items())
        return logs


