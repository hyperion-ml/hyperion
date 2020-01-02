"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging
#import numpy as np

import torch
import torch.nn as nn

from ..utils import MetricAcc
from .torch_trainer import TorchTrainer


class XVectorTrainer(TorchTrainer):

    def __init__(self, model, optimizer, epochs, exp_path, cur_epoch=0, 
                 grad_acc_steps=1, 
                 device=None, metrics=None, lr_scheduler=None, loggers=None, 
                 data_parallel=False, loss=None):

        if loss is None:
            loss = nn.CrossEntropyLoss()
        super(XVectorTrainer, self).__init__(
            model, optimizer, loss, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel)


        
    def train_epoch(self, data_loader):
        #epoch_batches = len(data_loader.dataset)
        #total_batches = self.cur_epoch * epoch_batches
        
        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        self.model.train()
        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            output = self.model(data, target)
            loss = self.loss(output, target).mean()/self.grad_acc_steps
            loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_opt_step()
                self.optimizer.step()

            batch_metrics['loss'] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)
            
            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs['lr'] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)
            #total_batches +=1

        logs = metric_acc.metrics
        logs['lr'] = self._get_lr()
        return logs

                                             


