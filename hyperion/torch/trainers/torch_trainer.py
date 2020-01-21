"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import os
from collections import OrderedDict as ODict

import logging
#import numpy as np

import torch
import torch.nn as nn
from apex import amp

from ..utils import MetricAcc
from ..loggers import LoggerList, CSVLogger, ProgLogger


class TorchDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(TorchDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



class TorchTrainer(object):

    def __init__(self, model, optimizer, loss, epochs, exp_path, cur_epoch=0, grad_acc_steps=1,
                 device=None, metrics=None, lr_scheduler=None, loggers=None, data_parallel=False, 
                 train_mode='train', use_amp=False):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.cur_epoch = cur_epoch
        self.grad_acc_steps = grad_acc_steps

        self.exp_path = exp_path
        
        if loggers is None:
            self.loggers = self._default_loggers()
        elif isinstance(loggers, list):
            self.loggers = LoggerList(loggers)
        else:
            self.loggers = loggers

        self.lr_scheduler = lr_scheduler
        
        self.metrics = metrics
        self.device = device
        self.train_mode = train_mode
        self.use_amp = use_amp

        if device is not None:
            self.model.to(device)
            self.loss.to(device)

        if self.use_amp:
            logging.info('using automatic mixed precision training')
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O1")

        if data_parallel:
            logging.info('training in multiple gpus with data-parallel')
            self.model = TorchDataParallel(self.model)
            self.loss = TorchDataParallel(self.loss)



        
    def fit(self, train_data, val_data=None):

        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        
        val_logs = {}
        self.loggers.on_train_begin(epochs=self.epochs)
        for epoch in range(self.cur_epoch, self.epochs):
            
            self.loggers.on_epoch_begin(epoch, batches=len(train_data))
            if self.lr_scheduler is not None:
                # this is needed by cosine scheduler
                epoch_updates = int(len(train_data)/self.grad_acc_steps)
                self.lr_scheduler.on_epoch_begin(epoch, epoch_updates=epoch_updates)
            
            logs = self.train_epoch(train_data)
            if val_data is not None:
                val_logs = self.validation_epoch(val_data)
                logs.update(val_logs)

            self.cur_epoch +=1
            
            self.loggers.on_epoch_end(logs)
            if self.lr_scheduler is not None:
                self.lr_scheduler.on_epoch_end(logs)

            self.save_checkpoint(logs)

            
            
    def train_epoch(self, data_loader):

        #epoch_batches = len(data_loader.dataset)
        #total_batches = self.cur_epoch * epoch_batches

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        if self.train_mode == 'train':
            self.model.train()
        else:
            self.model.train_mode(self.train_mode)

        for batch, (data, target) in enumerate(data_loader):
            
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]
            
            output = self.model(data)
            loss = self.loss(output, target).mean()/self.grad_acc_steps

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
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)
            
            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs['lr'] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)
            #total_batches += 1

        logs = metric_acc.metrics
        logs['lr'] = self._get_lr()
        return logs


    def validation_epoch(self, data_loader):

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        with torch.no_grad():
            self.model.eval()
            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                output = self.model(data)
                loss = self.loss(output, target)
                batch_metrics['loss'] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)
            
                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict(('val_' + k, v) for k,v in logs.items())
        return logs


    def _default_loggers(self):
        prog_log = ProgLogger(interval=10)
        csv_log = CSVLogger(self.exp_path + '/train.log', append=True)
        return LoggerList([prog_log, csv_log])
    
    
    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def checkpoint(self, logs=None):
        checkpoint = {
            'epoch': self.cur_epoch,
            'rng_state': torch.get_rng_state(),
            'model_cfg': self.model.get_config(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss.state_dict()
            }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.use_amp:
            checkpoint['amp'] = amp.state_dict()

        if logs is not None:
            checkpoint['logs'] = logs
            
        return checkpoint
    
        
    def save_checkpoint(self, logs=None):

        checkpoint = self.checkpoint(logs)
        file_path = '%s/model_ep%04d.pth' % (self.exp_path, self.cur_epoch)
            
        torch.save(checkpoint, file_path)


    def load_checkpoint(self, file_path):

        #checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss.load_state_dict(checkpoint['loss_state_dict'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if self.use_amp:
            amp.load_state_dict(checkpoint['amp'])

        logs = None
        if 'logs' in checkpoint:
            logs = checkpoint['logs']

        del checkpoint 
        if self.device is not None:
            torch.cuda.empty_cache()

        return logs

    
    def load_last_checkpoint(self):

        for epoch in range(self.epochs, 0, -1):
            file_path = '%s/model_ep%04d.pth' % (self.exp_path, epoch)
            if os.path.isfile(file_path):
                return self.load_checkpoint(file_path)

        return None

