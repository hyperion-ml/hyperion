"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
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

from ..utils import MetricAcc, TorchDataParallel
from ..loggers import LoggerList, CSVLogger, ProgLogger



class TorchTrainer(object):
    """Base Trainer class to train basic neural network models

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
    """
    def __init__(self, model, optimizer, loss, epochs, exp_path, cur_epoch=0, grad_acc_steps=1,
                 device=None, metrics=None, lr_scheduler=None, loggers=None, data_parallel=False, 
                 train_mode='train', use_amp=False, log_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.cur_epoch = cur_epoch
        self.grad_acc_steps = grad_acc_steps

        self.exp_path = exp_path
        
        if loggers is None:
            self.loggers = self._default_loggers(log_interval)
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
            if loss is not None:
                self.loss.to(device)

        if self.use_amp:
            logging.info('using automatic mixed precision training')
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O1")

        if data_parallel:
            logging.info('training in multiple gpus with data-parallel')
            self.model = TorchDataParallel(self.model)
            if loss is not None:
                self.loss = TorchDataParallel(self.loss)



        
    def fit(self, train_data, val_data=None):
        """Training function, it performs the training and validation epochs
         
        Args:
          train_data: PyTorch data loader for the training loop
          val_data: PyTorch data loader for the validation loop
        """
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
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """
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
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """

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


    def _default_loggers(self, log_interval):
        """Creates the default data loaders
        """
        prog_log = ProgLogger(interval=log_interval)
        csv_log = CSVLogger(self.exp_path + '/train.log', append=True)
        return LoggerList([prog_log, csv_log])
    
    
    def _get_lr(self):
        """Returns the current learning rate to show in the loggers
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def checkpoint(self, logs=None):
        """Creates a checkpoint of the training, to save and posterior recovery

        Args:
          logs: logs containing the current value of the metrics.
        """
        checkpoint = {
            'epoch': self.cur_epoch,
            'rng_state': torch.get_rng_state(),
            'model_cfg': self.model.get_config(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss.state_dict() if self.loss is not None else None
            }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.use_amp:
            checkpoint['amp'] = amp.state_dict()

        if logs is not None:
            checkpoint['logs'] = logs
            
        return checkpoint
    
        
    def save_checkpoint(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
        """

        checkpoint = self.checkpoint(logs)
        file_path = '%s/model_ep%04d.pth' % (self.exp_path, self.cur_epoch)
            
        torch.save(checkpoint, file_path)


    def load_checkpoint(self, file_path):
        """Loads a training checkpoint from file.

        Args:
           file_path: checkpoint file path
        """
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
        self.cur_epoch = checkpoint['epoch']
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.loss is not None:
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
        """Loads the last training checkpoint in the experiment dir.
        """
        for epoch in range(self.epochs, 0, -1):
            file_path = '%s/model_ep%04d.pth' % (self.exp_path, epoch)
            if os.path.isfile(file_path):
                return self.load_checkpoint(file_path)

        return None

