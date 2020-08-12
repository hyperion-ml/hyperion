"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging

import torch
import torch.nn as nn
from apex import amp

from ..utils import MetricAcc, TorchDataParallel
from .xvector_trainer import XVectorTrainer


class XVectorTrainerFromWav(XVectorTrainer):
    """Trainer to train x-vector style models.

       Attributes:
         model: x-Vector model object.
         feat_extractor: feature extractor nn.Module
         optimizer: pytorch optimizer object
         epochs: max. number of epochs
         exp_path: experiment output path
         cur_epoch: current epoch
         grad_acc_steps: gradient accumulation steps to simulate larger batch size.
         device: cpu/gpu device
         metrics: extra metrics to compute besides cxe.
         lr_scheduler: learning rate scheduler object
         loggers: LoggerList object, loggers write training progress to std. output and file.
         data_parallel: if True use nn.DataParallel
         loss: if None, it uses cross-entropy
         train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
         use_amp: uses mixed precision training.
         log_interval: 
    """
    def __init__(self, model, feat_extractor, optimizer, epochs, exp_path, cur_epoch=0, 
                 grad_acc_steps=1, 
                 device=None, metrics=None, lr_scheduler=None, loggers=None, 
                 data_parallel=False, loss=None, train_mode='train', use_amp=False,
                 log_interval=10):

        super().__init__(
            model, optimizer, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel, loss=loss,
            train_mode=train_mode, use_amp=use_amp)

        self.feat_extractor = feat_extractor
        if device is not None:
            self.feat_extractor.to(device)

        if data_parallel:
            self.feat_extractor = TorchDataParallel(self.feat_extractor)

        # super().__init__(
        #     model, optimizer, epochs, exp_path, cur_epoch=cur_epoch,
        #     grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
        #     lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=False, loss=loss,
        #     train_mode=train_mode, use_amp=False)

        # self.use_amp = use_amp

        # self.feat_extractor = feat_extractor
        # if device is not None:
        #     self.feat_extractor.to(device)

        # if data_parallel:
        #     self.feat_extractor = TorchDataParallel(self.feat_extractor)


        # if self.use_amp:
        #     logging.info('using automatic mixed precision training')
        #     [self.model, self.feat_extractor], self.optimizer  = amp.initialize(
        #         [self.model, self.feat_extractor], self.optimizer, opt_level="O1")

        # if data_parallel:
        #     logging.info('training in multiple gpus with data-parallel')
        #     self.model = TorchDataParallel(self.model)
        #     self.feat_extractor = TorchDataParallel(self.feat_extractor)
        #     self.loss = TorchDataParallel(self.loss)


        
    def train_epoch(self, data_loader):
        """Training epoch loop

           Args:
             data_loader: pytorch data loader returning features and class labels.
        """

        self.model.update_loss_margin(self.cur_epoch)

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

            with torch.no_grad():
                feats = self.feat_extractor(data)
            #logging.info('feats={}'.format(feats))

            output = self.model(feats, target)
            loss = self.loss(output, target).mean()/self.grad_acc_steps
            # logging.info('loss={} output={}'.format(loss.item(), output))
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

                feats = self.feat_extractor(data)
                output = self.model(feats)
                loss = self.loss(output, target)
                batch_metrics['loss'] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)
            
                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict(('val_' + k, v) for k,v in logs.items())
        return logs

