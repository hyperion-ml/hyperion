"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import time
import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc
from .xvector_trainer import XVectorTrainer


class XVectorFinetuner(XVectorTrainer):
    def __init__(
        self,
        model,
        optimizer,
        epochs,
        exp_path,
        cur_epoch=0,
        grad_acc_steps=1,
        device=None,
        metrics=None,
        lr_scheduler=None,
        loggers=None,
        data_parallel=False,
        loss=None,
        finetune_mode="ft-embed-affine",
    ):

        super(XVectorFinetuner, self).__init__(
            model,
            optimizer,
            epochs,
            exp_path,
            cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps,
            device=device,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            loggers=loggers,
            data_parallel=data_parallel,
            loss=loss,
        )

        self.finetune_mode = finetune_mode

    def train_epoch(self, data_loader):
        # epoch_batches = len(data_loader.dataset)
        # total_batches = self.cur_epoch * epoch_batches

        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        # self.model.train_mode(self.finetune_mode)
        self.model.eval()
        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            output = self.model(data, target)
            loss = self.loss(output, target).mean() / self.grad_acc_steps
            loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_opt_step()
                self.optimizer.step()

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            # logging.info('batch={} shape={} loss={} acc={}'.format(batch,data.shape, batch_metrics['loss'], batch_metrics['acc']))

            # if batch > 63:
            #     logging.info(str(self.model.classif_net.fc_blocks[0].linear.weight))
            #     logging.info(str(self.model.classif_net.fc_blocks[0].linear.weight.grad))
            # if batch > 63 :
            #     t=torch.nn.functional.cross_entropy(output, target, reduction='none')
            #     logging.info(str(t))
            #     if batch == 65:
            #         #torch.set_printoptions(profile="full")
            #         #logging.info(str(data[1]))
            #         #logging.info(str(target[1]))
            #         #logging.info(str(output[1]))

            #         #logging.info(str(data[33]))
            #         #logging.info(str(target[33]))
            #         logging.info(str(output[33, target[33]]))
            #         #time.sleep(1000)
            #         #torch.set_printoptions(profile="default")

            #     #logging.info(str(torch.sum(torch.isnan(data))))
            #     #logging.info(str(torch.sum(torch.isnan(target))))
            #     #logging.info(str(torch.sum(torch.isnan(output))))

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)
            # total_batches +=1

        logs = metric_acc.metrics
        logs["lr"] = self._get_lr()
        return logs
