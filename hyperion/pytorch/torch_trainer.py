"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import torch
import torch.nn as nn


class TorchTrainer(object):

    def __init__(self, model, optimizer, loss, epochs, cur_epoch=0, device=None, callbacks=None, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.cur_epoch = cur_epoch
        self.callbacks = callbacks
        self.metrics = metrics
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        
    def fit(self, train_data, val_data=None):

        for epoch in xrange(cur_epoch, epochs):
            self.train_epoch(train_data)
            if val_data is not None:
                self.validation_epoch(val_data)
            self.cur_epoch +=1


            
    def train_epoch(self, data_loader):

        epoch_steps = len(data_loader.dataset)
        total_steps = self.cur_epoch * epoch_steps

        loss_acc = 0
        self.model.train()
        for step, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            optimizer.step()
            loss_acc += loss.item()
            #print something
            total_steps +=1

        loss = loss_acc/num_steps
        return loss


    def validation_epoch(self, data_loader):
            
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for step, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)
                loss_acc += loss.item()
                #print something
                total_steps +=1

        loss = loss_acc/num_steps
        return loss
