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

    def __init__(self, model, optimizer, loss, epochs, cur_epoch=0, device=None, metrics=None, lr_scheduler=None, loggers=None):
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

        for epoch in xrange(self.cur_epoch, self.epochs):
            self.train_epoch(train_data)
            if val_data is not None:
                self.validation_epoch(val_data)
            self.cur_epoch +=1


            
    def train_epoch(self, data_loader):

        epoch_steps = len(data_loader.dataset)
        total_steps = self.cur_epoch * epoch_steps

        loss_accum = 0
        num_steps=0
        self.model.train()
        for step, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            loss_accum += loss.item()
            #print something
            num_steps += 1
            total_steps +=1
            print(loss_accum/num_steps)

        loss = loss_accum/num_steps
        print(loss)
        return loss


    def validation_epoch(self, data_loader):
            
        with torch.no_grad():
            self.model.eval()
            num_steps = 0
            loss_accum = 0
            for step, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)
                loss_accum += loss.item()
                num_steps +=1

        loss = loss_accum/num_steps
        print(loss)
        return loss
