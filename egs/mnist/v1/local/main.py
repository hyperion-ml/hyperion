#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
 Trains x-vectors
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from hyperion.hyp_defs import config_logger
from hyperion.torch.torch_utils import open_gpu
from hyperion.torch.archs import FFNetV1
from hyperion.torch.transforms import Reshape
from hyperion.torch.torch_trainer import TorchTrainer

input_width=28
input_height=28

def create_net(net_type):
    if net_type=='ffnet':
        return FFNetV1(3, 10, 1000, input_width*input_height, dropout_rate=0.5)


    
def main(net_type, batch_size, test_batch_size,
         lr, momentum, epochs, use_cuda, log_interval):

    if use_cuda:
        device = open_gpu()
    else:
        device = torch.device('cpu')

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))]
    if net_type == 'ffnet':
        transform_list.append(Reshape((-1,)))
    transform = transforms.Compose(transform_list)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./exp/data', train=True, download=True,
                       transform=transform), 
                       batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./exp/data', train=False, transform=transform),
                       batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = create_net(net_type)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = nn.CrossEntropyLoss()
    
    trainer = TorchTrainer(model, optimizer, loss, epochs, device=device)
    trainer.fit(train_loader, test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='PyTorch MNIST')
    parser.add_argument('--net-type', default='ffnet', metavar='N',
                        help='input batch size for training (def')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, 
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='how many batches to wait before logging training status')

    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    
    torch.manual_seed(args.seed)
    del args.seed

    main(**vars(args))



