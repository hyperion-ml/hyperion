"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv1d

from ..layers import ActivationFactory as AF

import logging

class SEBlock2D(nn.Module):
    """ From https://arxiv.org/abs/1709.01507
    """
    def __init__(self, num_channels, r=16, activation={'name':'relu', 'inplace': True}):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, int(num_channels/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(int(num_channels/r), num_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        z = torch.mean(x, dim=(2,3), keepdim=True)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        y = scale * x
        return y



class TSEBlock2D(nn.Module):
    """ From https://arxiv.org/abs/1709.01507
        Modified to do pooling only in time dimension
    """
    def __init__(self, num_channels, num_feats, r=16, activation={'name':'relu', 'inplace': True}):
        super().__init__()
        self.num_channels_1d = num_channels*num_feats
        self.conv1 = nn.Conv2d(self.num_channels_1d, int(self.num_channels_1d/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(int(self.num_channels_1d/r), self.num_channels_1d, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        num_feats = x.shape[2]
        num_channels = x.shape[1]
        z = torch.mean(x, dim=-1, keepdim=True)
        z = z.view(-1, self.num_channels_1d, 1, 1)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        scale = scale.view(-1, num_channels, num_feats, 1)
        y = scale * x
        return y


class SEStdBlock2D(nn.Module):
    """ From https://arxiv.org/abs/1709.01507
    """
    def __init__(self, num_channels, r=16, activation={'name':'relu', 'inplace': True}):
        super().__init__()
        self.conv1 = nn.Conv2d(2*num_channels, int(2*num_channels/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(int(2*num_channels/r), num_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # z = torch.mean(x, dim=(2,3), keepdim=True)
        mu = torch.mean(x, dim=(2,3), keepdim=True)
        delta = x - mu
        # logging.info(mu.shape)
        # mu.squeeze_()
        # logging.info(mu.shape)
        # this can produce slightly negative variance when relu6 saturates in all time steps
        # add 1e-5 for stability
        s = torch.sqrt(
            torch.mean(delta ** 2, dim=(2,3), keepdim=True) + 1e-5)

        mus = torch.cat((mu, s), dim=1)
        # logging.info(mus.shape)
            # mus.unsqueeze(dim=self.dim)

        scale = self.sigmoid(self.conv2(self.act(self.conv1(mus))))
        y = scale * x
        return y

class SEBlock1d(nn.Module):
    """ 1d Squeeze Excitation version of 
        https://arxiv.org/abs/1709.01507
    """
    def __init__(self, num_channels, r=16, activation={'name':'relu', 'inplace': True}):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, int(num_channels/r), kernel_size=1, bias=False)
        self.act = AF.create(activation)
        self.conv2 = nn.Conv1d(int(num_channels/r), num_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        z = torch.mean(x, dim=2, keepdim=True)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        y = scale * x
        return y


# aliases to mantein backwards compatibility
SEBlock2d = SEBlock2D
TSEBlock2d = TSEBlock2D
