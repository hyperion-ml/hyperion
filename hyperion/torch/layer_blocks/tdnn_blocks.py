"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np

import torch.nn as nn
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..layers import ActivationFactory as AF
from ..layers import Dropout1d

class TDNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 kernel_size, dilation=1, 
                 activation={'name':'relu', 'inplace': True},
                 dropout_rate=0,
                 use_norm=True, norm_before=False):

        super(TDNNBlock, self).__init__()

        self.activation = AF.create(activation)
        padding = int(dilation * (kernel_size -1)/2)

        self.dropout_rate =dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout1d(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            self.bn1 = BatchNorm1d(out_channels)        
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.conv1 = Conv1d(in_channels, out_channels, 
                            bias=(not self.norm_before),
                            kernel_size=kernel_size, dilation=dilation, 
                            padding=padding) # padding_mode='reflection') pytorch > 1.0


    def forward(self, x):

        x = self.conv1(x)

        if self.norm_before:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)
        
        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x




