"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..layers import ActivationFactory as AF

class FCBlock(nn.Module):

    def __init__(self, in_feats, out_feats, 
                 activation={'name':'relu', 'inplace': True},
                 dropout_rate=0,
                 use_batchnorm=True, batchnorm_before=False):

        super(FCBlock, self).__init__()

        self.activation = AF.create(activation)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

        self.batchnorm_before = False
        self.batchnorm_after = False
        if use_batchnorm:
            self.bn1 = BatchNorm1d(out_feats)        
            if batchnorm_before:
                self.batchnorm_before = True
            else:
                self.batchnorm_after = True

        self.linear = Linear(in_feats, out_feats, bias=(not self.batchnorm_before)) 



    def forward(self, x):

        x = self.linear(x)

        if self.batchnorm_before:
            x = self.bn1(x)

        x = self.activation(x)
        
        if self.batchnorm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


    def forward_linear(self, x):

        x = self.linear(x)

        if self.batchnorm_before:
            x = self.bn1(x)

        return x




