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
                 use_norm=True, norm_before=False):

        super(FCBlock, self).__init__()

        self.activation = AF.create(activation)

        self.dropout_rate = dropout_rate
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

        self.norm_before = False
        self.norm_after = False
        if use_norm:
            self.bn1 = BatchNorm1d(out_feats)        
            if norm_before:
                self.norm_before = True
            else:
                self.norm_after = True

        self.linear = Linear(in_feats, out_feats, bias=(not self.norm_before)) 



    def forward(self, x):

        x = self.linear(x)

        if self.norm_before:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)
        
        if self.norm_after:
            x = self.bn1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


    def forward_linear(self, x):

        x = self.linear(x)

        if self.norm_before:
            x = self.bn1(x)

        return x




