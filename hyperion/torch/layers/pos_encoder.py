"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import math

import torch
from torch import nn


class PosEncoder(nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, num_feats, dropout_rate=0, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PosEncoder, self).__init__()
        self.num_feats = num_feats
        self.dropout_rate = dropout_rate
        self.xscale = math.sqrt(self.num_feats)
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        s = '{}(num_feats={}, dropout_rate={})'.format(
            self.__class__.__name__, self.num_feats, self.dropout_rate)
        return s


    def _pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return self.pe

        pe = torch.zeros(x.size(1), self.num_feats)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.num_feats, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / self.num_feats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        return self.pe


    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        pe = self._pe(x)
        x = x * self.xscale + pe[:, :x.size(1)]
        if self.dropout_rate > 0:
            return self.dropout(x)
        return x
