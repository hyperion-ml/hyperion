"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.distributions as pdf


class StdNormal(nn.Module):
    """Storage for Standard Normal distribution
    """
    def __init__(self, shape):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(shape), requires_grad=False)
        #self.pdf = pdf.normal.Normal(self.loc, self.scale)


    def forward(self):
        return pdf.normal.Normal(self.loc, self.scale)
        #return self.pdf
