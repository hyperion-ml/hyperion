"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn


class MeanVarianceNorm(nn.Module):

    def __init__(self, norm_mean=True, norm_var=False, left_context=0, right_context=0, dim=1):

        super(MeanVarianceNorm, self).__init__()
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.left_context = left_context
        self.right_context = right_context
        self.dim = dim


    def forward(self, x):
        
        T = x.shape[self.dim]
        if (self.left_context==0 and self.right_context==0) or (
                T <= self.left_context + self.right_context + 1):
            return self.normalize_global(x)

        return self.normalize_cumsum(x)


    def normalize_global(self, x):
        # Global mean/var norm.
        if self.norm_mean:
            m_x = torch.mean(x, dim=self.dim, keepdim=True)
            x = x - m_x

        if self.norm_var:
            s_x = torch.std(x, dim=self.dim, keepdim=True).clamp(min=1e-5)
            x = x/s_x

        return x

    def normalize_cumsum(self, x):

        if self.norm_mean:
            #substract first global mean
            #it will help cumsum numerical stability
            m_x = torch.mean(x, dim=self.dim, keepdim=True)
            x = x - m_x

        if self.dim != 1:
            x = x.transpose(self.dim, 1)

        total_context = self.left_context + self.right_context + 1

        xx = nn.functional.pad(x.transpose(1,-1), (self.left_context, self.right_context), 
                               mode='reflect').transpose(1, -1)

        if self.norm_mean:
            c_x = torch.cumsum(xx, dim=1)
            m_x = (c_x[:,total_context-1:] - c_x[:,:-total_context+1])/total_context

        if self.norm_var:
            c_x = torch.cumsum(xx**2, dim=1)
            m_x2 = (c_x[:,total_context-1:] - c_x[:,:-total_context+1])/total_context

        if self.norm_mean:
            x = x - m_x

        if self.norm_var:
            s_x = torch.sqrt(m_x2 - m_x**2).clamp(min=1e-5)
            x = x/s_x

        if self.dim != 1:
            x = x.transpose(self.dim, 1)

        return x.contiguous()

        
        
