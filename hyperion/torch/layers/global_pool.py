"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.functional as F


class _GlobalPool1d(nn.Module):

    def __init__(self, dim=-1, keepdim=False, batch_dim=0):
        super(_GlobalPool1d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.batch_dim = batch_dim
        self.size_multiplier = 1

    def _standarize_weights(self, weigths, ndims):

        if weights.dims() == ndims:
            return weights

        assert weights.dims() == 2
        shape = ndims * [1]
        shape[self.batch_dim] = weights.shape[0]
        shape[self.dim] = weights.shape[1]
        return weights.view(tuple(shape))

    def get_config(self):
        
        config = { 'dim': self.dim,
                   'keepdim': self.keepdim,
                   'batch_dim': self.batch_dim}
        return config


class GlobalAvgPool1d(_GlobalPool1d):

    def __init__(self, dim=-1, keepdim=False, batch_dim=0):
        super(GlobalAvgPool1d, self).__init__(dim, keepdim, batch_dim)
        

    def forward(self, x, weights=None):
        if weights is None:
            y = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            return y

        weights = self._standarize_weights(weights, x.dim())

        xbar = torch.mean(weights*x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        return xbar/wbar



class GlobalMeanStdPool1d(_GlobalPool1d):

    def __init__(self, dim=-1, keepdim=False, batch_dim=0):
        super(GlobalMeanStdPool1d, self).__init__(dim, keepdim, batch_dim)
        self.size_multiplier = 2

    def forward(self, x, weights=None):
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            x2bar = torch.mean(x**2, dim=self.dim, keepdim=self.keepdim)
            s = torch.sqrt(x2bar - mu*mu+1e-5) #for stability in case var=0
            return torch.cat((mu,s), dim=-1)

        weights = self._standarize_weights(weights, x.dim())

        xbar = torch.mean(weights*x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        mu = xbar/wbar
        x2bar = torch.mean(weights*x**2, dim=self.dim, keepdim=self.keepdim)/wbar
        s = torch.sqrt(x2bar - mu*mu+1e-5)

        return torch.cat((mu,s), dim=-1)



    
class GlobalMeanLogVarPool1d(_GlobalPool1d):

    def __init__(self, dim=-1, keepdim=False, batch_dim=0):
        super(GlobalMeanLogVarPool1d, self).__init__(dim, keepdim, batch_dim)
        self.size_multiplier = 2

    def forward(self, x, weights=None):
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            x2bar = torch.mean(x**2, dim=self.dim, keepdim=self.keepdim)
            s = torch.log(x2bar - mu*mu+1e-5) #for stability in case var=0
            return torch.cat((mu,logvar), dim=-1)

        weights = self._standarize_weights(weights, x.dim())

        xbar = torch.mean(weights*x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        mu = xbar/wbar
        x2bar = torch.mean(weights*x**2, dim=self.dim, keepdim=self.keepdim)/wbar
        logvar = torch.log(x2bar - mu*mu)

        return torch.cat((mu,logvar), dim=-1)


#this is wrong            
class LDEPool1d(_GlobalPool1d):

    def __init__(self, input_units, num_comp=64, dist_pow=2, use_bias=False,
                 dim=-1, keepdim=False, batch_dim=0):
        super(LDEPool1d, self).__init__(dim, keepdim, batch_dim)
        self.mu = nn.Parameter(torch.randn(num_comp,input_units))
        self.prec = nn.Parameter(torch.ones(num_comp))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_comp))
        else:
            self.bias = 0            

        self.dist_pow = dist_pow
        if dist_pow == 1:
            self.dist_f = lambda x: torch.norm(x, p=2, dim=1)
        else:
            self.dist_f = lambda x: torch.sum(x**2, dim=1)

        self.size_multiplier = num_comp

        
    def forward(self, x, weights=None):
        x = torch.unsqueeze(x, dim=-2)
        delta = x - self.mu
        dist = self.dist_f(delta)

        llk = - self.prec**2 * dist + self.bias
        r = F.softmax(llk, dim=-1)
        if weights is not None:
            if weights.dim()==2:
                weights = torch.unsqueze(weights, dim=-1)
            r *= weights

        r = r/(torch.sum(r, dim=-1, keepdims=True)+1e-9)
        pool = torch.sum(r*delta, dim=-2)
        if self.keepdim:
            return pool.view(x.shape[0], 1, -1)

        return pool.view(x.shape[0], -1)
        

        
    def get_config(self):

        config = { 'input_units': self.mu.shape[1],
                   'num_comp': self.mu.shape[0],
                   'dist_pow': self.dist_pow,
                   'use_bias': self.use_bias }

        base_config = super(LDEPool1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
