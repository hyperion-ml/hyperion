"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                   'keepdim': self.keepdim }

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

    # def forward(self, x, weights=None):
    #     if weights is None:
    #         mu = torch.mean(x, dim=self.dim, keepdim=True)
    #         delta = x - mu
    #         if not self.keepdim:
    #             mu = mu.squeeze(dim=self.dim)

    #         #s = torch.sqrt(x2bar - mu*mu+1e-3) #for stability in case var=0
    #         s = torch.sqrt(
    #             torch.mean(delta**2, dim=self.dim, keepdim=self.keepdim)+1e-5)

    #         if len(x)==24:
    #             z = x
    #             x = mu
    #             mm = x.mean(dim=-1)
    #             mx = torch.max(x, dim=-1)[0]
    #             mn = torch.min(x, dim=-1)[0]
    #             na = torch.isnan(x).any(dim=-1)
    #             logging.info('pool-mean-mu-std-mean={}'.format(str(mm[9])))
    #             logging.info('pool-mean-mu-std-max={}'.format(str(mx[9])))
    #             logging.info('pool-mean-mu-std-min={}'.format(str(mn[9])))
    #             logging.info('pool-mean-mu-std-na={}'.format(str(na[9])))

    #             # x = x2bar
    #             # mm = x.mean(dim=-1)
    #             # mx = torch.max(x, dim=-1)[0]
    #             # mn = torch.min(x, dim=-1)[0]
    #             # na = torch.isnan(x).any(dim=-1)
    #             # logging.info('pool-mean-x2bar-std-mean={}'.format(str(mm[9])))
    #             # logging.info('pool-mean-x2bar-std-max={}'.format(str(mx[9])))
    #             # logging.info('pool-mean-x2bar-std-min={}'.format(str(mn[9])))
    #             # logging.info('pool-mean-x2bar-std-na={}'.format(str(na[9])))

    #             x = s
    #             mm = x.mean(dim=-1)
    #             mx = torch.max(x, dim=-1)[0]
    #             mn = torch.min(x, dim=-1)[0]
    #             na1 = torch.isnan(x)
    #             na = na1.any(dim=-1)
    #             logging.info('pool-mean-s-std-mean={}'.format(str(mm[9])))
    #             logging.info('pool-mean-s-std-max={}'.format(str(mx[9])))
    #             logging.info('pool-mean-s-std-min={}'.format(str(mn[9])))
    #             logging.info('pool-mean-s-std-na={}'.format(str(na[9])))
                
    #             logging.info('pool-mean-std-xnan={}'.format(str(z[9][na1[9]])))
    #             logging.info('pool-mean-std-munan={}'.format(str(mu[9][na1[9]])))
    #             #logging.info('pool-mean-std-x2barnan={}'.format(str(x2bar[9][na1[9]])))


    #         return torch.cat((mu,s), dim=-1)

    #     weights = self._standarize_weights(weights, x.dim())

    #     xbar = torch.mean(weights*x, dim=self.dim, keepdim=self.keepdim)
    #     wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
    #     mu = xbar/wbar
    #     x2bar = torch.mean(weights*x**2, dim=self.dim, keepdim=self.keepdim)/wbar
    #     s = torch.sqrt(x2bar - mu*mu+1e-3)

    #     return torch.cat((mu,s), dim=-1)


    def forward(self, x, weights=None):
        if weights is None:
            # # this can produce slightly negative variance when relu6 saturates in all time steps
            # mu = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            # x2bar = torch.mean(x**2, dim=self.dim, keepdim=self.keepdim)
            # s = torch.sqrt(x2bar - mu*mu+1e-5) #for stability in case var=0

            # this version should be more stable
            mu = torch.mean(x, dim=self.dim, keepdim=True)
            delta = x - mu
            if not self.keepdim:
                mu.squeeze_(dim=self.dim)

            s = torch.sqrt(
                torch.mean(delta**2, dim=self.dim, keepdim=self.keepdim)+1e-5)

            return torch.cat((mu,s), dim=-1)

        weights = self._standarize_weights(weights, x.dim())

        # xbar = torch.mean(weights*x, dim=self.dim, keepdim=self.keepdim)
        # wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        # mu = xbar/wbar
        # x2bar = torch.mean(weights*x**2, dim=self.dim, keepdim=self.keepdim)/wbar
        # s = torch.sqrt(x2bar - mu*mu+1e-5)

        # this version should be more stable
        xbar = torch.mean(weights*x, dim=self.dim, keepdim=True)
        wbar = torch.mean(weights, dim=self.dim, keepdim=True)
        mu = xbar/wbar
        delta = x - mu
        var = torch.mean(weights*delta**2, dim=self.dim, keepdim=True)/wbar
        s = torch.sqrt(var+1e-5)
        if not self.keepdim:
            mu.squeeze_(self.dim)
            s.squeeze_(self.dim)

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

    def __init__(self, in_units, num_comp=64, dist_pow=2, use_bias=False,
                 dim=-1, keepdim=False):
        super(LDEPool1d, self).__init__(dim, keepdim, batch_dim=0)
        self.mu = nn.Parameter(torch.randn((num_comp,in_units)))
        self.prec = nn.Parameter(torch.ones((num_comp,)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((num_comp,)))
        else:
            self.bias = 0            

        self.dist_pow = dist_pow
        if dist_pow == 1:
            self.dist_f = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dist_f = lambda x: torch.sum(x**2, dim=-1)

        self.size_multiplier = num_comp


    @property
    def num_comp(self):
        return self.mu.shape[0]

    @property
    def in_units(self):
        return self.mu.shape[1]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{}(in_units={}, num_comp={}, dist_pow={}, use_bias={}, dim={}, keepdim={}, batch_dim={})'.format(
            self.__class__.__name__, self.mu.shape[1], self.mu.shape[0], self.dist_pow,
            self.use_bias, self.dim, self.keepdim, self.batch_dim)
        return s

        
    def forward(self, x, weights=None):
        if self.dim != 1 or self.dim != -2:
            x = x.transpose(1, self.dim)

        x = torch.unsqueeze(x, dim=2)
        delta = x - self.mu
        dist = self.dist_f(delta)

        llk = - self.prec**2 * dist + self.bias
        r = F.softmax(llk, dim=-1)
        if weights is not None:
            r *= weights
            r = r/(torch.sum(r, dim=-1, keepdims=True)+1e-9)

        #r.unsqueeze_(dim=-1)
        r = torch.unsqueeze(r, dim=-1)
        pool = torch.sum(r*delta, dim=1)
        pool = pool.contiguous().view(-1, self.num_comp*self.in_units)
        if self.keepdim:
            if self.dim == 1 or self.dim == -2:
                pool.unsqueeze_(1)
            else:
                pool.unsqueeze_(-1)

        return pool
        

        
    def get_config(self):

        config = { 'in_units': self.in_units,
                   'num_comp': self.num_comp,
                   'dist_pow': self.dist_pow,
                   'use_bias': self.use_bias }

        base_config = super(LDEPool1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
