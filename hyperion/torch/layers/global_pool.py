"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

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


class LDEPool1d(_GlobalPool1d):

    def __init__(self, in_feats, num_comp=64, dist_pow=2, use_bias=False,
                 dim=-1, keepdim=False):
        super(LDEPool1d, self).__init__(dim, keepdim, batch_dim=0)
        self.mu = nn.Parameter(torch.randn((num_comp,in_feats)))
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
    def in_feats(self):
        return self.mu.shape[1]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{}(in_feats={}, num_comp={}, dist_pow={}, use_bias={}, dim={}, keepdim={}, batch_dim={})'.format(
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
        r = nnf.softmax(llk, dim=-1)
        if weights is not None:
            r *= weights

        r = torch.unsqueeze(r, dim=-1)
        N = torch.sum(r, dim=1) + 1e-9
        F = torch.sum(r*delta, dim=1)
        pool = F/N
        pool = pool.contiguous().view(-1, self.num_comp*self.in_feats)
        if self.keepdim:
            if self.dim == 1 or self.dim == -2:
                pool.unsqueeze_(1)
            else:
                pool.unsqueeze_(-1)

        return pool
        

        
    def get_config(self):

        config = { 'in_feats': self.in_feats,
                   'num_comp': self.num_comp,
                   'dist_pow': self.dist_pow,
                   'use_bias': self.use_bias }

        base_config = super(LDEPool1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class ScaledDotProdAttV1Pool1d(_GlobalPool1d):

    def __init__(self, in_feats, num_heads, d_k, d_v, dim=-1, keepdim=False):
        super(ScaledDotProdAttV1Pool1d, self).__init__(dim, keepdim)

        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.q = nn.Parameter(torch.Tensor(1, num_heads, 1, d_k))
        nn.init.orthogonal_(self.q)
        self.linear_k = nn.Linear(in_feats, num_heads*d_k)
        self.linear_v = nn.Linear(in_feats, num_heads*d_v)
        self.attn = None
        self.size_multiplier = num_heads*d_v/in_feats


    @property
    def in_feats(self):
        return self.linear_v.in_features

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        s = '{}(in_feats={}, num_heads={}, d_k={}, d_v={}, dim={}, keepdim={})'.format(
            self.__class__.__name__, self.in_feats, self.num_heads,
            self.d_k, self.d_v, self.dim, self.keepdim)
        return s


    def forward(self, x, weights=None):
        batch_size = x.size(0)
        if self.dim != 1:
            x = x.transpose(1, self.dim)

        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_v)
        k = k.transpose(1, 2)  # (batch, head, time, d_k)
        v = v.transpose(1, 2)  # (batch, head, time, d_v)

        scores = torch.matmul(self.q, k.transpose(-2,-1)) / math.sqrt(self.d_k)  # (batch, head, 1, time)
        scores = scores.squeeze(dim=-1)                    # (batch, head, time)
        if weights is not None:
            mask = weights.view(batch_size, 1, 1, -1).eq(0)  # (batch, 1, 1,time)
            min_value = -1e200
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, 1, time)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, 1, time)
        #print(self.q.shape, k.shape, v.shape, scores.shape, self.attn.shape)
        x = torch.matmul(self.attn, v)  # (batch, head, 1, d_v)
        if self.keepdim:
            x = x.view(batch_size, 1, self.num_heads * self.d_v)  # (batch, 1, d_model)
        else:
            x = x.view(batch_size, self.num_heads * self.d_v)  # (batch, d_model)
        return x  


    def get_config(self):
        config = {'in_feats': self.in_feats,
                  'num_heads': self.num_heads,
                  'd_k': self.d_k,
                  'd_v': self.d_v }

        base_config = super(ScaledDotProdAttV1Pool1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
