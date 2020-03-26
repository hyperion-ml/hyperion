"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch.nn as nn

from .global_pool import *

class GlobalPool1dFactory(object):

    @staticmethod
    def create(pool_type, in_feats=None,
               num_comp=64, dist_pow=2, use_bias=False,
               num_heads=8, d_k=256, d_v=256, bin_attn=False,
               dim=-1, keepdim=False, **kwargs):

        if pool_type == 'avg':
            return GlobalAvgPool1d(dim=dim, keepdim=keepdim)

        if pool_type == 'mean+stddev':
            return GlobalMeanStdPool1d(dim=dim, keepdim=keepdim)

        if pool_type == 'mean+logvar':
            return GlobalMeanLogVarPool1d(dim=dim, keepdim=keepdim)

        if pool_type == 'lde':
            return LDEPool1d(
                in_feats, num_comp=num_comp, dist_pow=dist_pow, 
                use_bias=use_bias, dim=dim, keepdim=keepdim)

        if pool_type == 'scaled-dot-prod-att-v1':
            return ScaledDotProdAttV1Pool1d(
                in_feats, num_heads=num_heads, d_k=d_k, d_v=d_v,
                bin_attn=bin_attn, dim=dim, keepdim=keepdim)


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if 'wo_bias' in kwargs:
            kwargs['use_bias'] = not kwargs['wo_bias']
            del kwargs['wo_bias']

        valid_args = ('pool_type', 'dim', 'keepdim', 
                      'in_feats', 'num_comp', 'use_bias', 'dist_pow', 
                      'num_heads', 'd_k', 'd_v', 'bin_attn')

        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)
    

        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        parser.add_argument(
            p1+'pool-type', type=str.lower, default='mean+stddev',
            choices=['avg','mean+stddev', 'mean+logvar', 
                     'lde', 'scaled-dot-prod-att-v1'],
            help=('Pooling methods: Avg, Mean+Std, Mean+logVar, LDE, '
                  'scaled-dot-product-attention-v1'))
        
        parser.add_argument(p1+'dim' , 
                            default=1, type=int,
                            help=('Pooling dimension, usually time dimension'))
        
        # parser.add_argument(p1+'batch-dim',
        #                     default=0, type=int,
        #                     help=('Batch-size dimension'))

        parser.add_argument(
            p1+'keepdim', default=False, action='store_true',
            help=('keeps the pooling dimension as singletone'))

        parser.add_argument(
            p1+'in-feats', default=0, type=int,
            help=('feature size for LDE/Att pooling'))

        parser.add_argument(
            p1+'num-comp', default=8, type=int,
            help=('number of components for LDE pooling'))

        parser.add_argument(
            p1+'dist-pow', default=2, type=int,
            help=('Distace power for LDE pooling'))
        
        parser.add_argument(
            p1+'wo-bias', default=False, action='store_true',
            help=('Don\'t use bias in LDE'))

        parser.add_argument(
            p1+'num-heads', default=4, type=int,
            help=('number of attention heads'))

        parser.add_argument(
            p1+'d-k', default=256, type=int,
            help=('key dimension for attention'))

        parser.add_argument(
            p1+'d-v', default=256, type=int,
            help=('value dimension for attention'))
        
        parser.add_argument(
            p1+'bin-attn', default=False, action='store_true',
            help=('Use binary attention, i.e. sigmoid instead of softmax'))


    @staticmethod
    def get_config(layer):
        
        config = layer.get_config()
        if isinstance(layer, GlobalAvgPool1d):
            config['pool_type'] = 'avg'
        
        if isinstance(layer, GlobalMeanStdPool1d):
            config['pool_type'] = 'mean+stddev'

        if isinstance(layer, GlobalMeanLogVarPool1d):
            config['pool_type'] = 'mean+logvar'

        if isinstance(layer, LDEPool1d):
            config['pool_type'] = 'lde'

        if isinstance(layer, ScaledDotProdAttV1Pool1d):
            config['pool_type'] = 'scaled-dot-prod-att-v1'

        return config
        
