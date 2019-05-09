"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
import six

import torch.nn as nn

from .global_pool import *

class GlobalPool1dFactory(object):

    @staticmethod
    def create(pool_type,
               num_comp=64, dist_exp=2, use_bias=False,
               dim=1, keepdim=False, batch_dim=0):
        if pool_type = 'avg':
            return GlobalAvgPool1d(
                dim=dim, keepdim=keepdim, batch_dim=batch_dim)

        if pool_type = 'mean+std':
            return GlobalMeanStdPool1d(
                dim=dim, keepdim=keepdim, batch_dim=batch_dim)

        if pool_type = 'mean+logvar':
            return GlobalMeanLogVarPool1d(
                dim=dim, keepdim=keepdim, batch_dim=batch_dim)

        if pool_type = 'lde':
            return LDEPool1d(
                num_comp=num_comp, dist_exp=dist_exp, use_bias=use_bias,
                dim=dim, keepdim=keepdim, batch_dim=batch_dim)



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('pool_type', 'batch_dim', 'pool_dim', 'keepdim', 'beta2',
                      'input_units', 'num_comp', 'wo_bias', 'dist_pow')

        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)
    

        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'pool_type', dest=(p2+'pool_type'), type=str.lower,
                        default='mean+stddev',
                        choices=['avg','mean+stddev', 'mean+logvar', 'lde'],
                        help=('Pooling methods: Avg, Mean+Std, Mean+logVar, LDE'))
        
        parser.add_argument(p1+'pool-dim' , dest=(p2+'pooling_dim'),
                            default=1, type=int,
                            help=('Pooling dimension, usually time dimension'))
        
        parser.add_argument(p1+'batch-dim' , dest=(p2+'batch_dim'),
                            default=0, type=int,
                            help=('Batch-size dimension'))

        parser.add_argument(p1+'keepdim' , dest=(p2+'keepdim'),
                            default=False, action='store_true',
                            help=('keeps the pooling dimension as singletone'))

        parser.add_argument(p1+'input-units' , dest=(p2+'input_units'),
                            default=0, type=int,
                            help=('feature size for LDE pooling'))

        parser.add_argument(p1+'num-comp' , dest=(p2+'num_comp'),
                            default=0, type=int,
                            help=('number of components for LDE pooling'))

        parser.add_argument(p1+'dist-pow' , dest=(p2+'dist_pow'),
                            default=2, type=int,
                            help=('Distace power for LDE pooling'))
        
        parser.add_argument(p1+'wo-bias' , dest=(p2+'wo_bias'),
                            default=False, action='store_true',
                            help=('Don\' use bias in LDE'))
        
