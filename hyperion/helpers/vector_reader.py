"""
Loads data to train LDA, PLDA, PDDA
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import copy

import numpy as np

from ..io import HypDataReader
from ..utils.scp_list import SCPList
from ..transforms import TransformList


class VectorReader(object):

    def __init__(self, v_file, key_file, preproc=None, scp_sep='=', v_field=''):

        self.r = HypDataReader(v_file)
        self.scp = SCPList.load(key_file, sep=scp_sep)
        self.preproc = preproc
        self.field = v_field

        
            
    def read(self):
        
        x = self.r.read(self.scp.file_path, self.field, return_tensor=True)
        if self.preproc is not None:
            x = self.preproc.predict(x)
        return x


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('scp_sep', 'v_field')
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
        parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default='=',
                            help=('scp file field separator'))
        parser.add_argument(p1+'v-field', dest=(p2+'v_field'), default='',
                            help=('dataset field in input vector file'))
        
    

                            
                    
