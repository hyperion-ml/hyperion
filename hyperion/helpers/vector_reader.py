"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
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

from ..io import RandomAccessDataReaderFactory as DRF
from ..utils.scp_list import SCPList
from ..transforms import TransformList


class VectorReader(object):
    """Class to load data to train PCA, centering, whitening.
    """
    def __init__(self, v_file, key_file, preproc=None, vlist_sep=' '):

        self.r = DRF.create(v_file)
        self.scp = SCPList.load(key_file, sep=vlist_sep)
        self.preproc = preproc

        
            
    def read(self):
        x = self.r.read(self.scp.key, squeeze=True)
        if self.preproc is not None:
            x = self.preproc.predict(x)
        return x


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('vlist_sep')
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
        parser.add_argument(p1+'vlist-sep', dest=(p2+'vlist_sep'), default=' ',
                            help=('utterance file field separator'))
        # parser.add_argument(p1+'v-field', dest=(p2+'v_field'), default='',
        #                     help=('dataset field in input vector file'))
        
    

                            
                    
