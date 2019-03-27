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
from ..utils import Utt2Info, TrialNdx, ExtSegmentList
from ..transforms import TransformList

class TrackingDataReader(object):
    """
    Loads ndx, enroll file and x-vectors to do speaker tracking with PLDA
    """

    def __init__(self, v_file, ndx_file, enroll_file, segments_file,
                 preproc, tlist_sep=' ', 
                 model_idx=1, num_model_parts=1, seg_idx=1, num_seg_parts=1):

        self.r = DRF.create(v_file)
        self.preproc = preproc

        enroll = Utt2Info.load(enroll_file, sep=tlist_sep)
        ndx = TrialNdx.load(ndx_file)

        ndx, enroll = TrialNdx.parse_eval_set(ndx, enroll)

        segments = ExtSegmentList.load(segments_file)
        if num_model_parts > 1 or num_seg_parts > 1:
            ndx = TrialNdx.split(model_idx, num_model_parts, seg_idx, num_seg_parts)
            enroll = enroll.filter_info(ndx.model_set)
            segments = segments.filter(ndx.seg_set)
        
        self.enroll = enroll
        self.ndx = ndx
        self.segments = segments


        
    def read(self, key=None):
        if key is None:
            enroll, ndx_seg, segments = self._read_all_utts()
        else:
            enroll, ndx_seg, segments = self._read_single_utt(key)

        x_e = self.r.read(enroll.key, squeeze=True)
        x_t = self.r.read(ndx_seg.seg_set, squeeze=True)
    
        if self.preproc is not None:
            x_e = self.preproc.predict(x_e)
            x_t = self.preproc.predict(x_t)

        return x_e, x_t, enroll.info, ndx_seg, segments

        
    def _read_all_utts(self):
        ndx_seg = self.ndx.apply_segmentation_to_test(self.segments)
        return self.enroll, ndx_seg, self.segments
    

    def _read_single_utt(self, key):
        ndx = self.ndx.filter(self.ndx.model_set, [key])
        ndx_seg = ndx.apply_segmentation_to_test(self.segments)
        enroll, ndx_seg = TrialNdx.parse_eval_set(self.enroll, ndx_seg)
        segments = self.segments.filter([key])
        return enroll, ndx_seg, segments


    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('tlist_sep', 
                      'model_idx','num_model_parts',
                      'seg_idx', 'num_seg_parts')
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
        parser.add_argument(p1+'tlist-sep', dest=(p2+'tlist_sep'), default=' ',
                            help=('trial lists field separator'))

        parser.add_argument(p1+'model-part-idx', dest=(p2+'model_idx'), default=1, type=int,
                            help=('model part index'))
        parser.add_argument(p1+'num-model-parts', dest=(p2+'num_model_parts'), default=1, type=int,
                            help=('number of parts in which we divide the model'
                                  'list to run evaluation in parallel'))
        parser.add_argument(p1+'seg-part-idx', dest=(p2+'seg_idx'), default=1, type=int,
                            help=('test part index'))
        parser.add_argument(p1+'num-seg-parts', dest=(p2+'num_seg_parts'), default=1, type=int,
                            help=('number of parts in which we divide the test list '
                                  'to run evaluation in parallel'))
        
