#!/usr/bin/env python
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
import re
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores

def convert(input_file, output_file, test_list, class_file, add_ext):

    scores = TrialScores.load(input_file)

    if test_list is None:
        seg_set = scores.seg_set
    else:
        with open(test_list, 'r') as f:
            seg_set = [ seg for seg in [line.rstrip().split(' ')[0] for line in f]
                        if seg!='segmentid']
            if add_ext:
                exts = [os.path.splitext(seg)[1] for seg in seg_set]
                seg_set = [os.path.splitext(seg)[0] for seg in seg_set]

    if class_file is None:
        model_set = scores.model_set
    else:
        with open(class_file, 'r') as f:
            model_set = [line.rstrip().split()[0] for line in f]

    ndx = TrialNdx(model_set, seg_set)
    scores = scores.set_missing_to_value(ndx, -100)

    if add_ext:
        scores.seg_set = [seg+ext for seg, ext in zip(scores.seg_set, exts)]
            
    with open(output_file, 'w') as f:
        f.write('segmentid\t')
        for model in scores.model_set[:-1]:
            f.write('%s\t' % model)
        f.write('%s\n' % scores.model_set[-1])
        for i in xrange(scores.scores.shape[1]):
            f.write('%s\t' % scores.seg_set[i])
            for j in xrange(scores.scores.shape[0]-1):
                f.write('%f\t' % scores.scores[j, i])
            f.write('%f\n' % scores.scores[-1, i])
            
                                  

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Convert scores to LRE format')

    parser.add_argument('--input-file', dest='input_file', required=True)
    parser.add_argument('--output-file', dest='output_file', required=True)
    parser.add_argument('--test-list', dest='test_list', default=None)
    parser.add_argument('--class-file', dest='class_file', default=None)
    parser.add_argument('--add-ext', dest='add_ext', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    convert(**vars(args))

