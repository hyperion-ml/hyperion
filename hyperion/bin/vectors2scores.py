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

import numpy as np

from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.utils.trial_scores import TrialScores

def convert(input_file, output_file, class_file):

    r = DRF.create(input_file)
    seg_set, score_mat = r.read(0, squeeze=True)

    with open(class_file, 'r') as f:
        model_set = [line.rstrip().split()[0] for line in f]

    scores = TrialScores(model_set, seg_set, score_mat.T)
    scores.save(output_file)
                                  

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Converts scores from vector format to TrialScores format')

    parser.add_argument('--input-file', dest='input_file', required=True)
    parser.add_argument('--output-file', dest='output_file', required=True)
    parser.add_argument('--class-file', dest='class_file', default=None)
    
    args=parser.parse_args()

    convert(**vars(args))

