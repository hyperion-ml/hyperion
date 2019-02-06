#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Converts from Ark format to h5 format (deprecated, use copy-feats.py)
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
from six.moves import xrange

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.io import HypDataWriter, KaldiDataReader

def ark2hyp(input_file, input_dir, output_file, field, chunk_size, squeeze):

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ark_r = KaldiDataReader(input_file, input_dir)
    h_w = HypDataWriter(output_file)
    
    while not(ark_r.eof()):
        X, keys = ark_r.read(num_records=chunk_size, squeeze=squeeze)
        h_w.write(keys, field, X)


if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Compacts .arr files into a hdf5 file.')

    parser.add_argument('--input-file',dest='input_file', required=True)
    parser.add_argument('--input-dir', dest='input_dir', default=None)
    parser.add_argument('--output-file', dest='output_file', required=True)
    parser.add_argument('--field', dest='field', default='')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=None)
    parser.add_argument('--squeeze', dest='squeeze', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    
    ark2hyp(**vars(args))
    
