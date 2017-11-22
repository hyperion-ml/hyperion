#!/usr/bin/env python
"""
Copy features/vectors and change format
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
import numpy as np
from six.moves import xrange

from hyperion.io2 import H5Merger

def copy_feats(input_path, output_path, compress, compression_method, write_num_frames, chunk_size):

    
    

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Copy features and change format')

    parser.add_argument('--input-path', dest='input_path', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--compress', dest='compress', default=False, action='store_true')
    parser.add_argument('--compression-method', dest='compression_method', default='auto',
                        choices=['auto', 'speech-feat', '2byte-auto', '2byte-signed-integer',
                                 '1byte-auto', '1byte-unsigned-integer', '1byte-0-1'])
    parser.add_argument('--write-num-frames', dest='write_num_frames', default=None)
    #parser.add_argument('--field', dest='field', default='')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=None)
    #parser.add_argument('--squeeze', dest='squeeze', default=False, action='store_true')

    args=parser.parse_args()

    copy(**vars(args))
    
