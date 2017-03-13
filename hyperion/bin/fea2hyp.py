#!/bin/env python
"""
Program to compact a list of .arr files into a unique h5 file
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
import numpy as np
from six.moves import xrange

from cnr_deeptk.utils.arr2h5_compactor import Arr2H5Compactor

def arr2h5(file_list,in_dir,h5file,target_dir,
           in_file_has_power,in_file_has_vad,
           chunk_size,resize_chunk_size):

    print(locals())
    h5=Arr2H5Compactor(file_list,in_dir,h5file,target_dir,
                       in_file_has_power,in_file_has_vad,
                       chunk_size,resize_chunk_size)
    h5.compact_files()


if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Compacts .arr files into a hdf5 file.')

    parser.add_argument('--file-list',dest='file_list',required=True)
    parser.add_argument('--in-dir',dest='in_dir',required=True)
    parser.add_argument('--target-dir',dest='target_dir',default=None)
    parser.add_argument('--h5file',dest='h5file',required=True)

    parser.add_argument('--in-file-has-power',dest='in_file_has_power',
                        default=False,action='store_true')
    parser.add_argument('--in-file-has-vad',dest='in_file_has_vad',
                        default=False,action='store_true')
    parser.add_argument('--chunk-size',dest='chunk_size',default=10000,type=int)
    parser.add_argument('--resize-chunk-size',dest='resize_chunk_size',
                        default=1000000,type=int)

    args=parser.parse_args()

    arr2h5(**vars(args))
    
