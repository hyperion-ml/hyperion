#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
import logging

import numpy as np
from six.moves import xrange

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import compression_methods
from hyperion.feats import MFCC

def compute_mfcc_feats(input_path, output_path,
                       compress, compression_method, write_num_frames, **kwargs):

    mfcc_args = MFCC.filter_args(**kwargs)
    mfcc = MFCC(**mfcc_args)

    if mfcc.input_step == 'wave':
        input_args = AR.filter_args(**kwargs)
        reader = AR(input_path, **input_args)
    else:
        input_args = DRF.filter_args(**kwargs)
        reader = DRF.create(input_path, **input_args)

    writer = DWF.create(output_path, scp_sep=' ',
                        compress=compress,
                        compression_method=compression_method)

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, 'w')
    
    for data in reader:
        if mfcc.input_step == 'wave':
            key, x, fs = data
        else:
            key, x = data
        logging.info('Extracting MFCC for %s' % (key))
        t1 = time.time()
        y = mfcc.compute(x)
        dt = (time.time() - t1)*1000
        rtf = mfcc.frame_shift*y.shape[0]/dt
        logging.info('Extracted MFCC for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f' %
                     (key, y.shape[0], dt, rtf))
        writer.write([key], [y])
        
        if write_num_frames is not None:
            f_num_frames.write('%s %d\n' % (key, y.shape[0]))

        mfcc.reset()
            
    if write_num_frames is not None:
        f_num_frames.close()
    

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Compute MFCC features')

    parser.add_argument('--input', dest='input_path', required=True)
    parser.add_argument('--output', dest='output_path', required=True)
    parser.add_argument('--write-num-frames', dest='write_num_frames', default=None)

    DRF.add_argparse_args(parser)
    MFCC.add_argparse_args(parser)
    parser.add_argument('--compress', dest='compress', default=False, action='store_true', help='Compress the features')
    parser.add_argument('--compression-method', dest='compression_method', default='auto',
                        choices=compression_methods, help='Compression method')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int,
                        help='Verbose level')
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    compute_mfcc_feats(**vars(args))
    
