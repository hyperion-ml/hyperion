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
#from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import DataWriterFactory as DWF
from hyperion.feats import energy_vad

def compute_vad(input_path, output_path,  **kwargs):

    mfcc_args = EnergyVAD.filter_args(**kwargs)
    mfcc = EnergVAD(**mfcc_args)
    
    input_args = AR.filter_args(**kwargs)
    reader = AR(input_path, **input_args)

    writer = DWF.create(output_path, scp_sep=' ')

    for data in reader:
        key, x, fs = data
        logging.info('Extracting VAD for %s' % (key))
        t1 = time.time()
        y = vad.compute(x)
        dt = (time.time() - t1)*1000
        rtf = vad.frame_shift*y.shape[0]/dt
        logging.info('Extracted VAD for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f' %
                     (key, y.shape[0], dt, rtf))
        writer.write([key], [y])
        
        vad.reset()
            
    

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Compute Kaldi Energy VAD')

    parser.add_argument('--input', dest='input_path', required=True)
    parser.add_argument('--output', dest='output_path', required=True)

    DRF.add_argparse_args(parser)
    EnergyVAD.add_argparse_args(parser)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int,
                        help='Verbose level')
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    compute_vad(**vars(args))
    
