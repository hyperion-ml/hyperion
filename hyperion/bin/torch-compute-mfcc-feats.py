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

#import numpy as np
import torch

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import compression_methods
from hyperion.torch.layers import AudioFeatsFactory as AFF
from hyperion.feats import MFCC

def compute_mfcc_feats(input_path, output_path,
                       compress, compression_method, write_num_frames, **kwargs):

    mfcc_args = AFF.filter_args(**kwargs)
    #mfcc_args['remove_dc_offset'] = False
    #mfcc_args['dither'] = 0
    #mfcc_args['preemph_coeff'] = 0
    #mfcc_args['window_type'] = 'rectangular'
    #mfcc_args['audio_feat'] = 'log_spec'
    #mfcc_args['audio_feat'] = 'mfcc'
    mfcc = AFF.create(**mfcc_args)
    print(mfcc_args)
    # mfcc_args['input_step'] = 'wave'
    # mfcc_args['output_step'] = 'logfb'
    # #mfcc_args['output_step'] = 'mfcc'
    # del mfcc_args['audio_feat']
    # del mfcc_args['use_fft_mag']
    # mfcc2 = MFCC(**mfcc_args)
    #print(mfcc.wav2win._window.numpy())
    #print(mfcc2._window)
    input_args = AR.filter_args(**kwargs)
    reader = AR(input_path, **input_args)
    
    writer = DWF.create(output_path, scp_sep=' ',
                        compress=compress,
                        compression_method=compression_method)

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, 'w')
    
    for data in reader:
        key, x, fs = data
        logging.info('Extracting MFCC for %s' % (key))
        t1 = time.time()
        #y2 = mfcc2.compute(x)
        x = torch.tensor(x[None,:], dtype=torch.get_default_dtype())
        y = mfcc(x).squeeze(0).detach().numpy()

        dt = (time.time() - t1)*1000
        rtf = mfcc.frame_shift*y.shape[0]/dt
        logging.info('Extracted MFCC for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f' %
                     (key, y.shape[0], dt, rtf))
        writer.write([key], [y])
        
        if write_num_frames is not None:
            f_num_frames.write('%s %d\n' % (key, y.shape[0]))
        
        # print(y.shape)
        # print(y2.shape)
        # print(y[1:-1])
        # print(y2[1:-1])
        # return
            
    if write_num_frames is not None:
        f_num_frames.close()
    

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Compute MFCC features in pytorch')

    parser.add_argument('--input', dest='input_path', required=True)
    parser.add_argument('--output', dest='output_path', required=True)
    parser.add_argument('--write-num-frames', dest='write_num_frames', default=None)

    DRF.add_argparse_args(parser)
    AFF.add_argparse_args(parser)
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
    
