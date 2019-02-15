#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import time
import logging

import numpy as np
from six.moves import xrange

from hyperion.hyp_defs import config_logger
from hyperion.utils.kaldi_matrix import compression_methods
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import RandomAccessDataReaderFactory as RDRF
from hyperion.feats import MeanVarianceNorm as MVN
from hyperion.feats import FrameSelector as FSel


def process_feats(input_spec, output_spec, vad_spec,
                  scp_sep, path_prefix, vad_path_prefix, part_idx, num_parts,
                  compress, compression_method, **kwargs):

    logging.info('initializing')
    mvn_args = MVN.filter_args(**kwargs)
    mvn = MVN(**mvn_args)
    if vad_spec is not None:
        fs_args = FSel.filter_args(**kwargs)
        fs = FSel(**fs_args)

    logging.info('opening output stream: %s' % (output_spec))
    with DWF.create(output_spec,
                    compress=compress, compression_method=compression_method,
                    scp_sep=scp_sep) as writer:

        logging.info('opening input stream: %s' % (output_spec))
        with DRF.create(input_spec, path_prefix=path_prefix, scp_sep=scp_sep,
                        part_idx=part_idx, num_parts=num_parts) as reader:
            if vad_spec is not None:
                logging.info('opening VAD stream: %s' % (vad_spec))
                v_reader = RDRF.create(vad_spec, path_prefix=vad_path_prefix, scp_sep=scp_sep)
    
            while not reader.eof():
                key, data = reader.read(1)
                if len(key) == 0:
                    break
                logging.info('processing feats at %s' % (key[0]))
                x = mvn.normalize(data[0])
                if vad_spec is not None:
                    vad = v_reader.read(key)[0]
                    tot_frames = x.shape[0]
                    x = fs.select(x, vad)
                    logging.info('detected %d/%d (%.2f \%) speech frames' % (x.shape[0], tot_frames, x.shape[0]/total_frames*100))
                if x.shape[0]>0:
                    writer.write(key, [x])


    
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Apply CMVN and remove silence')

    parser.add_argument('--input', dest='input_spec', required=True)
    parser.add_argument('--output', dest='output_spec', required=True)
    parser.add_argument('--vad', dest='vad_spec', default=False)
    parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default=' ',
                        help=('scp file field separator'))
    parser.add_argument(p1+'path-prefix', dest=(p2+'path_prefix'), default=None,
                        help=('scp file_path prefix'))
    parser.add_argument(p1+'vad-path-prefix', dest=(p2+'vad_path_prefix'), default=None,
                        help=('scp file_path prefix for vad'))
    parser.add_argument(p1+'part-idx', dest=(p2+'part_idx'), type=int, default=1,
                        help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument(p1+'num-parts', dest=(p2+'num_parts'), type=int, default=1,
                        help=('splits the list of files in num-parts and process part_idx'))
    
    parser.add_argument('--compress', dest='compress', default=False, action='store_true', help='Lossy compress the features')
    parser.add_argument('--compression-method', dest='compression_method', default='auto',
                        choices=compression_methods,
                        help=('Kaldi compression method: '
                              '{auto (default), speech_feat, '
                              '2byte-auto, 2byte-signed-integer, '
                              '1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.'))
    MVN.add_argparse_args(parser)
    FSel.add_argparse_args(parser)
                        
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    process_feats(**vars(args))
    
