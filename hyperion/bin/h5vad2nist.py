#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Converts from Ark format to NIST OpenSAT
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.io import HypDataReader

def bin2intervals(vad):
    delta = np.abs(np.diff(vad))
    change_points = np.where(delta>0)[0]
    num_interv = len(change_points)+1
    if vad[0] == 1:
        speech = True
    else:
        speech = False

    start = np.zeros((num_interv,))
    stop = np.zeros((num_interv,))
    state = np.zeros((num_interv,), dtype=bool)
    conf = np.ones((num_interv,))

    prev_stop = 0
    for i, p in enumerate(change_points):
        start[i] = prev_stop
        stop[i] = p
        prev_stop = p
        state[i] = speech
        speech = not speech
        conf[i] = 1.0
    start[-1] = prev_stop
    stop[-1] = len(vad)
    state[-1] = speech
    conf[-1] = 1.0

    start *= 0.01
    stop *= 0.01

    return start, stop, state, conf


def write_opensat(file_vad, key, vad):
    file_dir = os.path.dirname(file_vad)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_vad, 'w') as f:
        start, stop, state, conf = bin2intervals(vad)
        for i in xrange(len(start)):
            f.write('X\tX\tX\tSAD\t%s\t%.2f\t%.2f\t%s\t%.2f\n' % (key, start[i], stop[i], 'speech' if state[i] else 'non-speech', conf[i]))
    

def h5vad2nist(input_file, output_dir):

    r = HypDataReader(input_file)
    keys = r.get_datasets()
    
    for key in keys:
        X = r.read([key])
        file_vad = output_dir + '/' + key + '.txt'
        write_opensat(file_vad, key, X[0])


if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Converts from VAD hdf5 file to NIST OpenSAT format')

    parser.add_argument('--input-file',dest='input_file', required=True)
    parser.add_argument('--output-dir', dest='output_dir', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    h5vad2nist(**vars(args))
    
