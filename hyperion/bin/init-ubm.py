#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

"""
Initialize UBM
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import logging

import numpy as np

from keras import backend as K

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.utils.multithreading import threadsafe_generator
from hyperion.helpers import SequenceReader as SR
from hyperion.pdfs import DiagGMM


@threadsafe_generator
def data_generator(sr, max_length):
    kk=0
    while 1:
        kk+=1
        x, sample_weights = sr.read(return_3d=True, max_seq_length=max_length)
        return_sw = True
        if sr.max_batch_seq_length==max_length and (
                sr.min_seq_length==sr.max_seq_length or
                np.min(sr.seq_length)==sr.max_seq_length):
            return_sw = False
                                      
        if return_sw:
            yield (x, x, sample_weights)
        else:
            yield (x, x)

    
def init_ubm(seq_file, train_list, x_dim, num_comp,
             output_path, **kwargs):

    if seq_file is None:
        model = DiagGMM(x_dim=x_dim, num_comp=1)
        model.initialize()
        model.save(output_path)

        
    sr_args = SR.filter_args(**kwargs)
    sr = SR(seq_file, train_list, batch_size=1, **sr_args)


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Initializes UBM')

    parser.add_argument('--seq-file', dest='seq_file', default=None)
    parser.add_argument('--train-list', dest='train_list', default=None)
    parser.add_argument('--x-dim', dest='x_dim', type=int, required=True)
    parser.add_argument('--num-comp', dest='num_comp', default=1)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
        
    SR.add_argparse_args(parser)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    init_ubm(**vars(args))

            
