#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Evals sequence embeddings net.
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
from keras.layers import Input
from keras.models import Model

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as SDRF
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM
from hyperion.keras.keras_utils import *
from hyperion.keras.embed import SeqQEmbed


    
def eval_embed(seq_file, model_file, preproc_file, output_path,
               max_seq_length, **kwargs):

    set_float_cpu('float32')
    
    sr_args = SDRF.filter_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr = SDRF.create(seq_file, transform=preproc, **sr_args)
    
    t1 = time.time()

    model = SeqQEmbed.load(model_file)
    model.build(max_seq_length)
    model.build_eval()
    y_dim = model.num_classes

    _, seq_lengths = sr.read_num_rows()
    sr.reset()
    num_seqs = len(seq_lengths)

    y = np.zeros((num_seqs, y_dim), dtype=float_keras())
    keys = []

    for i in xrange(num_seqs):
        ti1 = time.time()
        key, data = sr.read(1)
        
        ti2 = time.time()
        logging.info('Extracting embeddings %d/%d for %s, num_frames: %d' %
                     (i, num_seqs, key[0], data[0].shape[0]))
        keys.append(key[0])
        y[i] = model.predict_eval(data[0])
                
        ti4 = time.time()
        logging.info('Elapsed time embeddings %d/%d for %s, total: %.2f read: %.2f, vae: %.2f' %
                     (i, num_seqs, key, ti4-ti1, ti2-ti1, ti4-ti2))
            
    logging.info('Extract elapsed time: %.2f' % (time.time() - t1))

    hw = DWF.create(output_path)
    hw.write(keys, y)

    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Evaluate q embeddings net')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--max-seq-length', dest='max_seq_length', default=None, type=int)

    SDRF.add_argparse_args(parser)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    eval_embed(**vars(args))

            
