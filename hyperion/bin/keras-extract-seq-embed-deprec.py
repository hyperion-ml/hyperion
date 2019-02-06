#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Extracts sequence embeddings
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time

import numpy as np

import gc

from keras import backend as K
from keras.layers import Input
from keras.models import Model

from hyperion.hyp_defs import set_float_cpu, float_cpu, config_logger
from hyperion.io import HypDataWriter
from hyperion.helpers import SequenceReader as SR
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM
from hyperion.keras.keras_utils import *
from hyperion.keras.embed.seq_embed import SeqEmbed



    
def extract_embed(seq_file, file_list, model_file, preproc_file, output_path,
                  max_length, layer_names, **kwargs):

    set_float_cpu('float32')
    
    sr_args = SR.filter_eval_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    sr = SR(seq_file, file_list, batch_size=1,
            shuffle_seqs=False,
            preproc=preproc, **sr_args)
    
    t1 = time.time()

    model = SeqEmbed.load(model_file)
    model.build()
    print(layer_names)
    model.build_embed(layer_names)
    y_dim = model.embed_dim

    max_length = np.minimum(sr.max_batch_seq_length, max_length)

    y = np.zeros((sr.num_seqs, y_dim), dtype=float_keras())
    xx = np.zeros((1, max_length, model.x_dim), dtype=float_keras())
    keys = []

    for i in xrange(sr.num_seqs):
        ti1 = time.time()
        x, key = sr.read_next_seq()
        ti2 = time.time()
        print('Extracting embeddings %d/%d for %s, num_frames: %d' % (i, sr.num_seqs, key, x.shape[0]))
        keys.append(key)
        xx[:,:,:] = 0
        
        if x.shape[0] <= max_length:
            xx[0,:x.shape[0]] = x
            y[i] = model.predict_embed(xx, batch_size=1)
        else:
            num_chunks = int(np.ceil(float(x.shape[0])/max_length))
            chunk_size = int(np.ceil(float(x.shape[0])/num_chunks))
            for j in xrange(num_chunks-1):
                start = j*chunk_size
                xx[0,:chunk_size] = x[start:start+chunk_size]
                y[i] += model.predict_embed(xx, batch_size=1).ravel()
            xx[0,:chunk_size] = x[-chunk_size:]
            y[i] += model.predict_embed(xx, batch_size=1).ravel()
            y[i] /= num_chunks
                
        ti4 = time.time()
        print('Elapsed time embeddings %d/%d for %s, total: %.2f read: %.2f, vae: %.2f' %
              (i, sr.num_seqs, key, ti4-ti1, ti2-ti1, ti4-ti2))
            
    print('Extract elapsed time: %.2f' % (time.time() - t1))
    
    hw = HypDataWriter(output_path)
    hw.write(keys, '', y)

    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Extract embeddings')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--file-list', dest='file_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--layer-names', dest='layer_names', required=True, nargs='+')
    parser.add_argument('--max-length', dest='max_length', default=60000, type=int)

    SR.add_argparse_eval_args(parser)

    # parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
    #                     help=('Batch size (default: %(default)s)'))
    args=parser.parse_args()
    
    extract_embed(**vars(args))

            
