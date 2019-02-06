#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
"""
Trains x-vectors with attention
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
from hyperion.helpers import SequenceClassReader as SR
from hyperion.transforms import TransformList
from hyperion.keras.keras_utils import *
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.embed.seq_embed_att import SeqEmbedAtt as SeqEmbed



@threadsafe_generator
def data_generator(sr, max_length):
    kk=0
    while 1:
        kk+=1
        # print('dg %d.' % kk)
        x, sample_weight, _, y = sr.read(return_3d=True, max_seq_length=max_length)

        return_sw = True
        if sr.max_batch_seq_length == max_length and (
                sr.min_seq_length == sr.max_seq_length or
                np.min(sr.seq_length) == sr.max_seq_length):
            return_sw = False
                                      
        if return_sw:
            yield (x, y, sample_weight)
        else:
            yield (x, y)

    
def train_embed(seq_file, train_list, val_list,
                class_list,
                embed1_file, embed2_file, att_file,
                init_path,
                epochs, batch_size,
                preproc_file, output_path,
                pooling, **kwargs):
    
    set_float_cpu(float_keras())
    
    sr_args = SR.filter_args(**kwargs)
    sr_val_args = SR.filter_val_args(**kwargs)
    opt_args = KOF.filter_args(**kwargs)
    cb_args = KCF.filter_args(**kwargs)
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

        
    sr = SR(seq_file, train_list, class_list, batch_size=batch_size,
            preproc=preproc, **sr_args)
    max_length = sr.max_batch_seq_length
    gen_val = None
    if val_list is not None:
        sr_val = SR(seq_file, val_list, class_list, batch_size=batch_size,
                    preproc=preproc,
                    shuffle_seqs=False,
                    seq_split_mode='sequential', seq_split_overlap=0,
                    reset_rng=True,
                    **sr_val_args)
        max_length = max(max_length, sr_val.max_batch_seq_length)
        gen_val = data_generator(sr_val, max_length)

    gen_train = data_generator(sr, max_length)
    
    t1 = time.time()
    if init_path is None:
        embed_net1 = load_model_arch(embed1_file)
        embed_net2 = load_model_arch(embed2_file)
        att_net = load_model_arch(att_file)

        model = SeqEmbed(embed_net1, embed_net2, att_net,
                         pooling=pooling, loss='categorical_crossentropy')
    else:
        logging.info('loading init model: %s' % init_path)
        model = SeqEmbed.load(init_path)

    logging.info('max length: %d' % max_length)
    model.build(max_length)
    logging.info(time.time()-t1)
    
    cb = KCF.create_callbacks(model, output_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)
    model.compile(optimizer=opt)
    
    h = model.fit_generator(gen_train, validation_data=gen_val,
                            steps_per_epoch=sr.num_batches,
                            validation_steps=sr_val.num_batches,
                            epochs=epochs, callbacks=cb, max_queue_size=10)
                          
    logging.info('Train elapsed time: %.2f' % (time.time() - t1))
    
    model.save(output_path + '/model')


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train sequence embeddings')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--class-list', dest='class_list', required=True)
    parser.add_argument('--embed1-file', dest='embed1_file', required=True)
    parser.add_argument('--embed2-file', dest='embed2_file', required=True)
    parser.add_argument('--att-file', dest='att_file', required=True)

    parser.add_argument('--init-path', dest='init_path', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--output-path', dest='output_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
                        help=('Batch size (default: %(default)s)'))

    parser.add_argument('--pooling', dest='pooling', default='mean+std',
                        choices=['mean+std', 'mean'])
    
    SR.add_argparse_args(parser)
    KOF.add_argparse_args(parser)
    KCF.add_argparse_args(parser)
    
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
        
    train_embed(**vars(args))

            
