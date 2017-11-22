#!/usr/bin/env python

"""
Trains TCVAE
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
import scipy.stats as scps

from keras import backend as K

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.utils.multithreading import threadsafe_generator
from hyperion.helpers import SequenceClassReader as SR
from hyperion.transforms import TransformList
from hyperion.pdfs import DiagGMM
from hyperion.keras.keras_utils import *
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.embed.seq_q_embed import SeqQEmbed



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

        y_kl = np.zeros((y.shape[0], 1), dtype=float_keras())
        if return_sw:
            yield (x, [y, y_kl], sample_weight)
        else:
            yield (x, [y, y_kl])

    
def train_embed(seq_file, train_list, val_list,
                class_list,
                embed_file, 
                init_path,
                epochs, batch_size,
                preproc_file, output_path,
                post_pdf, pooling_input, pooling_output,
                min_var, kl_weight, **kwargs):

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
        embed_net = load_model_arch(embed_file)

        model = SeqQEmbed(embed_net, num_classes=sr.num_classes, post_pdf=post_pdf,
                          pooling_input=pooling_input, pooling_output=pooling_output,
                          min_var=min_var, kl_weight=kl_weight)
    else:
        print('loading init model: %s' % init_path)
        model = SeqQEmbed.load(init_path)

    print('max length: %d' % max_length)
    model.build(max_length)
    print(time.time()-t1)
    
    cb = KCF.create_callbacks(model, output_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)
    model.compile(optimizer=opt)

    # h = model.fit_generator(gen_train, validation_data=gen_val,
    #                         steps_per_epoch=1,
    #                         validation_steps=1,
    #                         epochs=epochs, callbacks=cb, max_queue_size=10)

    h = model.fit_generator(gen_train, validation_data=gen_val,
                            steps_per_epoch=sr.num_batches,
                            validation_steps=sr_val.num_batches,
                            epochs=epochs, callbacks=cb, max_queue_size=10)
                          
    print('Train elapsed time: %.2f' % (time.time() - t1))
    
    model.save(output_path + '/model')


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train sequence meta embeddings')

    parser.add_argument('--seq-file', dest='seq_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--class-list', dest='class_list', required=True)
    parser.add_argument('--embed-file', dest='embed_file', required=True)

    parser.add_argument('--init-path', dest='init_path', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--output-path', dest='output_path', required=True)

    parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
                        help=('Batch size (default: %(default)s)'))

    parser.add_argument('--post-pdf', dest='post_pdf', default='diag_normal',
                        choices=['diag_normal'])
    parser.add_argument('--pooling-input', dest='pooling_input', default='nat+logitvar',
                        choices=['nat+logitvar', 'nat+logprec-1', 'nat+logvar', 'nat+logprec', 'nat+var', 'nat+prec', 'nat+prec-1',
                                 'mean+logitvar', 'mean+logprec-1', 'mean+logvar', 'mean+logprec', 'mean+var', 'mean+prec', 'mean+prec-1'])
    parser.add_argument('--pooling-output', dest='pooling_output', default='nat+prec',
                        choices=['nat+logar', 'nat+logprec', 'nat+var', 'nat+prec',
                                 'mean+logar', 'mean+logprec', 'mean+var', 'mean+prec'])
    
    parser.add_argument('--min-var',dest='min_var',default=0.9,type=float,
                        help=('Minimum frame variance (default: %(default)s)'))
    parser.add_argument('--kl-weight',dest='kl_weight',default=0.1,type=float,
                        help=('Weight of the KL divergence (default: %(default)s)'))

    SR.add_argparse_args(parser)
    KOF.add_argparse_args(parser)
    KCF.add_argparse_args(parser)
    
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)
    
    args=parser.parse_args()
    
    train_embed(**vars(args))

            
