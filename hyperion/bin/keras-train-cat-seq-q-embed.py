#!/usr/bin/env python

"""
Trains q-embeddings
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

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.utils.multithreading import threadsafe_generator
from hyperion.helpers import SequenceBatchGenerator as G
from hyperion.transforms import TransformList
from hyperion.keras.keras_utils import *
from hyperion.keras.keras_model_loader import KerasModelLoader as KML
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.embed.seq_q_embed import SeqQEmbed



@threadsafe_generator
def data_generator(sg, max_length):

    while 1:
        key, x, sample_weight, y = sg.read(max_seq_length=max_length)
        y_kl = np.zeros((y.shape[0], 1), dtype=float_keras())
        yield (x, [y, y_kl])


    
def train_embed(data_path, train_list, val_list,
                embed_file, 
                init_path,
                epochs,
                preproc_file, output_path,
                post_pdf, pooling_input, pooling_output,
                min_var, kl_weight, **kwargs):

    set_float_cpu(float_keras())

    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None
        
    sg_args = G.filter_args(**kwargs)
    sg = G(data_path, train_list,
           shuffle_seqs=True, reset_rng=False,
           transform=preproc, **sg_args)
    max_length = sg.max_seq_length
    gen_val = None
    if val_list is not None:
        sg_val = G(data_path, val_list,
                    transform=preproc,
                    shuffle_seqs=False, reset_rng=True,
                    **sg_args)
        max_length = max(max_length, sg_val.max_seq_length)
        gen_val = data_generator(sg_val, max_length)

    gen_train = data_generator(sg, max_length)

    
    if init_path is None:
        model, init_epoch = KML.load_checkpoint(output_path, epochs)
        if model is None:
            embed_net = load_model_arch(embed_file)

            model = SeqQEmbed(embed_net, num_classes=sg.num_classes,
                              post_pdf=post_pdf,
                              pooling_input=pooling_input,
                              pooling_output=pooling_output,
                              min_var=min_var, kl_weight=kl_weight)
        else:
            sg.cur_epoch = init_epoch
            sg.reset()
    else:
        print('loading init model: %s' % init_path)
        model = KML.load(init_path)


    opt_args = KOF.filter_args(**kwargs)
    cb_args = KCF.filter_args(**kwargs)
    print(sg_args)
    print(opt_args)
    print(cb_args)
    

    print('max length: %d' % max_length)

    t1 = time.time()    
    model.build(max_length)
    print(time.time()-t1)
    
    cb = KCF.create_callbacks(model, output_path, **cb_args)
    opt = KOF.create_optimizer(**opt_args)
    model.compile(optimizer=opt)

    h = model.fit_generator(gen_train, validation_data=gen_val,
                            steps_per_epoch=sg.steps_per_epoch,
                            validation_steps=sg_val.steps_per_epoch,
                            initial_epoch=sg.cur_epoch,
                            epochs=epochs, callbacks=cb, max_queue_size=10)

    print('Train elapsed time: %.2f' % (time.time() - t1))
    
    model.save(output_path + '/model')


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train sequence meta embeddings')

    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--embed-file', dest='embed_file', required=True)

    parser.add_argument('--init-path', dest='init_path', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--output-path', dest='output_path', required=True)

    parser.add_argument('--post-pdf', dest='post_pdf', default='diag_normal',
                        choices=['diag_normal'])
    parser.add_argument('--pooling-input', dest='pooling_input',
                        default='nat+logitvar',
                        choices=['nat+logitvar', 'nat+logprec-1',
                                 'nat+logvar', 'nat+logprec',
                                 'nat+var', 'nat+prec', 'nat+prec-1',
                                 'mean+logitvar', 'mean+logprec-1',
                                 'mean+logvar', 'mean+logprec',
                                 'mean+var', 'mean+prec', 'mean+prec-1'])
    parser.add_argument('--pooling-output', dest='pooling_output',
                        default='nat+prec',
                        choices=['nat+logar', 'nat+logprec',
                                 'nat+var', 'nat+prec',
                                 'mean+logar', 'mean+logprec',
                                 'mean+var', 'mean+prec'])
    
    parser.add_argument('--min-var', dest='min_var', default=0.9, type=float,
                        help=('Minimum frame variance (default: %(default)s)'))
    parser.add_argument('--kl-weight', dest='kl_weight', default=0, type=float,
                        help=('Weight of the KL divergence (default: %(default)s)'))

    G.add_argparse_args(parser)
    KOF.add_argparse_args(parser)
    KCF.add_argparse_args(parser)
    
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)
    
    args=parser.parse_args()
    
    train_embed(**vars(args))

            
