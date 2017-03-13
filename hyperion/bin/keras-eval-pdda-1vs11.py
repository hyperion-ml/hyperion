#!/usr/bin/env python
"""
Evals PDDA LLR
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse

import numpy as np

from hyperion.io import HypDataReader
from hyperion.transforms import LNorm
from hyperion.utils.scp_list import SCPList
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.utils.tensors import to3D_by_class
from hyperion.keras.keras_utils import *
from hyperion.keras.vae import TiedVAE_qYqZgY2 as TVAE


def load_data(hyp_reader, ndx_file, enroll_file, test_file,
              lnorm,
              model_idx, nb_model_parts, seg_idx, nb_seg_parts,
              eval_set):

    enroll = SCPList.load(enroll_file, sep='=')
    test = None
    if test_file is not None:
        test = SCPList.load(test_file, sep='=')
    ndx = None
    if ndx_file is not None:
        ndx = TrialNdx.load(ndx_file)

    ndx, enroll = TrialNdx.parse_eval_set(ndx, enroll, test, eval_set)
    if nb_model_parts > 1 or nb_seg_parts > 1:
        ndx = TrialNdx.split(model_idx, nb_model_parts, seg_idx, nb_seg_parts)
        enroll = enroll.filter(ndx.key)

        
    x_e = hyp_reader.read(enroll.file_path, '.ivec')
    x_t = hyp_reader.read(ndx.seg_set, '.ivec')
    
    if lnorm is not None:
        x_e = lnorm.predict(x_e)
        x_t = lnorm.predict(x_t)

    return x_e, x_t, ndx


def eval_pdda(iv_file, ndx_file, enroll_file, test_file,
              lnorm_file, model_file, score_file, eval_method, nb_samples_elbo, **kwargs):
    
    if lnorm_file is not None:
        lnorm = LNorm.load(lnorm_file)
    else:
        lnorm = None

    hr = HypDataReader(iv_file)
    x_e, x_t, ndx = load_data(hr, ndx_file, enroll_file, test_file, lnorm, **kwargs)

    model = TVAE.load(model_file)
    model.build()
    scores = model.eval_llr_1vs1(x_e, x_t, method=eval_method,
                                 nb_samples=nb_samples_elbo)

    s = TrialScores(ndx.model_set, ndx.seg_set, scores)
    s.save(score_file)
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Eval PDDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', default=None)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--test-file', dest='test_file', default=None)
    # parser.add_argument('--decoder-file', dest='decoder_file', required=True)
    # parser.add_argument('--qy-file', dest='qy_file', required=True)
    # parser.add_argument('--qz-file', dest='qz_file', required=True)
    parser.add_argument('--lnorm-file', dest='lnorm_file', default=None)
    parser.add_argument('--model-file', dest='model_file', required=True)
    parser.add_argument('--score-file', dest='score_file', required=True)

    # parser.add_argument('--batch-size',dest='batch_size',default=512,type=int,
    #                     help=('Batch size (default: %(default)s)'))

    parser.add_argument('--model-part-idx', dest='model_idx', default=1, type=int)
    parser.add_argument('--nb-model-parts', dest='nb_model_parts', default=1, type=int)
    parser.add_argument('--seg-part-idx', dest='seg_idx', default=1, type=int)
    parser.add_argument('--nb-seg-parts', dest='nb_seg_parts', default=1, type=int)

    parser.add_argument('--eval-set', dest='eval_set', type=str.lower,
                        default='enroll-test',
                        choices=['enroll-test','enroll-coh','coh-test','coh-coh'],
                        help=('(default: %(default)s)'))

    parser.add_argument('--eval-method', dest='eval_method', type=str.lower,
                        default='elbo',
                        choices=['elbo','cand','qscr'],
                        help=('(default: %(default)s)'))

    parser.add_argument('--nb-samples-elbo', dest='nb_samples_elbo', default=1, type=int)
    
    # parser.add_argument('--optimizer', dest='opt_type', type=str.lower,
    #                     default='adam',
    #                     choices=['sgd','nsgd','rmsprop','adam','nadam','adamax'],
    #                     help=('Optimizers: SGD, '
    #                           'NSGD (SGD with Nesterov momentum), '
    #                           'RMSprop, Adam, Adamax, '
    #                           'Nadam (Adam with Nesterov momentum), '
    #                           '(default: %(default)s)'))

    # parser.add_argument('--lr' , dest='lr',
    #                     default=0.002, type=float,
    #                     help=('Initial learning rate (default: %(default)s)'))
    # parser.add_argument('--momentum', dest='momentum', default=0.6, type=float,
    #                     help=('Momentum (default: %(default)s)'))
    # parser.add_argument('--lr-decay', dest='lr_decay', default=1e-6, type=float,
    #                     help=('Learning rate decay in SGD optimizer '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--rho', dest='rho', default=0.9, type=float,
    #                     help=('Rho in RMSprop optimizer (default: %(default)s)'))
    # parser.add_argument('--epsilon', dest='epsilon', default=1e-8, type=float,
    #                     help=('Epsilon in RMSprop and Adam optimizers '
    #                           '(default: %(default)s)'))
    # parser.add_argument('--beta1', dest='beta_1', default=0.9, type=float,
    #                     help=('Beta_1 in Adam optimizers (default: %(default)s)'))
    # parser.add_argument('--beta2', dest='beta_2', default=0.999, type=float,
    #                     help=('Beta_1 in Adam optimizers (default: %(default)s)'))
    # parser.add_argument('--schedule-decay', dest='schedule_decay',
    #                     default=0.004,type=float,
    #                     help=('Schedule decay in Nadam optimizer '
    #                           '(default: %(default)s)'))

    # parser.add_argument('--nb-epoch', dest='nb_epoch', default=1000, type=int)

    # parser.add_argument('--rng-seed', dest='rng_seed', default=1024, type=int,
    #                     help=('Seed for the random number generator '
    #                           '(default: %(default)s)'))
    
    args=parser.parse_args()

    assert(args.test_file is not None or args.ndx_file is not None)
    eval_pdda(**vars(args))

            
