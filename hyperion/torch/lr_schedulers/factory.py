"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import

import torch

from .red_lr_on_plateau import ReduceLROnPlateau
from .exp_lr import ExponentialLR
from .cos_lr import CosineLR, AdamCosineLR


class LRSchedulerFactory(object):

    def create(optimizer, lrsch_type,
               decay_rate=1/100, decay_steps=100, hold_steps=10,
               T=10, T_mul=1, 
               warm_restarts=False, gamma=1,
               monitor='val_loss', mode='min',
               factor=0.1, patience=10,
               threshold=1e-4, threshold_mode='rel',
               cooldown=0, eps=1e-8,
               min_lr=0, warmup_steps=0, update_lr_on_batch=False):

        if lrsch_type == 'none':
            return None
        
        if lrsch_type == 'exp_lr':
            return ExponentialLR(
                optimizer, decay_rate, decay_steps, hold_steps,
                min_lr=min_lr, warmup_steps=warmup_steps, 
                update_lr_on_batch=False)

        if lrsch_type == 'cos_lr':
            return CosineLR(optimizer, T, T_mul, min_lr=min_lr,
                            warmup_steps=warmup_steps,
                            warm_restarts=warm_restarts, gamma=gamma,
                            update_lr_on_batch=update_lr_on_batch)

        if lrsch_type == 'adamcos_lr':
            return AdamCosineLR(optimizer, T, T_mul, warmup_steps=warmup_steps,
                            warm_restarts=warm_restarts, gamma=gamma,
                            update_lr_on_batch=update_lr_on_batch)

        if lrsch_type == 'red_lr_on_plateau':
            return ReduceLROnPlateau(
                optimizer, monitor, mode,
                factor=factor, patience=patience,
                threshold=threshold, threshold_mode=threshold_mode,
                cooldown=cooldown, min_lr=min_lr, warmup_steps=warmup_steps, eps=eps)
         

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'


        valid_args = ('lrsch_type', 'decay_rate', 'decay_steps', 'hold_steps',
                      'T', 'T_mul', 'warm_restarts', 'gamma', 'monitor', 
                      'mode','factor','patience','threshold',
                      'threshold_mode','cooldown','eps','min_lr', 'warmup_steps', 'update_lr_on_batch')

        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)
    

        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'lrsch-type', dest=(p2+'lrsch_type'), type=str.lower,
                            default='none',
                            choices=['none','exp_lr', 'cos_lr', 'adamcos_lr', 'red_lr_on_plateau'],
                            help=('Learning rate schedulers: None, Exponential,'
                                  'Cosine Annealing, Cosine Annealing for Adam,' 
                                  'Reduce on Plateau'))

        parser.add_argument(p1+'decay-rate' , dest=(p2+'decay_rate'),
                            default=1/100, type=float,
                            help=('LR decay rate in exp lr'))
        parser.add_argument(p1+'decay-steps' , dest=(p2+'decay_steps'),
                            default=100, type=int,
                            help=('LR decay steps in exp lr'))
        parser.add_argument(p1+'hold-steps' , dest=(p2+'hold_steps'),
                            default=10, type=int,
                            help=('LR hold steps in exp lr'))
        parser.add_argument(p1+'t' , dest=(p2+'T'),
                            default=10, type=int,
                            help=('Period in cos lr'))
        parser.add_argument(p1+'t-mul' , dest=(p2+'T_mul'),
                            default=1, type=int,
                            help=('Period multiplicator for each restart in cos lr'))
        parser.add_argument(p1+'gamma' , dest=(p2+'gamma'),
                            default=1/100, type=float,
                            help=('LR decay rate for each restart in cos lr'))

        parser.add_argument(p1+'warm-restarts', dest=(p2+'warm_restarts'), default=False,
                            action='store_true',
                            help=('Do warm restarts in cos lr'))

        parser.add_argument(p1+'monitor', dest=(p2+'monitor'), default='val_loss',
                            help=('Monitor metric to reduce lr'))
        parser.add_argument(p1+'mode', dest=(p2+'mode'), default='min',
                            choices =['min','max'],
                            help=('Monitor metric mode to reduce lr'))

        parser.add_argument(p1+'factor' , dest=(p2+'factor'),
                            default=0.1, type=float,
                            help=('Factor by which the learning rate will be reduced on plateau'))

        parser.add_argument(p1+'patience' , dest=(p2+'patience'),
                            default=10, type=int,
                            help=('Number of epochs with no improvement after which learning rate will be reduced'))

        parser.add_argument(p1+'threshold' , dest=(p2+'threshold'),
                            default=1e-4, type=float,
                            help=('Minimum metric improvement'))

        parser.add_argument(p1+'threshold_mode', dest=(p2+'threshold_mode'), default='rel',
                            choices =['rel','abs'],
                            help=('Relative or absolute'))
        
        parser.add_argument(p1+'cooldown' , dest=(p2+'cooldown'),
                            default=0, type=int,
                            help=('Number of epochs to wait before resuming normal operation after lr has been reduced'))

        parser.add_argument(p1+'eps' , dest=(p2+'eps'),
                            default=1e-8, type=float,
                            help=('Minimum decay applied to lr'))

        parser.add_argument(p1+'min-lr' , dest=(p2+'min_lr'),
                            default=0, type=float,
                            help=('Minimum lr'))

        parser.add_argument(p1+'warmup-steps' , dest=(p2+'warmup_steps'),
                            default=0, type=int,
                            help=('Number of batches to warmup lr'))

        parser.add_argument(p1+'update-lr-on-batch', dest=(p2+'update_lr_on_batch'), default=False,
                            action='store_true',
                            help=('Update lr based on batch number instead of epoch number'))
