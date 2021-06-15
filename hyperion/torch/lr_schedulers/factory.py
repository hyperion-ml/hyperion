"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

import torch

from .red_lr_on_plateau import ReduceLROnPlateau
from .exp_lr import ExponentialLR
from .invpow_lr import InvPowLR
from .cos_lr import CosineLR, AdamCosineLR


class LRSchedulerFactory(object):

    def create(optimizer, lrsch_type,
               decay_rate=1/100, decay_steps=100, 
               power=0.5,
               hold_steps=10,
               t=10, t_mul=1, 
               warm_restarts=False, gamma=1,
               monitor='val_loss', mode='min',
               factor=0.1, patience=10,
               threshold=1e-4, threshold_mode='rel',
               cooldown=0, eps=1e-8,
               min_lr=0, warmup_steps=0, update_lr_on_opt_step=False, **kwargs):

        if lrsch_type == 'none' or lrsch_type == 'dinossl' :
            return None
        
        if lrsch_type == 'exp_lr':
            return ExponentialLR(
                optimizer, decay_rate, decay_steps, hold_steps,
                min_lr=min_lr, warmup_steps=warmup_steps, 
                update_lr_on_opt_step=update_lr_on_opt_step)

        if lrsch_type == 'invpow_lr':
            return InvPowLR(
                optimizer, power, hold_steps,
                min_lr=min_lr, warmup_steps=warmup_steps, 
                update_lr_on_opt_step=update_lr_on_opt_step)


        if lrsch_type == 'cos_lr':
            return CosineLR(optimizer, t, t_mul, min_lr=min_lr,
                            warmup_steps=warmup_steps,
                            warm_restarts=warm_restarts, gamma=gamma,
                            update_lr_on_opt_step=update_lr_on_opt_step)

        if lrsch_type == 'adamcos_lr':
            return AdamCosineLR(optimizer, t, t_mul, warmup_steps=warmup_steps,
                            warm_restarts=warm_restarts, gamma=gamma,
                            update_lr_on_opt_step=update_lr_on_opt_step)

        if lrsch_type == 'red_lr_on_plateau':
            return ReduceLROnPlateau(
                optimizer, monitor, mode,
                factor=factor, patience=patience,
                threshold=threshold, threshold_mode=threshold_mode,
                cooldown=cooldown, min_lr=min_lr, warmup_steps=warmup_steps, eps=eps)
         

    @staticmethod
    def filter_args(**kwargs):

        valid_args = ('lrsch_type', 'decay_rate', 'decay_steps', 'hold_steps', 'power',
                      't', 't_mul', 'warm_restarts', 'gamma', 'monitor', 
                      'mode','factor','patience','threshold',
                      'threshold_mode','cooldown','eps','min_lr','warmup_steps','update_lr_on_opt_step',
                      'dinossl_lr','dinossl_min_lr','dinossl_warmup_epochs','dinossl_weight_decay',
                      'dinossl_weight_decay_end','dinossl_momentum_teacher')

        return dict((k, kwargs[k])
                    for k in valid_args if k in kwargs)
    

        
    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog='')

        parser.add_argument('--lrsch-type', type=str.lower,
                            default='none',
                            choices=['none', 'dinossl', 'exp_lr', 'invpow_lr', 'cos_lr', 'adamcos_lr', 'red_lr_on_plateau'],
                            help=('Learning rate schedulers: None, Exponential,'
                                  'Cosine Annealing, Cosine Annealing for Adam,' 
                                  'Reduce on Plateau'))

        parser.add_argument('--decay-rate' , 
                            default=1/100, type=float,
                            help=('LR decay rate in exp lr'))
        parser.add_argument('--decay-steps' ,
                            default=100, type=int,
                            help=('LR decay steps in exp lr'))
        parser.add_argument('--power' , 
                            default=0.5, type=float,
                            help=('power in inverse power lr'))

        parser.add_argument('--hold-steps' , 
                            default=10, type=int,
                            help=('LR hold steps in exp lr'))
        parser.add_argument('--t' , 
                            default=10, type=int,
                            help=('Period in cos lr'))
        parser.add_argument('--t-mul' , 
                            default=1, type=int,
                            help=('Period multiplicator for each restart in cos lr'))
        parser.add_argument('--gamma' , 
                            default=1/100, type=float,
                            help=('LR decay rate for each restart in cos lr'))

        parser.add_argument('--warm-restarts', default=False,
                            action='store_true',
                            help=('Do warm restarts in cos lr'))

        parser.add_argument('--monitor',  default='val_loss',
                            help=('Monitor metric to reduce lr'))
        parser.add_argument('--mode', default='min',
                            choices =['min','max'],
                            help=('Monitor metric mode to reduce lr'))

        parser.add_argument('--factor' , 
                            default=0.1, type=float,
                            help=('Factor by which the learning rate will be reduced on plateau'))

        parser.add_argument('--patience' , 
                            default=10, type=int,
                            help=('Number of epochs with no improvement after which learning rate will be reduced'))

        parser.add_argument('--threshold' , 
                            default=1e-4, type=float,
                            help=('Minimum metric improvement'))

        parser.add_argument('--threshold_mode', default='rel',
                            choices =['rel','abs'],
                            help=('Relative or absolute'))
        
        parser.add_argument('--cooldown' , 
                            default=0, type=int,
                            help=('Number of epochs to wait before resuming normal operation after lr has been reduced'))

        parser.add_argument('--eps' , 
                            default=1e-8, type=float,
                            help=('Minimum decay applied to lr'))

        parser.add_argument('--min-lr' , 
                            default=0, type=float,
                            help=('Minimum lr'))

        parser.add_argument('--warmup-steps' , 
                            default=0, type=int,
                            help=('Number of batches to warmup lr'))

        parser.add_argument('--update-lr-on-opt-step', default=False,
                            action='store_true',
                            help=('Update lr based on batch number instead of epoch number'))
        # dinossl related - start
        parser.add_argument('--dinossl_lr', default=0.005, type=float,
            help=("""Learning rate at the end of linear warmup (highest LR used during training). 
            The learning rate is linearly scaled with the batch size, and specified here for a 
            reference batch size of 256."""))
        parser.add_argument('--dinossl_min_lr' , 
                            default=1e-6, type=float,
            help=("Target LR at the end of optimization. We use a cosine LR schedule with linear warmup."))
        parser.add_argument('--dinossl_warmup_epochs' , 
                            default=10, type=int,
                        help=("Number of epochs for the linear learning-rate warm up."))
        parser.add_argument('--dinossl_weight_decay' , 
                            default=0.04, type=float,
            help=("Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well."))
        parser.add_argument('--dinossl_weight_decay_end' , 
                            default=0.4, type=float,
            help=("""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by 
            the end of training improves performance for ViTs."""))
        parser.add_argument('--dinossl_momentum_teacher' , 
                            default=0, type=float,
            help=("""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. 
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256."""))
        # dinossl related - end

        if prefix is not None:
            outer_parser.add_argument(
                '--' + prefix,
                action=ActionParser(parser=parser))
                # help='learning rate scheduler options')



    add_argparse_args = add_class_args
