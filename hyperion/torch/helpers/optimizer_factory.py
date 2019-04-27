"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.optim as optim


class OptimizerFactory(object):

    @staticmethod
    def create(params, opt_type, lr, momentum=0,
               beta1=0.9, beta2=0.99, rho=0.9, eps=1e-8, weight_decay=0,
               amsgrad=False, nesterov=False,
               lambd=0.0001, asgd_alpha=0.75, t0=1000000.0,
               rmsprop_alpha=0.99, centered=False,
               lr_decay=0, init_acc_val=0, max_iter=20):

        if opt_type == 'sgd':
            return optim.SGD(params, lr, momentum=momentum, dampening=0,
                             weight_decay=weight_decay, nesterov=nesterov)

        if opt_type == 'adam':
            return optim.Adam(
                params, lr, betas=(beta1, beta2), eps=eps,
                weight_decay=weight_decay, amsgrad=amsgrad)


        if opt_type == 'adadelta':
            return optim.Adadelta(params, lr, rho=rho, eps=eps,
                                  weight_decay=weight_decay)

        if opt_type == 'adagrad':
            return optim.Adagrad(
                params, lr, lr_decay=lr_decay,
                weight_decay=weight_decay, initial_accumulator_value=init_acc_val)

        
        if opt_type == 'sparse_adam':
            return optim.SparseAdam(params, lr, betas=(beta1, beta2), eps=eps)

        if opt_type == 'adamax':
            return optim.Adamax(params, lr, betas=(beta1, beta2), eps=eps,
                                weight_decay=weight_decay)

        if opt_type == 'asgd':
            return optim.ASGD(params, lr, lambd=lambd, alpha=asgd_alpha, t0=t0,
                              weight_decay=weight_decay)

        if opt_type == 'lbfgs':
            return optim.LBFGS(
                params, lr, max_iter=max_iter)

        if opt_type == 'rmsprop':
            return optim.RMSprop(
                params, lr, alpha=rmsprop_alpha, eps=eps,
                weight_decay=weight_decay, momentum=momentum, centered=centered)

        if opt_type == 'rprop':
            return optim.Rprop(params, lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('opt_type', 'lr', 'momentum', 'beta1', 'beta2',
                      'rho', 'eps', 'weight_decay', 'amsgrad', 'nesterov', 
                      'lambd','asgd_alpha','t0','rmsprop_alpha',
                      'centered','lr_decay','init_acc_val','max_iter')

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

        parser.add_argument(p1+'optimizer', dest=(p2+'opt_type'), type=str.lower,
                        default='adam',
                        choices=['sgd','adam', 'adadelta', 'adagrad', 'sparse_adam',
                                 'adamax', 'asgd', 'lbfgs', 'rmsprop','rprop'],
                        help=('Optimizers: SGD, Adam, AdaDelta, AdaGrad, SparseAdam '
                              'AdaMax, ASGD, LFGS, RMSprop, Rprop'))
        parser.add_argument(p1+'lr' , dest=(p2+'lr'),
                            default=0.001, type=float,
                            help=('Initial learning rate'))
        parser.add_argument(p1+'momentum', dest=(p2+'momentum'), default=0.6, type=float,
                            help=('Momentum'))
        parser.add_argument(p1+'beta1', dest=(p2+'beta1'), default=0.9, type=float,
                            help=('Beta_1 in Adam optimizers,  '
                                  'coefficient used for computing running averages of gradient'))
        parser.add_argument(p1+'beta2', dest=(p2+'beta2'), default=0.99, type=float,
                            help=('Beta_2 in Adam optimizers'
                                  'coefficient used for computing running averages of gradient square'))
        parser.add_argument(p1+'rho', dest=(p2+'rho'), default=0.9, type=float,
                            help=('Rho in AdaDelta,' 
                                  'coefficient used for computing a running average of squared gradients'))
        parser.add_argument(p1+'eps', dest=(p2+'eps'), default=1e-8, type=float,
                            help=('Epsilon in RMSprop and Adam optimizers '
                                  'term added to the denominator to improve numerical stability'))

        parser.add_argument(p1+'weight-decay', dest=(p2+'weight_decay'), default=1e-6, type=float,
                            help=('L2 regularization coefficient'))

        parser.add_argument(p1+'amsgrad', dest=(p2+'amsgrad'), default=False,
                            action='store_true',
                            help=('AMSGrad variant of Adam'))

        parser.add_argument(p1+'nesterov', dest=(p2+'nesterov'), default=False,
                            action='store_true',
                            help=('Use Nesterov momentum in SGD'))

        parser.add_argument(p1+'lambd', dest=(p2+'lambd'), default=0.0001, type=float,
                            help=('decay term in ASGD'))

        parser.add_argument(p1+'asgd-alpha', dest=(p2+'asgd_alpha'), default=0.75, type=float,
                            help=('power for eta update in ASGD'))

        parser.add_argument(p1+'t0', dest=(p2+'t0'), default=1e6, type=float,
                            help=('point at which to start averaging in ASGD'))

        parser.add_argument(p1+'rmsprop-alpha', dest=(p2+'rmsprop_alpha'), default=0.99, type=float,
                            help=('smoothing constant in RMSprop'))
        
        parser.add_argument(p1+'centered', dest=(p2+'centered'), default=False,
                            action='store_true',
                            help=('Compute centered RMSprop, gradient normalized by its variance'))

        parser.add_argument(p1+'lr-decay', dest=(p2+'lr_decay'), default=1e-6, type=float,
                            help=('Learning rate decay in AdaGrad optimizer'))
    
        parser.add_argument(p1+'init-acc-val', dest=(p2+'init_acc_val'), default=0, type=float,
                            help=('Init accum value in Adagrad'))

        parser.add_argument(p1+'max-iter', dest=(p2+'max_iter'), default=20, type=int,
                            help=('max iterations in LBGS'))
