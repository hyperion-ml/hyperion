"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .fgsm_attack import FGSMAttack
from .snr_fgsm_attack import SNRFGSMAttack
from .rand_fgsm_attack import RandFGSMAttack
from .iter_fgsm_attack import IterFGSMAttack
from .carlini_wagner_l2 import CarliniWagnerL2
from .carlini_wagner_l0 import CarliniWagnerL0
from .carlini_wagner_linf import CarliniWagnerLInf
from .pgd_attack import PGDAttack

class AttackFactory(object):

    @staticmethod
    def create(attack_type, model, attack_eps=0, attack_snr=100, attack_alpha=0, 
               attack_norm=float('inf'), attack_random_eps=False, attack_num_random_init=0,
               attack_confidence=0.0, attack_lr=1e-2, 
               attack_binary_search_steps=9, attack_max_iter=10,
               attack_abort_early=True, attack_c=1e-3,
               attack_reduce_c=False, attack_c_incr_factor=2,
               attack_tau_decr_factor=0.9,
               attack_indep_channels=False,
               attack_norm_time=False, time_dim=None,
               attack_use_snr=False,
               loss=None, 
               targeted=False, range_min=None, range_max=None):

        if attack_type == 'fgsm':
            return FGSMAttack(
                model, attack_eps, loss=loss, targeted=targeted,
                range_min=range_min, range_max=range_max)

        if attack_type == 'snr-fgsm':
            return SNRFGSMAttack(
                model, attack_snr, loss=loss, targeted=targeted,
                range_min=range_min, range_max=range_max)

        if attack_type == 'rand-fgsm':
            return RandFGSMAttack(
                model, attack_eps, attack_alpha, loss=loss, 
                targeted=targeted, range_min=range_min, range_max=range_max)

        if attack_type == 'iter-fgsm':
            return IterFGSMAttack(
                model, attack_eps, attack_alpha, loss=loss, 
                targeted=targeted, range_min=range_min, range_max=range_max)

        if attack_type == 'cw-l2':
            return CarliniWagnerL2(
                model, attack_confidence, attack_lr, 
                attack_binary_search_steps, attack_max_iter, 
                attack_abort_early, attack_c, 
                norm_time=attack_norm_time, time_dim=time_dim, 
                use_snr=attack_use_snr,
                targeted=targeted, range_min=range_min, range_max=range_max)

        if attack_type == 'cw-l0':
            return CarliniWagnerL0(
                model, attack_confidence, attack_lr, attack_max_iter, 
                attack_abort_early, attack_c,
                attack_reduce_c, attack_c_incr_factor, attack_indep_channels,
                targeted=targeted, range_min=range_min, range_max=range_max)

        if attack_type == 'cw-linf':
            return CarliniWagnerLInf(
                model, attack_confidence, attack_lr, attack_max_iter, 
                attack_abort_early, attack_c,
                attack_reduce_c, attack_c_incr_factor, attack_tau_decr_factor,
                targeted=targeted, range_min=range_min, range_max=range_max)


        if attack_type == 'pgd':
            return PGDAttack(
                model, attack_eps, attack_alpha, attack_norm, attack_max_iter, 
                attack_random_eps, attack_num_random_init, loss=loss, 
                targeted=targeted, range_min=range_min, range_max=range_max)
                
        raise Exception('%s is not a valid attack type' % (attack_type))



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if 'attack_no_abort' in kwargs:
            kwargs['attack_abort_early'] = not kwargs['attack_no_abort']

        if 'attack_norm' in kwargs:
            if isinstance(kwargs['attack_norm'], str):
                kwargs['attack_norm'] == float(kwargs['attack_norm'])

        valid_args = ('attack_type', 'attack_eps', 'attack_snr', 
                      'attack_norm', 'attack_random_eps', 'attack_num_random_init',
                      'attack_alpha', 'attack_confidence',
                      'attack_lr', 'attack_binary_search_steps',
                      'attack_max_iter', 'attack_abort_early',
                      'attack_c', 'attack_reduce_c', 
                      'attack_c_incr_factor', 'attack_tau_decr_factor',
                      'attack_indep_channels', 'attack_use_snr', 'attack_norm_time',
                      'targeted')

        args = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        return args



    @staticmethod
    def add_argparse_args(parser, prefix=None):
        
        if prefix is None:
            p1 = '--'
        else:
            p1 = '--' + prefix + '-'

        parser.add_argument(
            p1+'attack-type', type=str.lower, default='fgsm',
            choices=['fgsm', 'snr-fgsm', 'rand-fgsm', 'iter-fgsm', 'cw-l0', 'cw-l2', 'cw-linf', 'pgd'], 
            help=('Attack type'))

        parser.add_argument(
            p1+'attack-norm', type=float, default=float('inf'),
            choices=[float('inf'), 1, 2],  help=('Attack perturbation norm'))


        parser.add_argument(
            p1+'attack-eps', default=0, type=float,
            help=('attack epsilon, upper bound for the perturbation norm'))

        parser.add_argument(
            p1+'attack-snr', default=100, type=float,
            help=('upper bound for the signal-to-noise ratio of the perturved signal'))

        parser.add_argument(
            p1+'attack-alpha', default=0, type=float,
            help=('alpha for iter and rand fgsm attack'))

        parser.add_argument(
            p1+'attack-random-eps', default=False, action='store_true',
            help=('use random epsilon in PGD attack'))

        parser.add_argument(
            p1+'attack-confidence', default=0, type=float,
            help=('confidence for carlini-wagner attack'))

        parser.add_argument(
            p1+'attack-lr', default=1e-2, type=float,
            help=('learning rate for attack optimizers'))

        parser.add_argument(
            p1+'attack-binary-search-steps', default=9, type=int,
            help=('num bin. search steps in carlini-wagner-l2 attack'))

        parser.add_argument(
            p1+'attack-max-iter', default=10, type=int,
            help=('max. num. of optim iters in attack'))

        parser.add_argument(
            p1+'attack-c', default=1e-2, type=float,
            help=('initial weight of constraint function f in carlini-wagner attack'))

        parser.add_argument(
            p1+'attack-reduce-c', default=False, action='store_true',
            help=('allow to reduce c in carline-wagner-l0/inf attack'))

        parser.add_argument(
            p1+'attack-c-incr-factor', default=2, type=float,
            help=('factor to increment c in carline-wagner-l0/inf attack'))

        parser.add_argument(
            p1+'attack-tau-decr-factor', default=0.75, type=float,
            help=('factor to reduce tau in carline-wagner-linf attack'))

        parser.add_argument(
            p1+'attack-indep-channels', default=False, action='store_true',
            help=('consider independent input channels in carline-wagner-l0 attack'))

        parser.add_argument(
            p1+'attack-no-abort', default=False, action='store_true',
            help=('do not abort early in optimizer iterations'))

        parser.add_argument(
            p1+'attack-num-random-init', default=0, type=int,
            help=('number of random initializations in PGD attack'))

        parser.add_argument(
            p1+'targeted', default=False, action='store_true',
            help='use targeted attack intead of non-targeted')

        parser.add_argument(
            p1+'attack-use-snr', default=False, action='store_true',
            help=('In carlini-wagner attack maximize SNR instead of minimize perturbation norm'))

        parser.add_argument(
            p1+'attack-norm-time', default=False, action='store_true',
            help=('normalize norm by number of samples in time dimension'))
