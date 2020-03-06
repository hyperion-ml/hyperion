"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .fgsm_attack import FGSMAttack
from .snr_fgsm_attack import SNRFGSMAttack
from .rand_fgsm_attack import RandFGSMAttack


class AttackFactory(object):

    @staticmethod
    def create(attack_type, model, attack_eps=0, attack_snr=100, attack_alpha=0, loss=None, 
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
                model, attack_eps, attack_alpha, loss=loss, targeted=targeted,
                range_min=range_min, range_max=range_max)

        raise Exception('%s is not a valid attack type' % (attack_type))


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        valid_args = ('attack_type', 'attack_eps', 'attack_snr', 'targeted')

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
            choices=['fgsm', 'snr-fgsm', 'rand-fgsm'], help=('Attack type'))

        parser.add_argument(
            p1+'attack-eps', default=0, type=float,
            help=('attack epsilon, upper bound for the perturbation norm'))

        parser.add_argument(
            p1+'attack-snr', default=100, type=float,
            help=('upper bound for the signal-to-noise ratio of the perturved signal'))

        parser.add_argument(p1+'targeted', default=False, action='store_true',
                            help='use targeted attack intead of non-targeted')
