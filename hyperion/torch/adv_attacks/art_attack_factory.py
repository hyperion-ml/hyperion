"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np
try:
    from art import attacks
except ImportError:
    pass

class ARTAttackFactory(object):

    @staticmethod
    def create(attack_type, model, attack_eps=0, attack_delta=0.01,
               attack_step_adapt=0.667, 
               attack_num_trial=25, attack_sample_size=20,
               attack_init_size=100,
               attack_norm=np.inf, attack_eps_step=0.1,
               attack_num_random_init=0, attack_minimal=False,
               attack_random_eps=False,
               attack_theta=0.1, attack_gamma=1.0,
               attack_etha=0.01,
               attack_confidence=0.0, attack_lr=1e-2, 
               attack_binary_search_steps=9, attack_max_iter=10,
               attack_c=1e-3, attack_max_halving=5, attack_max_doubling=5,
               targeted=False, num_samples=1):

        if attack_type == 'fgm' or attack_type == 'pgd':
            if attack_norm == 1:
                attack_eps = attack_eps * num_samples
                attack_eps_step = attack_eps_step * num_samples
            elif attack_norm == 2:
                attack_eps = attack_eps * np.sqrt(num_samples)
                attack_eps_step = attack_eps_step * np.sqrt(num_samples)


        if attack_type == 'boundary':
            return attacks.BoundaryAttack(
                model, targeted=targeted, delta=attack_delta,
                epsilon=attack_eps, step_adapt=attack_step_adapt,
                max_iter=attack_max_iter, num_trials=attack_num_trials,
                sample_size=attack_sample_size, init_size=attack_init_size)
                                          
        if attack_type == 'fgm':
            return attacks.FastGradientMethod(
                model, norm=attack_norm, eps=attack_eps, 
                eps_step=attack_eps_step,
                targeted=targeted,
                num_random_init=attack_num_random_init, minimal=attack_minimal)

        if attack_type == 'bim':
            return attacks.BasicIterativeMethod(
                model, eps=attack_eps, eps_step=attack_eps_step,
                max_iter=attack_max_iter,targeted=targeted)

        if attack_type == 'pgd':
            return attacks.ProjectedGradientDescent(
                model, norm=attack_norm, eps=attack_eps, 
                eps_step=attack_eps_step, max_iter=attack_max_iter,
                targeted=targeted, 
                num_random_init=attack_num_random_init, random_eps=attack_random_eps)

        if attack_type == 'jsma':
            return attacks.SaliencyMapMethod(
                model, theta=attack_theta, gamma=attack_gamma)

        if attack_type == 'newtonfool':
            return attacks.NewtonFool(
                model, theta=attack_eta, max_iter=attack_max_iter)


        if attack_type == 'cw-l2':
            return attacks.CarliniL2Method(
                model, attack_confidence, learning_rate=attack_lr, 
                binary_search_steps=attack_binary_search_steps, 
                max_iter=attack_max_iter, initial_const=attack_c,
                targeted=targeted,
                max_halving=attack_max_halving, max_doubling=attack_max_doubling)


        if attack_type == 'cw-linf':
            return attacks.CarliniLInfMethod(
                model, attack_confidence, learning_rate=attack_lr, 
                max_iter=attack_max_iter, targeted=targeted,
                max_halving=attack_max_halving, max_doubling=attack_max_doubling,
                eps=attack_eps)

        raise Exception('%s is not a valid attack type' % (attack_type))


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'

        if 'attack_norm' in kwargs:
            if kwargs['attack_norm'] == 'inf':
                kwargs['attack_norm'] = np.inf
            else:
                kwargs['attack_norm'] = int(kwargs['attack_norm'])


        valid_args = ('attack_type', 'attack_eps', 'attack_delta',
                      'attack_step_adapt', 
                      'attack_num_trial', 'attack_sample_size',
                      'attack_init_size', 'attack_norm',
                      'attack_eps_step', 
                      'attack_num_random_init', 'attack_minimal',
                      'attack_random_eps',
                      'attack_theta', 'attack_gamma', 'attack_etha',
                      'attack_confidence',
                      'attack_lr', 'attack_binary_search_steps',
                      'attack_max_iter', 
                      'attack_c', 'attack_max_halving', 'attack_max_doubling',
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
            choices=['boundary','fgm', 'bim', 'pgd', 'jsma', 'newtonfool', 'cw-l2', 'cw-linf'], 
            help=('Attack type'))

        parser.add_argument(
            p1+'attack-norm', type=str.lower, default='inf',
            choices=['inf','1','2'], help=('Attack norm'))

        parser.add_argument(
            p1+'attack-eps', default=0, type=float,
            help=('attack epsilon, upper bound for the perturbation norm'))

        parser.add_argument(
            p1+'attack-eps-step', default=0.1, type=float,
            help=('Step size of input variation for minimal perturbation computation'))

        parser.add_argument(
            p1+'attack-delta', default=0.1, type=float,
            help=('Initial step size for the orthogonal step in boundary-attack'))
        
        parser.add_argument(
            p1+'attack-step-adapt', default=0.667, type=float,
            help=('Factor by which the step sizes are multiplied or divided, '
                  'must be in the range (0, 1).'))
        
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
            p1+'attack-num-trial', default=25, type=int,
            help=('Maximum number of trials per iteration (boundary attack).'))

        parser.add_argument(
            p1+'attack-sample-size', default=20, type=int,
            help=('Number of samples per trial (boundary attack).'))

        parser.add_argument(
            p1+'attack-init-size', default=100, type=int,
            help=('Maximum number of trials for initial generation of '
                  'adversarial examples. (boundary attack).'))

        parser.add_argument(
            p1+'attack-num-random-init', default=0, type=int,
            help=('Number of random initialisations within the epsilon ball. '
                  'For random_init=0 starting at the original input.'))

        parser.add_argument(
            p1+'attack-minimal', default=False, action='store_true',
            help=('Indicates if computing the minimal perturbation (True). '
                  'If True, also define eps_step for the step size and eps '
                  'for the maximum perturbation.'))

        parser.add_argument(
            p1+'attack-random-eps', default=False, action='store_true',
            help=('When True, epsilon is drawn randomly from '
                  'truncated normal distribution. '
                  'The literature suggests this for FGSM based training to '
                  'generalize across different epsilons. eps_step is modified '
                  'to preserve the ratio of eps / eps_step. '
                  'The effectiveness of this method with PGD is untested'))


        parser.add_argument(
            p1+'attack-theta', default=0.1, type=float,
            help=('Amount of Perturbation introduced to each modified '
                  'feature per step (can be positive or negative).'))

        parser.add_argument(
            p1+'attack-gamma', default=1.0, type=float,
            help=('Maximum fraction of features being perturbed (between 0 and 1).'))

        parser.add_argument(
            p1+'attack-eta', default=0.01, type=float,
            help=('Eta coeff. for NewtonFool'))

        parser.add_argument(
            p1+'attack-c', default=1e-2, type=float,
            help=('initial weight of constraint function f in carlini-wagner attack'))

        parser.add_argument(
            p1+'attack-max-halving', default=5, type=int,
            help=('Maximum number of halving steps in the line search optimization.'))

        parser.add_argument(
            p1+'attack-max-doubling', default=5, type=int,
            help=('Maximum number of doubling steps in the line search optimization.'))


        parser.add_argument(
            p1+'targeted', default=False, action='store_true',
            help='use targeted attack intead of non-targeted')


