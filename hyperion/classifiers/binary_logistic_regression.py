from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from .logistic_regression import LogisticRegression

class BinaryLogisticRegression(LogisticRegression):

    def __init__(self, A=None, b=None, penalty='l2', C=1.0,
                 use_bias=True, prior=0.5,
                 random_state=None, solver='liblinear', max_iter=100,
                 dual=False, tol=0.0001, verbose=0, warm_start=True,
                 lr_seed=1024, **kwargs):
        
        if use_bias and solver == 'liblinear':
            bias_scaling = 10/C
        else:
            bias_scaling = 1
        self.prior = prior
        class_weight = {0:1-prior, 1:prior}
        super(BinaryLogisticRegression, self).__init__(
            A=A, b=b, penalty=penalty, C=C,
            use_bias=use_bias, bias_scaling=bias_scaling,
            class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            dual=dual, tol=tol, verbose=verbose, warm_start=warm_start,
            balance_class_weight=False, lr_seed=1024, **kwargs)


    def fit(self, x, class_ids, sample_weight=None):
        if x.ndim == 1:
            x = x[:, None]
        self.lr.fit(x, class_ids, sample_weight=sample_weight)
        self.lr.intercept_ -= np.log(self.prior/(1-self.prior))/self.bias_scaling
        

    
    @staticmethod
    def filter_train_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('penalty', 'C',
                      'use_bias', 'bias_scaling',
                      'class_weight', 'lr_seed',
                      'solver', 'max_iter',
                      'dual', 'tol', 'verbose',
                      'warm_start', 
                      'prior', 'name')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)


    
    @staticmethod
    def add_argparse_train_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'penalty', dest=(p2+'penalty'), 
                            default='l2', choices=['l2', 'l1'],
                            help='used to specify the norm used in the penalization')
        parser.add_argument(p1+'c', dest=(p2+'C'), 
                            default=1.0, type=float,
                            help='inverse of regularization strength')
        parser.add_argument(p1+'no-use-bias', dest=(p2+'no_use_bias'),
                            default=False, action='store_true',
                            help='Not use bias')
        parser.add_argument(p1+'lr-seed', dest=(p2+'lr_seed'), 
                            default=1024, type=int,
                            help='random number generator seed')
        parser.add_argument(p1+'solver', dest=(p2+'solver'), 
                            default='liblinear', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                            help='type of solver')
        parser.add_argument(p1+'max-iter', dest=(p2+'max_iter'), 
                            default=100, type=int,
                            help='only for the newton-cg, sag and lbfgs solvers')
        parser.add_argument(p1+'dual', dest=(p2+'dual'),
                            default=False, action='store_true',
                            help=('dual or primal formulation. '
                                  'Dual formulation is only implemented for l2 penalty with liblinear solver'))
        parser.add_argument(p1+'tol', dest=(p2+'tol'), default=1e-4, type=float,
                            help='tolerance for stopping criteria')
        parser.add_argument(p1+'verbose', dest=(p2+'verbose'), 
                            default=0, type=int,
                            help='For the liblinear and lbfgs solvers')
        parser.add_argument(p1+'no-warm-start', dest=(p2+'no_warm_start'),
                            default=False, action='store_true',
                            help='use previous model to start')

        parser.add_argument(p1+'prior', dest=(p2+'prior'),
                            default=0.1, type=float,
                            help='Target prior')

        parser.add_argument(p1+'name', dest=(p2+'name'), 
                            default='lr',
                            help='model name')


    
    
    @staticmethod
    def add_argparse_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'model-file', dest=(p2+'model_file'), required=True,
                            help=('model file'))
        parser.add_argument(p1+'eval-type', dest=(p2+'eval_type'), default='logit',
                            choices=['logit','bin-logpost','bin-post'],
                            help=('type of evaluation'))
    
