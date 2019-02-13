"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from .logistic_regression import LogisticRegression

class BinaryLogisticRegression(LogisticRegression):

    def __init__(self, A=None, b=None, penalty='l2', lambda_reg=1e-6,
                 use_bias=True, bias_scaling=1, prior=0.5,
                 random_state=None, solver='liblinear', max_iter=100,
                 dual=False, tol=0.0001, verbose=0, warm_start=True,
                 lr_seed=1024, **kwargs):
        
        priors = {0:1-prior, 1:prior}
        super(BinaryLogisticRegression, self).__init__(
            A=A, b=b, penalty=penalty, labmda_reg=lambda_reg,
            use_bias=use_bias, bias_scaling=bias_scaling, priors=priors,
            random_state=random_state, solver=solver, max_iter=max_iter,
            dual=dual, tol=tol, verbose=verbose, warm_start=warm_start,
            multi_class='ovr', lr_seed=1024, **kwargs)



    @property
    def prior(self):
        return self.priors[1]


    
    def get_config(self):
        config = {'prior': self.prior }
        base_config = super(BinaryLogisticRegression, self).get_config()
        del base_config['priors']
        return dict(list(base_config.items()) + list(config.items()))

    

    def predict(self, x, eval_type='logit'):
        if x.ndim == 1:
            x = x[:, None]
        
        y = np.dot(x, self.A).ravel() + self.b

        if eval_type == 'log-post':
            y = - np.log(1+np.exp(-(y+np.log(self.prior/(1-self.prior)))))
        if eval_type == 'post':
            y = 1/(1+np.exp(-(y+np.log(self.prior/(1-self.prior)))))
        
        return y



    @staticmethod
    def filter_train_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('penalty', 'lambda_reg',
                      'use_bias', 'bias_scaling', 'no_use_bias',
                      'prior', 'lr_seed',
                      'solver', 'max_iter',
                      'dual', 'tol', 'verbose',
                      'warm_start', 'no_warm_start', 'name')
        d = dict((k, kwargs[p+k])
               for k in valid_args if p+k in kwargs)
        if 'no_use_bias' in d:
            d['use_bias'] = not d['no_use_bias']
        if 'no_warm_start' in d:
            d['warm_start'] = not d['no_warm_start']
        
        return d


    
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
        parser.add_argument(p1+'lambda-reg', dest=(p2+'lambda_reg'), 
                            default=1e-5, type=float,
                            help='regularization strength')
        parser.add_argument(p1+'no-use-bias', dest=(p2+'no_use_bias'),
                            default=False, action='store_true',
                            help='Not use bias')
        parser.add_argument(p1+'bias-scaling', dest=(p2+'bias_scaling'),
                            default=1.0, type=float,
                            help='useful only when the solver liblinear is used and use_bias is set to True')
        parser.add_argument(p1+'lr-seed', dest=(p2+'lr_seed'), 
                            default=1024, type=int,
                            help='random number generator seed')
        parser.add_argument(p1+'solver', dest=(p2+'solver'), 
                            default='lbfgs', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
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
                            help='don\'t use previous model to start')

        parser.add_argument(p1+'prior', dest=(p2+'prior'),
                            default=0.1, type=float,
                            help='Target prior')


        parser.add_argument(p1+'name', dest=(p2+'name'), 
                            default='lr',
                            help='model name')

        
        


    
    
    
