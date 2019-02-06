"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import logging
import numpy as np
from scipy.special import gammaln

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..utils.math import int2onehot, logdet_pdmat, invert_pdmat, softmax



class LinearGBE(HypModel):

    def __init__(self, mu=None, W=None, 
                 update_mu=True, update_W=True,
                 x_dim=1, num_classes=None, balance_class_weight=True,
                 beta=None, nu=None,
                 prior=None, prior_beta=None, prior_nu=None,
                 post_beta=None, post_nu=None,
                 **kwargs):
        
        super(LinearGBE, self).__init__(**kwargs)
        if mu is not None:
            num_classes = mu.shape[0]
            x_dim = mu.shape[1]

        self.mu = mu
        self.W = W
        self.update_mu = update_mu
        self.update_W = update_W
        self.x_dim = x_dim
        self.num_classes = num_classes
        self.balance_class_weight = balance_class_weight
        self.A = None
        self.b = None
        self.prior = prior
        self.beta = beta
        self.nu = nu
        self.prior_beta = prior_beta
        self.prior_nu = prior_nu
        self.post_beta= post_beta
        self.post_nu = post_nu

        self._compute_Ab()


        
    def get_config(self):
        config = { 'update_mu': self.update_mu,
                   'update_W': self.update_W,
                   'x_dim': self.x_dim,
                   'num_classes': self.num_classes,
                   'balance_class_weight': self.balance_class_weight,
                   'prior_beta': self.prior_beta,
                   'prior_nu': self.prior_nu,
                   'post_beta': self.post_beta,
                   'post_nu': self.post_nu }
        
        base_config = super(LinearGBE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def _load_prior(self):
        if isinstance(self.prior, string_types):
            self.prior = LinearGBE.load(self.prior)
        num_classes = self.prior.mu.shape[0]
        if self.prior_beta is not None:
            self.prior.beta = self.prior_beta*np.ones((num_classes,), dtype=float_cpu())
        if self.prior_nu is not None:
            self.prior.nu = num_classes*self.prior_nu


            
    def _change_post_r(self):
        
        if self.post_beta is not None:
            self.beta = self.post_beta*np.ones((self.num_classes,), dtype=float_cpu())
        if self.post_nu is not None:
            self.nu = self.num_classes*self.post_nu

    

    def eval_linear(self, x):
        return np.dot(x, self.A) + self.b


    
    def eval_llk(self, x):
        logp = np.dot(x, self.A) + self.b
        K = 0.5*logdet_pdmat(self.W) - 0.5*self.x_dim*np.log(2*np.pi)
        K += -0.5*np.sum(np.dot(x, self.W)*x, axis=1, keepdims=True)
        logp += K
        return logp

    

    def eval_predictive(self, x):

        K = self.W/self.nu
        c = (self.nu+1-self.x_dim)
        r = self.beta/(self.beta+1)
        
        # T(mu, L, c) ; L = c r K
        
        logg = gammaln((c+self.x_dim)/2) - gammaln(c/2) - 0.5*self.x_dim*np.log(c*np.pi)

        # 0.5*log|L| = 0.5*log|K| + 0.5*d*log(c r) 
        logK = logdet_pdmat(K)
        logL_div_2 = 0.5*logK + 0.5*self.x_dim*r 
        
        # delta2_0 = (x-mu)^T W (x-mu)
        delta2_0 = np.sum(np.dot(x, self.W)*x, axis=1, keepdims=True) - 2*(
            np.dot(x, self.A) + self.b)
        # delta2 = (x-mu)^T L (x-mu) = c r delta0 / nu
        # delta2/c = r delta0 / nu
        delta2_div_c = r*delta2_0/self.nu

        D = -0.5*(c+self.x_dim)*np.log(1+delta2_div_c)
        logging.debug(self.nu)
        logging.debug(c)
        logging.debug(self.x_dim)
        logging.debug(logg)
        logging.debug(logL_div_2.shape)
        logging.debug(D.shape)
        
        logp = logg + logL_div_2 + D
        return logp


    
    def predict(self, x, eval_method='linear', normalize=False):
        if eval_method == 'linear':
            logp = self.eval_linear(x)
        elif eval_method == 'llk':
            logp = self.eval_llk(x)
        elif eval_method == 'predictive':
            logp = self.eval_predictive(x)
        else:
            raise ValueError('wrong eval method %s' % eval_method)
            
        if normalize:
            logp = np.log(softmax(logp, axis=1))
            
        return logp



    
    
    def fit(self, x, class_ids=None, p_theta=None, sample_weight=None):

        assert class_ids is not None or p_theta is not None

        do_map = True if self.prior is not None else False
        if do_map:
            self._load_prior()
        
        self.x_dim = x.shape[-1]
        if self.num_classes is None:
            if class_ids is not None:
                self.num_classes = np.max(class_ids)+1
            else:
                self.num_classes = p_theta.shape[-1]
        
        if class_ids is not None:
            p_theta = int2onehot(class_ids, self.num_classes)

        if sample_weight is not None:
            p_theta = sample_weight[:, None]*p_theta
      
        N = np.sum(p_theta, axis=0)

        F = np.dot(p_theta.T, x)

        if self.update_mu:
            xbar = F/N[:,None]
            if do_map:
                alpha_mu = (N/(N+self.prior.beta))[:, None]
                self.mu = (1-alpha_mu)*self.prior.mu + alpha_mu*xbar
                self.beta = N+self.prior.beta
            else:
                self.mu = xbar
                self.beta = N
        else:
            xbar = self.mu

            
        if self.update_W:
            if do_map:
                nu0 = self.prior.nu
                S0 = invert_pdmat(self.prior.W, return_inv=True)[-1]
                if self.balance_class_weight:
                    alpha_W = (N/(N+nu0/self.num_classes))[:, None]
                    S = (self.num_classes - np.sum(alpha_W))*S0
                else:
                    S = nu0*S0
            else:
                nu0 = 0
                S = np.zeros((x.shape[1], x.shape[1]), dtype=float_cpu())
                
            for k in xrange(self.num_classes):
                delta = x - xbar[k]
                S_k = np.dot(p_theta[:, k]*delta.T, delta)
                if do_map and self.update_mu:
                    mu_delta = xbar[k] - self.prior.mu[k]
                    S_k += N[k]*(1-alpha_mu[k])*np.outer(mu_delta, mu_delta)

                if self.balance_class_weight:
                    S_k /= (N[k]+nu0/self.num_classes)

                S += S_k
                
            if self.balance_class_weight:
                S /= self.num_classes
            else:
                S /= (nu0+np.sum(N))

            self.W = invert_pdmat(S, return_inv=True)[-1]
            self.nu = np.sum(N)+nu0
            
        self._change_post_r()
        self._compute_Ab()



    def save_params(self, f):
        params = { 'mu': self.mu,
                   'W': self.W,
                   'beta': self.beta,
                   'nu': self.nu }
        self._save_params_from_dict(f, params)
        

        
    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'W', 'beta', 'nu']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)


    
    def _compute_Ab(self):
        if self.mu is not None and self.W is not None:
            self.A = np.dot(self.W, self.mu.T)
            self.b = -0.5 * np.sum(self.mu.T*self.A, axis=0)
            

            
    @staticmethod
    def filter_train_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('update_mu', 'update_W',
                      'no_update_mu', 'no_update_W',
                      'balance_class_weight',
                      'prior', 'prior_beta', 'prior_nu',
                      'post_beta', 'post_nu',
                      'name')
        d = dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)
        if 'no_update_mu' in d:
            d['update_mu'] = not d['no_update_mu']
        if 'no_update_W' in d:
            d['update_W'] = not d['no_update_W']
            
        return d

    
    @staticmethod
    def add_argparse_train_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'no-update-mu', dest=(p2+'no_update_mu'), 
                            default=False, action='store_true',
                            help='do not update mu')
        parser.add_argument(p1+'no-update-W', dest=(p2+'no_update_W'),
                            default=False, action='store_true',
                            help='do not update W')
        parser.add_argument(p1+'balance-class-weight', dest=(p2+'balance_class_weight'),
                            default=False, action='store_true',
                            help='Balances the weight of each class when computing W')
        parser.add_argument(p1+'prior', dest=(p2+'prior'),
                            default=None, 
                            help='prior file for MAP adaptation')
        parser.add_argument(p1+'prior-beta', dest=(p2+'prior_beta'),
                            default=16, type=float,
                            help='relevance factor for the means')
        parser.add_argument(p1+'prior-nu', dest=(p2+'prior_nu'),
                            default=16, type=float,
                            help='relevance factor for the variances')
        parser.add_argument(p1+'post-beta', dest=(p2+'post_beta'),
                            default=None, type=float,
                            help='relevance factor for the means')
        parser.add_argument(p1+'post-nu', dest=(p2+'post_nu'),
                            default=None, type=float,
                            help='relevance factor for the variances')

        parser.add_argument(p1+'name', dest=(p2+'name'), 
                            default='lgbe',
                            help='model name')

        

    @staticmethod
    def filter_eval_args(prefix, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('model_file', 'normalize', 'eval_method')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)


    
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
        parser.add_argument(p1+'normalize', dest=(p2+'normalize'), default=False,
                            action='store_true',
                            help=('normalizes the ouput probabilities to sum to one'))
        parser.add_argument(p1+'eval-method', dest=(p2+'eval_method'), default='linear',
                            choices=['linear','llk','predictive'],
                            help=('evaluates full gaussian likelihood, linear function'
                                  'or predictive distribution'))
                            
        
