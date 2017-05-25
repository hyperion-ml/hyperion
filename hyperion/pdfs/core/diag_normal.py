
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py
from scipy.special import erf

# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab

from ...hyp_defs import float_cpu
from ...utils.plotting import plot_gaussian_1D, plot_gaussian_ellipsoid_2D, plot_gaussian_ellipsoid_3D, plot_gaussian_3D

from .exp_family import ExpFamily


class DiagNormal(ExpFamily):
    
    def __init__(self, mu=None, Lambda=None, var_floor=1e-5, **kwargs):
        super(DiagNormal, self).__init__(**kwargs)
        self.mu = mu
        self.Lambda = Lambda
        self.var_floor = var_floor
        self._lnLambda = None
        self._cholLambda = None
        self._Sigma = None


    @property
    def lnLambda(self):
        if self._lnLambda is None:
            self._lnLambda = np.sum(np.log(self.Lambda))
        return self._lnLambda


    @property
    def cholLambda(self):
        if self._cholLambda is None:
            self._cholLambda = np.sqrt(self.Lambda)
        return self._cholLambda
            
            
    @property
    def Sigma(self):
        if self._Sigma is None:
            self._Sigma = 1./self.Lambda
        return self._Sigma
    

    def initialize(self):
        self.validate()
        if (self.mu is not None) and (self.Lambda is not None):
            self._compute_nat_params()
        elif self.eta is not None:
            self.A = self.compute_A_nat(self.eta)
            self._compute_std_params()

    
    def stack_suff_stats(self, F, S=None):
        if S is None:
            return F
        return np.hstack((F,S))
    
    
    def unstack_suff_stats(self, stats):
        F=stats[:self.x_dim]
        S=stats[self.x_dim:]
        return F, S

    
    def accum_norm_suff_stats(self, x, u_x=None, sample_weights=None,
                              return_order2=False):
        N, acc_u_x = self.accum_suff_stats(x, u_x, sample_weights)
        F, S = self.unstack_suff_stats(acc_u_x)
        if return_order2:
            S=S-2*self.mu*F+N*self.mu**2
        else:
            S=None
        F=self.cholLambda*(F-N*self.mu)
        return N, self.stack_suff_stats(F, S)
    

    def Mstep(self, N, u_x):

        F, S = self.unstack_suff_stats(u_x)
        mu = F/N
        S = S/N-mu**2
        S[S<self.var_floor] = self.var_floor

        self.mu=mu
        self.Lambda=1/S
        self._Sigma = S
        self._cholLambda = None
        self._lnLambda = None
        self._compute_nat_params()
        

    def eval_llk_std(self, x):
        mah_dist2=np.sum(((x-self.mu)*self.cholLambda)**2, axis=1)
        return 0.5*self.lnLambda-0.5*self.x_dim*np.log(2*np.pi)-0.5*mah_dist2


    
    def eval_logcdf(self, x):
        delta=(x-self.mu)*self.cholLambda
        lk=0.5*(1+erf(delta/np.sqrt(2)))
        return np.sum(np.log(lk), axis=-1)


    
    def generate(self, num_samples, rng=None, seed=1024):
        if rng is None:
            rng=np.random.RandomState(seed)
        x=rng.normal(size=(num_samples, self.x_dim)).astype(float_cpu())
        return self.mu+1./self.cholLambda*x


    
    def get_config(self):
        config = {'var_floor': self.var_floor }
        base_config = super(DiagNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save_params(self, f):
        params = {'mu': self.mu,
                  'Lambda': self.Lambda}
        self._save_params_from_dict(f, params)


        
    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'Lambda']
        params = self._load_params_to_dict(f, config['name'], param_list)
        return cls(x_dim=config['x_dim'],
                   mu=params['mu'], Lambda=params['Lambda'],
                   var_floor=config['var_floor'], name=config['name'])


    
    def validate(self):
        if (self.mu is not None) and (self.Lambda is not None):
            assert(self.mu.shape[0] == self.x_dim)
            assert(self.Lambda.shape[0] == self.x_dim)
            assert(np.all(self.Lambda > 0))
        if self.eta is not None:
            assert(self.eta.shape[0] == self.x_dim*2)
            

            
    @staticmethod
    def compute_eta(mu, Lambda):
        Lmu = Lambda*mu
        eta = np.hstack((Lmu, -0.5*Lambda))
        return eta


    
    @staticmethod
    def compute_std(eta):
        x_dim = int(eta.shape[0]/2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        mu = -0.5*eta1/eta2
        Lambda = -2*eta2
        return mu, Lambda


    
    @staticmethod
    def compute_A_nat(eta):
        x_dim = int(eta.shape[0]/2)
        eta1 = eta[:x_dim]
        eta2 = eta[x_dim:]
        r1 = 0.5 * x_dim*np.log(2*np.pi)
        r2 = -1/4 * np.sum(eta1*eta1/eta2)
        r3 = -1/2 * np.sum(np.log(-2*eta2))
        return r1 + r2 + r3


    
    @staticmethod
    def compute_A_std(mu, Lambda):
        x_dim = mu.shape[0]
        r1 = 0.5*x_dim*np.log(2*np.pi)
        r2 = -0.5*np.sum(np.log(Lambda))
        r3 = 0.5*np.sum(mu*mu*Lambda)
        return r1 + r2 + r3

    
    def _compute_nat_params(self):
        self.eta = self.compute_eta(self.mu, self.Lambda)
        self.A = self.compute_A_nat(self.eta)
        # Lmu = self.Lambda*self.mu
        # muLmu = np.sum(self.mu*Lmu)
        # lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        # self.eta=np.hstack((lnr, Lmu, -0.5*self.Lambda)).T


    def _compute_std_params(self):
        self.mu, self.Lambda = self.compute_std(self.eta)
        self._cholLambda = None
        self._lnLambda = None
        self._Sigma = None
        
        
    @staticmethod
    def compute_suff_stats(x):
        d = x.shape[1]
        u = np.zeros((x.shape[0],2*d), dtype=float_cpu())
        u[:,:d] = x
        u[:,d:] = x*x
        return u
    

    def plot1D(self, feat_idx=0, num_sigmas=2, num_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=1/self.Lambda[feat_idx]
        plot_gaussian_1D(mu, C, num_sigmas, num_pts, **kwargs)

    
    def plot2D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_2D(mu, C, num_sigmas, num_pts, **kwargs)

        
    def plot3D(self, feat_idx=[0, 1], num_sigmas=2, num_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_idx])
        plot_gaussian_3D(mu, C, num_sigmas, num_pts, **kwargs)
    
        
    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], num_sigmas=2, num_pts=100,
                         **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_idx])
        plot_gaussian_ellipsoid_3D(mu, C, num_sigmas, num_pts, **kwargs)
