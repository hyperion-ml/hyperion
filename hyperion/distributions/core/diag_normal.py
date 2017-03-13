
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py
from scipy.special import erf

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from ...utils.plotting import plot_gaussian_1D, plot_gaussian_ellipsoid_2D, plot_gaussian_ellipsoid_3D, plot_gaussian_3D

from .exp_family import ExpFamily


class DiagNormal(ExpFamily):
    
    def __init__(self, x_dim=1, mu=None, Lambda=None, var_floor=1e-5, **kwargs):
        super(DiagNormal, self).__init__(x_dim, **kwargs)
        self.mu = mu
        self.Lambda = Lambda
        self.var_floor = var_floor
        self.lnLambda = None
        self.cholLambda = None
        self._Sigma = None

    @property
    def Sigma(self):
        if self.Lambda is None:
            return None
        if self._Sigma is not None:
            self._Sigma = 1./Lambda
        return self._Sigma
    

    def initialize(self):
        if (self.mu is not None) and (self.Lambda is not None):
            self.validate()
            self._compute_aux_params()
            self._compute_nat_params()

    
    def stack_suff_stats(self, N, F, S=None):
        if S is None:
            return np.hstack((N,F))
        return np.hstack((N,F,S))
    
    
    def unstack_suff_stats(self,stats):
        N=stats[0]
        F=stats[1:self.x_dim+1]
        S=stats[self.x_dim+1:]
        return N, F, S

    def accum_norm_suff_stats(self, x, u_x=None, sample_weights=None, compute_order2=False):
        stats=self.compute_suff_stats(x, u_x, sample_weights)
        N, F, S = self.unstack_suff_stats(stats)
        if compute_order2:
            S=S-2*self.mu*F+N*self.mu**2
        else:
            S=None
        F=self.cholLambda*(F-N*self.mu)
        return self.stack_suff_stats(N, F, S)
    

    def Mstep(self, stats):
        N, F, S = self.unstack_suff_stats(stats)
        mu=F/N
        S=S/N-mu**2
        S[S<self.var_floor]=var_floor

        self.mu=mu
        self.Lambda=1/S
        self._compute_aux_params()
        self._compute_nat_params()
        self._Sigma = None

    def ElnPx_g_muL(self, N, F, S):
        u_x=np.hstack((F, S, N))
        return self.eval_llk_nat([], u_x)

    
    def eval_llk_std(self, x):
        mah_dist2=np.sum(((x-self.mu)*self.cholLambda)**2, axis=1)
        return 0.5*self.lnLambda-0.5*self.x_dim*np.log(2*np.pi)-0.5*mah_dist2


    def eval_logcdf(self, x):
        delta=(x-self.mu)*self.cholLambda
        lk=0.5*(1+erf(delta/np.sqrt(2)))
        return np.sum(np.log(lk), axis=1)

        
    def generate(self, nb_samples, rng=None, seed=1024):
        if rng is None:
            rng=np.random.RandomState(seed)
        x=rng.normal(size=(nb_samples, self.x_dim))
        return self.mu+1./self.cholLambda*x


    def get_config(self):
        config = {'var_floor': self.var_floor }
        base_config = super(DiagNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def save_params(self, f):
        params = {'mu': self.mu,
                  'Lambda': self.Lambda}
        self._save_params_from_dict(f, params)
        # f.create_dataset('mu', self.mu.shape, dtype='float32')
        # f['mu']=self.mu.astype('float32')
        # f.create_dataset('Lambda', self.Lambda.shape, dtype='float32')
        # f['Lambda']=self.Lambda.astype('float32')

        
    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'Lambda']
        params = self._load_params_to_dict(f, config['name'], param_list)
        return cls(x_dim=config['x_dim'],
                   mu=params['mu'], Lambda=params['Lambda'],
                   var_floor=config['var_floor'], name=config['name'])

        
    # @classmethod
    # def load(cls, file_path):
    #     with h5py.File(file_path,'r') as f:
    #         config = self.load_config_from_json(f['config'])
    #         param_list = ['mu', 'Lambda']
    #         params = self._load_params_to_dict(f, config['name'], param_list)
    #         return cls(x_dim=config['x_dim'], params['mu'], params['Lambda'],
    #                    var_floor=config['var_floor'], name=config['name'])

    # @classmethod
    # def load(cls, file_path):
    #     with h5py.File(file_path,'r') as f:
    #         config=self.load_config_from_json(f['config'])
    #         mu=np.asarray(f['mu'], dtype=np.float64)
    #         Lambda=np.asarray(f['Lambda'], dtype=np.float64)
    #         return cls(x_dim=config['x_dim'], mu=mu, Lambda=Lambda,
    #                    var_floor=config['var_floor'])


    def validate(self):
        if (self.mu is not None) and (self.Lambda is not None):
            assert(self.mu.shape[0] == self.x_dim)
            assert(self.Lambda.shape[0] == self.x_dim)
            assert(np.all(self.Lambda > 0))
            if self.eta is not None:
                assert(self.eta.shape[0] == self.x_dim*2+1)

            
    def plot1D(self, feat_idx=0, nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=1/self.Lambda[feat_idx]
        plot_gaussian_1D(mu, C, nb_sigmas, nb_pts, **kwargs)

    
    def plot2D(self, feat_idx=[0, 1], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_indx])
        plot_gaussian_ellipsoid_2D(mu, C, nb_sigmas, nb_pts, **kwargs)

        
    def plot3D(self, feat_idx=[0, 1, 2], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_indx])
        plot_gaussian_3D(mu, C, nb_sigmas, nb_pts, **kwargs)
    
        
    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        C=np.diag(1./self.Lambda[feat_indx])
        plot_gaussian_ellipsoid_3D(mu, C, nb_sigmas, nb_pts, **kwargs)

    
    def _compute_aux_params(self):
        self.lnLambda = np.sum(np.log(self.Lambda))
        self.cholLambda = np.sqrt(self.Lambda)

        
    def _compute_nat_params(self):
        Lmu = self.Lambda*self.mu
        muLmu = np.sum(self.mu*Lmu)
        lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        self.eta=np.hstack((lnr, Lmu, -0.5*self.Lambda)).T

        
    @staticmethod
    def compute_suff_stats(x):
        d=x.shape[1]
        u=np.ones((x.shape[0],d))
        u[:,1:d+1]=x
        u[:,d+1:]=x*x
        return u
    
