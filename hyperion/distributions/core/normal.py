
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import h5py

import scipy.linalg as la

from ...utils.plotting import plot_gaussian_1D, plot_gaussian_ellipsoid_2D, plot_gaussian_ellipsoid_3D, plot_gaussian_3D
from ...utils.math import invert_pdmat, invert_trimat, symmat2vec, vec2symmat, fullcov_varfloor

from .diag_normal import DiagNormal


class Normal(DiagNormal):

    def __init__(self, x_dim=1, mu=None, Lambda=None, var_floor=1e-5):
        super(Normal, self).__init__(x_dim, mu, Lambda, var_floor)

    @property
    def Sigma(self):
        if self.Lambda is None:
            return None
        if self._Sigma is None:
            self._Sigma = invert_pdmat(self.Lambda, compute_inv=True)[3]
        return self._Sigma

    def accum_suff_stats(self, x, u_x=None, sample_weights=None, batch_size=None):
        if u_x is None:
            if sample_weights is None:
                N = x.shape[0]
                F = np.sum(x, axis=0)
                S = symmat2vec(np.dot(x.T, x))
            else:
                N = np.sum(sample_weights)
                wx = sample_weights[:, None]*x
                F = np.sum(wx, axis=0)
                S = symmat2vec(np.dot(wx.T, x))
            return self.stack_suff_stats(N, F, S)
        else:
            return self._accum_suff_stats_1batch(x, u_x, sample_weights)

        
    def accum_norm_suff_stats(self, x, u_x=None, sample_weights=None, compute_order2=False):
        stats=self.compute_suff_stats(x, u_x, sample_weights)
        N, F, S = self.unstack_suff_stats(stats)
        if compute_order2:
            SS = vec2symat(S)
            Fmu = np.outer(self.F,self.mu)
            SS = SS-Fmu-Fmu.T+N*np.outer(self.mu,self.mu)
            SS = np.dot(self.cholLambda.T,np.dot(SS,self.cholLambda))
            S = symmat2vec(SS)
        else:
            S = None
        F = np.dot(F-N*self.mu,self.cholLambda)
        return self.stack_suff_stats(N, F, S)
    

    def Mstep(self, stats):
        self._Sigma = None
        N, F, S = self.unstack_suff_stats(stats)
        self.mu = F/N

        S = vec2symmat(S/N)
        S -= np.outer(self.mu,self.mu)
        cholS = la.cholesky(S, overwrite_a=True)
        cholS = fullcov_varfloor(cholS, np.sqrt(self.var_floor))
        kk=la.inv(cholS)
        icholS = invert_trimat(cholS, compute_inv=True)[2]
        self.Lambda=np.dot(icholS, icholS.T)

        self._compute_aux_params()
        self._compute_nat_params()

        
    def eval_llk_std(self, x):
        mah_dist2 = np.sum(np.dot(x-self.mu,self.cholLambda.T)**2, axis=1)
        return 0.5*self.lnLambda-0.5*self.x_dim*np.log(2*np.pi)-0.5*mah_dist2

    
    def eval_logcdf(self, x):
        delta = np.dot((x-self.mu), self.cholLambda.T)
        lk = 0.5*(1+erf(delta/np.sqrt(2)))
        return np.sum(np.log(lk), axis=1)

    
    def generate(self, nb_samples, rng=None, seed=1024):
        if rng is None:
            rng=np.random.RandomState(seed)
        x=rng.normal(size=(nb_samples, self.x_dim))
        cholS=la.cholesky(invert_pdmat(self.Lambda, compute_inv=True)[3],
                          lower=True, overwrite_a=True)
        return self.mu+np.dot(x, cholS)

    
    def plot1D(self, feat_idx=0, nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C=invert_pdmat(self.Lambda, compute_inv=True)[3][i, j]
        plot_gaussian_1D(mu, C, nb_sigmas, nb_pts, **kwargs)

    
    def plot2D(self, feat_idx=[0, 1], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C=invert_pdmat(self.Lambda, compute_inv=True)[3][i, j]
        plot_gaussian_ellipsoid_2D(mu, C, nb_sigmas, nb_pts, **kwargs)

        
    def plot3D(self, feat_idx=[0, 1, 2], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C=invert_pdmat(self.Lambda, compute_inv=True)[3][i, j]
        plot_gaussian_3D(mu, L, nb_sigmas, nb_pts, **kwargs)
    
        
    def plot3D_ellipsoid(self, feat_idx=[0, 1, 2], nb_sigmas=2, nb_pts=100, **kwargs):
        mu=self.mu[feat_idx]
        j, i = np.meshgrid(feat_idx, feat_idx)
        C=invert_pdmat(self.Lambda, compute_inv=True)[3][i, j]
        plot_gaussian_ellipsoid_3D(mu, C, nb_sigmas, nb_pts, **kwargs)

    
    def validate(self):
        if (self.mu is not None) and (self.Lambda is not None):
            assert(self.mu.shape[0] == self.x_dim)
            assert(self.Lambda.shape == (self.x_dim, self.x_dim))
            if self.eta is not None:
                assert(self.eta.shape[0] == (self.x_dim**2+3*self.x_dim+1)/2)

                
    def _compute_aux_params(self):
        self.cholLambda, self.lnLambda = invert_pdmat(
            self.Lambda, compute_logdet=True)[1:3]

        
    def _compute_nat_params(self):
        Lmu = np.dot(self.Lambda, self.mu[:, None])
        muLmu = np.dot(self.mu, Lmu)
        lnr = 0.5*self.lnLambda - 0.5*self.x_dim*np.log(2*np.pi)-0.5*muLmu
        Lambda=np.copy(self.Lambda)
        Lambda[np.diag_indices(self.x_dim)] /= 2
        self.eta=np.vstack((lnr, Lmu, symmat2vec(Lambda)[:, None]))

        
    @staticmethod
    def compute_suff_stats(x):
        d=x.shape[1]
        u=np.ones((x.shape[0], int(d+d*(d+1)/2+1)))
        u[:,1:d+1]=x
        k=d
        for i in xrange(d):
            for j in xrange(i, d):
                k+=1
                u[:,k]=x[:,i]*x[:,j]
        return u
