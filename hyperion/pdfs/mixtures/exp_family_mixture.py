
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from abc import ABCMeta, abstractmethod

from ...hyp_defs import float_cpu
from ...utils.math import softmax
from ..core import PDF

class ExpFamilyMixture(PDF):
    __metaclass__ = ABCMeta
    
    def __init__(self, weights=None, eta=None, min_N=1, **kwargs):
        super(ExpFamilyMixture, self).__init__(**kwargs)
        self.weights = weights
        self.eta = eta
        self.min_N = min_N
        self.A = None
            

    def fit(self, x, sample_weights=None,
            x_val=None, sample_weights_val=None,
            epochs=10, batch_size=None):

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in xrange(epochs):
            N, u_x =self.Estep(x=x, sample_weights=sample_weights,
                               batch_size=batch_size)
            self.Mstep(N, u_x)
            elbo[epoch]=self.elbo(x, N=N, u_x=u_x)

        
            if x_val is not None:
                N, u_x = self.Estep(x=x_val, sample_weights=sample_weights_val,
                                    batch_size=batch_size)
                elbo_val[epoch] = self.elbo(x_val, N=N, u_x=u_x)

        if x_val is None:
            return elbo, elbo/x.shape[0]
        else:
            return elbo, elbo/x.shape[0], elbo_val, elbo_val/x.shape[0]

    
    def logh(self, x):
        return 0


    def accum_logh(self, x, sample_weights=None):
        if sample_weights is None:
            return np.sum(self.logh(x))
        return np.sum(sample_weights * self.logh(x))


    def compute_z(self, x, u_x=None, mode='nat'):
        if mode == 'nat':
            return self.compute_z_nat(x, u_x)
        else:
            return self.compute_z_std(x)


    def compute_z_nat(self, x, u_x=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        logr = np.dot(u_x, self.eta) - self.A + self.logw
        return softmax(logr)
    
    
    def compute_suff_stats(self, x):
        return x

    
    
    def accum_suff_stats(self, x, u_x=None, sample_weights=None, batch_size=None):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_1batch(x, u_x, sample_weights)
        else:
            return self._accum_suff_stats_nbatches(x, sample_weights, batch_size)

        
    def _accum_suff_stats_1batch(self, x, u_x=None, sample_weights=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        if sample_weights is None:
            N = u_x.shape[0]
        else:
            u_x *= sample_weights[:, None]
            N = np.sum(sample_weights)
        acc_u_x=np.sum(u_x, axis=0)
        return N, acc_u_x

    
    def _accum_suff_stats_nbatches(self, x, sample_weights, batch_size):
        sw_i = None
        for i1 in xrange(0, x.shape[0], batch_size):
            i2 = np.minimum(i1+batch_size, x.shape[0])
            x_i = x[i1:i2,:]
            if sample_weights is not None:
                sw_i = sample_weights[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_1batch(x_i, sample_weights=sw_i)
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x
    
    
    def Estep(self, x, u_x=None, sample_weights=None, batch_size=None):
        return self.accum_suff_stats(x, u_x, sample_weights, batch_size)

    
    @abstractmethod
    def Mstep(self, stats):
        pass

    
    def elbo(self, x, u_x=None, N=1, logh=None, sample_weights=None, batch_size=None):
        if u_x is None:
            N, u_x = self.accum_suff_stats(x, sample_weights=sample_weights,
                                        batch_size=batch_size)
        if logh is None:
            logh = self.accum_logh(x, sample_weights=sample_weights)
        return logh + np.inner(u_x, self.eta) - N*self.A

    
    def eval_llk(self, x, u_x=None, mode='nat'):
        if mode == 'nat':
            return self.eval_llk_nat(x, u_x)
        else:
            return self.eval_llk_std(x)

        
    def eval_llk_nat(self, x, u_x = None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        return self.logh(x) + np.inner(u_x, self.eta) - self.A

    
    
    @staticmethod
    def compute_A_nat(eta):
        raise NotImplementedError()

    
    @staticmethod
    def compute_A_std(params):
        raise NotImplementedError()

    
    @staticmethod
    def compute_eta(param):
        raise NotImplementedError()

    
    @staticmethod
    def compute_std(eta):
        raise NotImplementedError()

    
    @abstractmethod
    def _compute_nat_params(self):
        pass

    
    @abstractmethod
    def _compute_std_params(self):
        pass
    
    
