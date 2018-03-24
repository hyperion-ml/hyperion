
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from abc import ABCMeta, abstractmethod
from .pdf import PDF

class ExpFamily(PDF):
    __metaclass__ = ABCMeta
    
    def __init__(self, eta=None, **kwargs):
        super(ExpFamily, self).__init__(**kwargs)
        self.eta = eta
        self.A = None
            

    def fit(self, x, sample_weight=None,
            x_val=None, sample_weight_val=None, batch_size=None):

        N, u_x =self.Estep(x=x, sample_weight=sample_weight,
                           batch_size=batch_size)
        self.Mstep(N, u_x)
        elbo=self.elbo(x, N=N, u_x=u_x)
        elbo = [elbo, elbo/N]
        
        if x_val is not None:
            N, u_x = self.Estep(x=x_val, sample_weight=sample_weight_val,
                                batch_size=batch_size)
            elbo_val = self.elbo(x_val, N=N, u_x=u_x)
            elbo += [elbo_val, elbo_val/N]
        return elbo

    
    def log_h(self, x):
        return 0


    def accum_logh(self, x, sample_weight=None):
        if sample_weight is None:
            return np.sum(self.logh(x))
        return np.sum(sample_weight * self.logh(x))
    
    
    def compute_suff_stats(self, x):
        return x

    
    
    def accum_suff_stats(self, x, u_x=None, sample_weight=None, batch_size=None):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_1batch(x, u_x, sample_weight)
        else:
            return self._accum_suff_stats_nbatches(x, sample_weight, batch_size)

        
    def _accum_suff_stats_1batch(self, x, u_x=None, sample_weight=None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        if sample_weight is None:
            N = u_x.shape[0]
        else:
            u_x *= sample_weight[:, None]
            N = np.sum(sample_weight)
        acc_u_x=np.sum(u_x, axis=0)
        return N, acc_u_x

    
    def _accum_suff_stats_nbatches(self, x, sample_weight, batch_size):
        sw_i = None
        for i1 in xrange(0, x.shape[0], batch_size):
            i2 = np.minimum(i1+batch_size, x.shape[0])
            x_i = x[i1:i2,:]
            if sample_weight is not None:
                sw_i = sample_weight[i1:i2]
            N_i, u_x_i = self._accum_suff_stats_1batch(x_i, sample_weight=sw_i)
            if i1 == 0:
                N = N_i
                u_x = u_x_i
            else:
                N += N_i
                u_x += u_x_i
        return N, u_x


    def add_suff_stats(self, N, u_x):
        assert(len(N)==len(u_x))
        acc_N = N[1]
        acc_u_x = u_x[1]
        for i in xrange(1,len(N)):
            acc_N += N
            acc_u_x += u[i]
        return acc_N, acc_u_x

    
    
    def Estep(self, x, u_x=None, sample_weight=None, batch_size=None):
        return self.accum_suff_stats(x, u_x, sample_weight, batch_size)

    
    @abstractmethod
    def Mstep(self, stats):
        pass

    
    def elbo(self, x, u_x=None, N=1, logh=None, sample_weight=None, batch_size=None):
        if u_x is None:
            N, u_x = self.accum_suff_stats(x, sample_weight=sample_weight,
                                        batch_size=batch_size)
        if logh is None:
            logh = self.accum_logh(x, sample_weight=sample_weight)
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
    
    
