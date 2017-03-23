
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from .pdf import PDF

class ExpFamily(PDF):
    
    def __init__(self, x_dim=1, **kwargs):
        super(ExpFamily, self).__init__(x_dim, **kwargs)
        self.eta = None

        
    def logh(self, x):
        return 0
            

    def fit(self, x_train, sample_weights_train=None,
            x_val=None, sample_weights_val=None, batch_size=None):
        stats=self.Estep(x=x_train, sample_weights=sample_weights_train,
                         batch_size=batch_size)
        self.Mstep(stats)
        elbo=self.elbo(stats)
        if x_val is not None:
            stats=self.Estep(x=x_val, sample_weights=sample_weights_val,
                             batch_size=batch_size)
            elbo_val=self.elbo(stats)
            elbo=[elbo, elbo_val]
        return elbo

    
    def accum_suff_stats(self, x, u_x=None, sample_weights=None, batch_size=None):
        if u_x is not None or batch_size is None:
            return self._accum_suff_stats_1batch(x, u_x, sample_weights)
        else:
            return self._accum_suff_stats_nbatches(x, sample_weights, batch_size)

        
    def _accum_suff_stats_1batch(self, x, u_x=None, sample_weights=None):
        if u_x is None:
            u_x=self.compute_suff_stats(x)
        if sample_weights is not None:
            u_x*=sample_weights[:, None]
        stats=np.sum(u_x, axis=0)
        return stats

    
    def _accum_suff_stats_nbatches(self, x, sample_weights, batch_size):
        sw_i = None
        for i1 in xrange(0, x.shape[0], batch_size):
            i2 = np.minimum(i1+batch_size, x.shape[0])
            x_i = x[i1:i2,:]
            if sample_weights is not None:
                sw_i = sample_weights[i1:i2]
            stats_i = self._accum_suff_stats_1batch(x_i, sample_weights=sw_i)
            if i1 == 0:
                stats = stats_i
            else:
                stats += stats_i
        return stats
    
    
    def Estep(self, x, u_x=None, sample_weights=None, batch_size=None):
        return self.accum_suff_stats(x, u_x, sample_weights, batch_size)

        
    def Mstep(self, stats):
        pass

    
    def elbo(self, stats):
        return self.eval_llk_nat([], u_x=stats)

    
    def eval_llk(self, x, u_x=None, mode='natural'):
        if mode == 'natural':
            return self.eval_llk_nat(x, u_x)
        else:
            return self.eval_llk_std(x)

        
    def eval_llk_nat(self, x, u_x = None):
        if u_x is None:
            u_x = self.compute_suff_stats(x)
        return self.logh(x)+np.dot(u_x, self.eta)

    
    def compute_suff_stats(self, x):
        return x

    
    def _compute_nat_params(self):
        pass

    
