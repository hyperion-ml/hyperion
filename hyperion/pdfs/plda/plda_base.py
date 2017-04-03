
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from abc import ABCMeta, abstractmethod

from ...hyp_defs import float_cpu
from ..core.pdf import PDF
from ...transforms import LNorm

class PLDABase(PDF):
    __metaclass__ = ABCMeta

    def __init__(self, y_dim=None, mu=None, update_mu=True, **kwargs):
        super(PLDABase, self).__init__(**kwargs)
        self.mu = mu
        self.y_dim = y_dim
        self.update_mu = update_mu
        if mu is not None:
            self.x_dim = mu.shape[0]

    @abstractmethod
    def initialize(self, D):
        pass

    @abstractmethod
    def compute_py(self, D):
        pass

            
    def fit(self, x, class_ids=None, ptheta=None, sample_weight=None,
            x_val=None, class_ids_val=None, ptheta_val=None, sample_weight_val=None,
            epochs=20, md_epochs=None):


        assert(not(class_ids is  None and ptheta is None))
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta)
        else:
            D = self.compute_stats_hard(x, class_ids)

        if x_val is not None:
            assert(not(class_ids_val is  None and ptheta_val is None))        
            if class_ids_val is None:
                D_val = self.compute_stats_soft(x_val, ptheta_val)
            else:
                D_val = self.compute_stats_hard(x_val, class_ids_val)

        if self.mu is None:
            self.initialize(D)
                
        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in xrange(epochs):
            
            stats=self.Estep(D)
            elbo[epoch]=self.elbo(stats)
            if x_val is not None:
                stats_val=self.Estep(D_val)
                elbo_val[epoch]=self.elbo(stats_val)

            self.MstepML(stats)
            if md_epochs is None or epoch in md_epochs:
                self.MstepMD(stats)

        if x_val is None:
            return elbo
        else:
            return elbo, elbo_val

        
    @abstractmethod
    def Estep(self, x):
        pass

    
    @abstractmethod
    def MstepML(self, x):
        pass

    
    @abstractmethod
    def MstepMD(self, x):
        pass

    
    @abstractmethod
    def eval_llr_1vs1(self, x1, x2):
        pass

    
    @abstractmethod
    def eval_llr_NvsM_book(self, D1, D2):
        pass

    
    @staticmethod
    def compute_stats_soft(x, p_theta, sample_weight=None, scal_factor=None):
        if sample_weight is not None:
            p_theta = sample_weight[:, None]*p_theta
        if scal_factor is not None:
            p_theta *= scal_factor
        N = np.sum(p_theta, axis=0)
        F = np.dot(p_theta.T, x)
        wx = np.sum(p_theta, axis=1, keepdims=True)*x
        S = np.dot(x.T, wx)
        return N, F, S

    
    @staticmethod
    def compute_stats_hard(x, class_ids, sample_weight=None, scal_factor=None):
        x_dim=x.shape[1]
        num_classes = np.max(class_ids)+1
        p_theta = np.zeros((x.shape[0], num_classes), dtype=float_cpu())
        p_theta[np.arange(x.shape[0]), class_ids]=1
        return PLDABase.compute_stats_soft(x, p_theta, sample_weight, scal_factor)


    @staticmethod
    def center_stats(D, mu):
        N, F, S = D
        Fc = F - np.outer(N, mu)
        Fmu = np.outer(np.sum(F, axis=0), mu)
        Sc = S - Fmu - Fmu.T + np.sum(N)*np.outer(mu, mu)
        return N, Fc, Sc

        
    def eval_llr_NvsM(self, D1, D2, mode='avg-lnorm', ids1=None, ids2=None):
        if mode == 'book':
            return self.eval_llr_NvsM_book(D1, D2)
        if mode == 'vavg':
            return self.eval_llr_NvsM_vavg(D1, D2, do_lnorm=False)
        if mode == 'vavg-lnorm':
            return self.eval_llr_NvsM_vavg(D1, D2, do_lnorm=True)
        if mode == 'savg':
            return self.eval_llr_NvsM_savg(D1, ids1, D1, ids2)

        
    def eval_llr_NvsM_vavg(self, D1, D2, do_lnorm=True):
        x1=D1.F/D1.N
        x2=D2.F/D2.N
        if do_lnorm:
            lnorm=Lnorm()
            x1=lnorm.predict(x1)
            x2=lnorm.predict(x2)

        return self.eval_llr_1vs1(x1, x2)

    
    def eval_llr_NvsM_savg(self, x1, ids1, x2, ids2):
        scores_1vs1 = self.eval_llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats(scores_1vs1, ids1)
        scores_Nvs1 = F/N[:, None]
        N, F, _ = self.compute_stats(scores_Nvs1.T, ids2)
        scores = F.T/N
        return scores

    
    def eval_llr_Nvs1(self, D1, x2, mode='avg-lnorm', ids1=None):
        if mode == 'book':
            D2 = self.compute_stats(x2, np.arange(x.shape[0]))
            return self.eval_llr_NvsM_book(D1, D2)
        if mode == 'vavg':
            return self.eval_llr_Nvs1_vavg(D1, x2, do_lnorm=False)
        if mode == 'vavg-lnorm':
            return self.eval_llr_Nvs1_vavg(D1, x2, do_lnorm=True)
        if mode == 'savg':
            return self.eval_llr_Nvs1_savg(D1, ids1, x1)

        
    def eval_llr_Nvs1_vavg(self, D1, x2, do_lnorm=True):
        x1=D1.F/D1.N
        if do_lnorm:
            lnorm=Lnorm()
            x1=lnorm.predict(x1)
            x2=lnorm.predict(x2)

        return self.eval_llr_1vs1(x1, x2)

    
    def eval_llr_Nvs1_savg(self, x1, ids1, x2):
        scores_1vs1 = self.eval_llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats(scores_1vs1, ids1)
        scores = F/N[:, None]
        return scores
        
    @abstractmethod
    def generate(self, num_samples, rng=None, seed=1024):
        pass


    def get_config(self):
        config = { 'y_dim': self.y_dim,
                   'update_mu': self.update_mu}
        base_config = super(PLDABase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

