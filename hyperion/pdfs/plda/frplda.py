
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
from numpy import linalg as nla
from scipy import linalg as sla

from abc import ABCMeta, abstractmethod

from ...hyp_defs import float_cpu
from ...utils.math import invert_pdmat, invert_trimat, logdet_pdmat
from .plda_base import PLDABase

class FRPLDA(PLDABase):

    def __init__(self, mu=None, B=None, W=None, fullcov_W=True,
                 update_mu=True, update_B=True, update_W=True, **kwargs):
        super(FRPLDA, self).__init__(mu=mu, update_mu=update_mu, **kwargs)
        if mu is not None:
            self.y_dim = mu.shape[0]
        self.B = B
        self.W = W
        self.fullcov_W = fullcov_W
        self.update_B = update_B
        self.update_W = update_W

        
    def initialize(self, D):
        N, F, S = D
        self.x_dim = F.shape[1]
        self.y_dim = F.shape[1]
        M = F.shape[0]
        N_tot = np.sum(N)

        y = F/N[:,None]
        Fy = np.dot(F.T, y)
        C = S - Fy - Fy.T
        for i in xrange(M):
            yy = np.outer(y[i,:], y[i,:])
            C += N[i] * yy

        C = (C+C.T)/2
        mu = np.mean(y, axis=0)
        iB = np.dot(y.T, y)/M - np.outer(mu, mu)
        iW = C/N_tot

        B = invert_pdmat(iB, return_inv=True)[-1]
        W = invert_pdmat(iW, return_inv=True)[-1]

        self.mu = mu
        self.B = B
        self.W = W

        
    def compute_py_g_x(self, D, return_cov=False, return_logpy_0=False,
                       return_acc=False):
        N, F, S = D

        M=F.shape[0]
        y_dim = self.y_dim
        assert(y_dim == F.shape[1])

        compute_inv = return_cov or return_acc

        N_is_int = False
        if np.all(np.ceil(N) == N):
            N_is_int = True
        
        gamma = np.dot(F, self.W) + np.dot(self.mu, self.B)
        if N_is_int:
            iterator = np.unique(N)
        else:
            iterator = xrange(M)

        y = np.zeros_like(F)
        if return_cov:
            Sigma_y = np.zeros((M, y_dim, y_dim), dtype=float_cpu())
        else:
            Sigma_y = None
            
        if return_logpy_0:
            logpy = - 0.5*y_dim*np.log(2*np.pi) * np.ones((M,), dtype=float_cpu())

        if return_acc:
            Py = np.zeros((y_dim, y_dim), dtype=float_cpu())
            Ry = np.zeros((y_dim, y_dim), dtype=float_cpu())

        for k in iterator:
            if N_is_int:
                i = (N == k).nonzero()[0]
                N_i = k
                M_i = len(i)
            else:
                i = k
                N_i = N[k]
                M_i = 1
                
            L_i = self.B + N_i*self.W
            
            r = invert_pdmat(L_i, right_inv=True,
                             return_logdet=return_logpy_0,
                             return_inv=compute_inv)
            mult_iL = r[0]
            if return_logpy_0:
                logL = r[2]
            if compute_inv:
                iL = r[-1]
            
            y[i,:]=mult_iL(gamma[i,:])
            
            if return_cov:
                Sigma_y[i,:,:]=iL

            if return_logpy_0:
                logpy[i] += 0.5*(logL - np.sum(y[i,:]*gamma[i,:], axis=-1))
                
            if return_acc:
                Py += M_i*iL

            r = [y]
            if return_cov:
                r += [Sigma_y]
            if return_logpy_0:
                r += [logpy]
            if return_acc:
                r += [Ry, Py]
        return r


    def Estep(self, D):
        N, F, S = D
        y, logpy, Ry, Py = self.compute_py_g_x(
            D, return_logpy_0=True, return_acc=True)

        M=F.shape[0]
        N_tot=np.sum(N)

        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F.T, y)
        
        Niy = y * N[:,None]
        Ry += np.dot(Niy.T, y)
        Py += np.dot(y.T, y)

        logpy_acc = np.sum(logpy)
        
        stats = [N_tot, M, S, logpy_acc, y_acc, Ry, Cy, Py]
        return stats

    
    def elbo(self, stats):
        N, M, S, logpy_x  = stats[:4]

        logW = logdet_pdmat(self.W)
        logB = logdet_pdmat(self.B)

        logpx_y = 0.5*(- N*self.x_dim*np.log(2*np.pi) + N*logW
                       - np.inner(self.W.ravel(), S.ravel()))
        logpy = 0.5*M*(- self.y_dim*np.log(2*np.pi) + logB
                       -np.inner(np.dot(self.mu, self.B), self.mu))
        
        elbo = logpx_y + logpy - logpy_x
        return elbo
        # N, M, sumy, yy, _, _, CW, logL = stats
        # ymu = np.outer(sumy, mu)
        # CB = yy - ymu -ymu.T + M*np.outer(self.mu, self.mu.T)

        # logW = logdet_pdmat(self.W)
        # logB = logdet_pdmat(self.B)
        
        # elbo = 0.5*(-logL - N*self.x_dim*np.log(2*np.pi)
        #             +N*logW - np.inner(self.W.ravel(), CW.ravel())
        #             +M*logB - np.inner(self.B.ravel(), CB.ravel()))
        # return elbo
        

    def MstepML(self, stats):
        N, M, S, _, y_acc, Ry, Cy, Py = stats
        ybar = y_acc/M
        if self.update_mu:
            self.mu = ybar
        if self.update_B:
            if self.update_mu:
                iB = Py/M - np.outer(self.mu, self.mu)
            else:
                muybar = np.outer(self.mu, ybar)
                iB = Py/M - muybar - muybar + np.outer(self.mu, self.mu)
            self.B = invert_pdmat(iB, return_inv=True)[-1]
        if self.update_W:
            iW = (S - Cy - Cy.T + Ry)/N
            if self.fullcov_W:
                self.W = invert_pdmat(iW, return_inv=True)[-1]
            else:
                self.W=np.diag(1/np.diag(iW))


                
    def MstepMD(self, stats):
        pass


    
    def get_config(self):
        config = { 'update_W': self.update_W,
                   'update_B': self.update_B,
                   'fullcov_W': self.fullcov_W}
        base_config = super(FRPLDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save_params(self, f):
        params = { 'mu': self.mu,
                   'B': self.B,
                   'W': self.W}
        self._save_params_from_dict(f, params)


        
    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'B', 'W']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)


    def eval_llr_1vs1(self, x1, x2):

        Lnon = self.B + self.W
        mult_icholLnon, logcholLnon = invert_trimat(
            sla.cholesky(Lnon, lower=False, overwrite_a=True),
            right_inv=True, return_logdet=True)[:2]
        logLnon = 2*logcholLnon

        Ltar = self.B + 2*self.W
        mult_icholLtar, logcholLtar = invert_trimat(
            sla.cholesky(Ltar, lower=False, overwrite_a=True),
            right_inv=True, return_logdet=True)[:2]
        logLtar = 2*logcholLtar

        WF1 = np.dot(x1, self.W)
        WF2 = np.dot(x2, self.W)
        Bmu = np.dot(self.mu, self.B)

        gamma_non_1 = mult_icholLnon(WF1+Bmu)
        gamma_non_2 = mult_icholLnon(WF2+Bmu)

        Qnon_1 = np.sum(gamma_non_1*gamma_non_1, axis=1)[:, None]
        Qnon_2 = np.sum(gamma_non_2*gamma_non_2, axis=1)

        gamma_tar_1 = mult_icholLtar(WF1+0.5*Bmu)
        gamma_tar_2 = mult_icholLtar(WF2+0.5*Bmu)

        Qtar_1 = np.sum(gamma_tar_1*gamma_tar_1, axis=1)[:, None]
        Qtar_2 = np.sum(gamma_tar_2*gamma_tar_2, axis=1)

        scores = 2*np.dot(gamma_tar_1, gamma_tar_2.T)
        scores += (Qtar_1-Qnon_1+Qtar_2-Qnon_2)
        scores += (2*logLnon-logLtar
                   -logdet_pdmat(self.B)
                   +np.inner(np.dot(self.mu, self.B), self.mu))
        scores *= 0.5
        return scores
                
            
    def eval_llr_NvsM_book(self, D1, D2):
        N1, F1, _ = D1
        N2, F2, _ = D2
        
        Bmu = np.dot(self.mu, self.B)
        
        scores = np.zeros((len(N1), len(N2)), dtype=float_cpu())
        for N1_i in np.unique(N1):
            for N2_j in np.unique(N2):
                i = np.where(N1 == N1_i)
                j = np.where(N2 == N2_j)

                L1 = self.B + N1_i*self.W
                mult_icholL1, logcholL1 = invert_trimat(
                    sla.cholesky(L1, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logL1 = 2*logcholL1

                L2 = self.B + N2_j*self.W
                mult_icholL2, logcholL2 = invert_trimat(
                    sla.cholesky(L2, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logL2 = 2*logcholL2

                Ltar = self.B + (N1_i + N2_j)*self.W
                mult_icholLtar, logcholLtar = invert_trimat(
                    sla.cholesky(Ltar, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logLtar = 2*logcholLtar
                
                WF1 = np.dot(F1[i,:], self.W)
                WF2 = np.dot(F2[j,:], self.W)
                
                gamma_non_1 = mult_icholL1(WF1 + Bmu)
                gamma_non_2 = mult_icholL2(WF2 + Bmu)
                
                Qnon_1 = np.sum(gamma_non_1*gamma_non_1, axis=1)[:, None]
                Qnon_2 = np.sum(gamma_non_2*gamma_non_2, axis=1)
                
                gamma_tar_1 = mult_icholLtar(WF1 + 0.5*Bmu)
                gamma_tar_2 = mult_icholLtar(WF2 + 0.5*Bmu)
                
                Qtar_1 = np.sum(gamma_tar_1*gamma_tar_1, axis=1)[:, None]
                Qtar_2 = np.sum(gamma_tar_2*gamma_tar_2, axis=1)
                
                scores_ij = 2*np.dot(gamma_tar_1, gamma_tar_2.T)
                scores_ij += (Qtar_1 - Qnon_1 + Qtar_2 - Qnon_2)
                scores_ij += (logL1 + logL2 - logLtar)
                scores[i,j] = scores_ij
                
        scores += (-logdet_pdmat(self.B) + np.inner(np.dot(self.mu, self.B), self.mu))
        scores *= 0.5

                

