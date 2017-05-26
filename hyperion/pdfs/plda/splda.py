
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

class SPLDA(PLDABase):

    def __init__(self, y_dim=None, mu=None, V=None, W=None, fullcov_W=True,
                 update_mu=True, update_V=True, update_W=True, **kwargs):
        super(SPLDA, self).__init__(y_dim=y_dim, mu=mu, update_mu=update_mu, **kwargs)
        if V is not None:
            self.y_dim = V.shape[0]
        self.V = V
        self.W = W
        self.fullcov_W = fullcov_W
        self.update_V = update_V
        self.update_W = update_W

        
    def initialize(self, D):
        N, F, S = D
        self.x_dim = F.shape[1]
        M = F.shape[0]
        N_tot = np.sum(N)

        Vytilde = F/N[:,None]
        mu = np.mean(Vytilde, axis=0)

        Vy = Vytilde - mu
        U, s, Vt = sla.svd(Vy, full_matrices=False, overwrite_a=True)
        V = s[:self.y_dim,None]*Vt[:self.y_dim,:]
        NVytilde = N[:, None]*Vytilde
        C = (S - np.dot(NVytilde.T, Vytilde))/N_tot
        if self.fullcov_W:
            W = invert_pdmat(C, return_inv=True)[-1]
        else:
            W = 1/np.diag(C)
        
        self.mu = mu
        self.V = V
        self.W = W

        
    def compute_py_g_x(self, D, return_cov=False, return_logpy_0=False,
                       return_acc=False):
        N, F, S = D

        M=F.shape[0]
        y_dim = self.y_dim

        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)

        compute_inv = return_cov or return_acc

        N_is_int = False
        if np.all(np.ceil(N) == N):
            N_is_int = True

        I = np.eye(y_dim, dtype=float_cpu())
        gamma = np.dot(F, WV)
        if N_is_int:
            iterator = np.unique(N)
        else:
            iterator = xrange(M)

        y = np.zeros((M, y_dim), dtype=float_cpu())
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
                
            L_i = I + N_i*VV
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
        F_tot = np.sum(F, axis=0)

        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F.T, y)
        
        Niy = y * N[:,None]
        Ry1 = np.sum(Niy, axis=0)
        Ry += np.dot(Niy.T, y)
        Py += np.dot(y.T, y)

        logpy_acc = np.sum(logpy)
        
        stats = [N_tot, M, F_tot, S, logpy_acc, y_acc, Ry1, Ry, Cy, Py]
        return stats

    
    def elbo(self, stats):
        N, M, F, S, logpy_x  = stats[:5]

        logW = logdet_pdmat(self.W)
        Fmu = np.outer(F, self.mu)
        Shat = S - Fmu - Fmu.T + N*np.outer(self.mu, self.mu)

        logpx_y = 0.5*(- N*self.x_dim*np.log(2*np.pi) + N*logW
                       - np.inner(self.W.ravel(), Shat.ravel()))
        logpy = -0.5*M*self.y_dim*np.log(2*np.pi)
        
        elbo = logpx_y + logpy - logpy_x
        return elbo
        

    def MstepML(self, stats):
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py = stats

        a = np.hstack((Ry, Ry1[:, None]))
        b = np.hstack((Ry1, N))
        Rytilde = np.vstack((a,b))

        Cytilde = np.hstack((Cy, F[:, None]))
        
        if self.update_mu and not self.update_V:
            self.mu = (F - np.dot(Ry1, self.V))/N

        if not self.update_mu and self.update_V:
            iRy_mult = invert_pdmat(Ry, right_inv=False)[0]
            self.V = iRy_mult(Cy.T - np.outer(Ry1, self.mu))

        if self.update_mu and self.update_V:
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            Vtilde = iRytilde_mult(Cytilde.T)
            self.V = Vtilde[:-1,:]
            self.mu = Vtilde[-1,:]

        if self.update_W:
            if self.update_mu and self.update_V:
                iW = (S - np.dot(Cy, self.V)  - np.outer(F, self.mu))/N
            else:
                Vtilde = np.vstack((self.V, self.mu))
                CVt = np.dot(Cytilde, Vtilde)
                iW = (S - CVt - CVt.T + np.dot(
                    np.dot(Vtilde.T, Rytilde), Vtilde))/N
            if self.fullcov_W:
                self.W = invert_pdmat(iW, return_inv=True)[-1]
            else:
                self.W=np.diag(1/np.diag(iW))

                
    def MstepMD(self, stats):
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py = stats
        mu_y = y_acc/M
        
        if self.update_mu:
            self.mu += np.dot(mu_y, self.V)

        if self.update_V:
            Cov_y = Py/M - np.outer(mu_y, mu_y)
            chol_Cov_y = sla.cholesky(Cov_y, lower=False, overwrite_a=True)
            self.V = np.dot(chol_Cov_y, self.V)


    def get_config(self):
        config = { 'update_W': self.update_W,
                   'update_V': self.update_V,
                   'fullcov_W': self.fullcov_W}
        base_config = super(SPLDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def save_params(self, f):
        params = { 'mu': self.mu,
                   'V': self.V,
                   'W': self.W}
        self._save_params_from_dict(f, params)


    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'V', 'W']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)


    def eval_llr_1vs1(self, x1, x2):

        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)
        I = np.eye(self.y_dim, dtype=float_cpu())
        
        Lnon = I + VV
        mult_icholLnon, logcholLnon = invert_trimat(
            sla.cholesky(Lnon, lower=False, overwrite_a=True),
            right_inv=True, return_logdet=True)[:2]
        logLnon = 2*logcholLnon

        Ltar = I + 2*VV
        mult_icholLtar, logcholLtar = invert_trimat(
            sla.cholesky(Ltar, lower=False, overwrite_a=True),
            right_inv=True, return_logdet=True)[:2]
        logLtar = 2*logcholLtar

        VWF1 = np.dot(x1-self.mu, WV)
        VWF2 = np.dot(x2-self.mu, WV)

        gamma_non_1 = mult_icholLnon(VWF1)
        gamma_non_2 = mult_icholLnon(VWF2)

        Qnon_1 = np.sum(gamma_non_1*gamma_non_1, axis=1)[:, None]
        Qnon_2 = np.sum(gamma_non_2*gamma_non_2, axis=1)

        gamma_tar_1 = mult_icholLtar(VWF1)
        gamma_tar_2 = mult_icholLtar(VWF2)

        Qtar_1 = np.sum(gamma_tar_1*gamma_tar_1, axis=1)[:, None]
        Qtar_2 = np.sum(gamma_tar_2*gamma_tar_2, axis=1)

        scores = 2*np.dot(gamma_tar_1, gamma_tar_2.T)
        scores += (Qtar_1-Qnon_1+Qtar_2-Qnon_2)
        scores += (2*logLnon-logLtar)
        scores *= 0.5
        return scores
                
            
    def eval_llr_NvsM_book(self, D1, D2):
        N1, F1, _ = D1
        N2, F2, _ = D2

        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)
        I = np.eye(self.y_dim, dtype=float_cpu())

        F1 -= N1[:, None]*self.mu
        F2 -= N2[:, None]*self.mu

        scores = np.zeros((len(N1), len(N2)), dtype=float_cpu())
        for N1_i in np.unique(N1):
            for N2_j in np.unique(N2):
                i = np.where(N1 == N1_i)
                j = np.where(N2 == N2_j)

                L1 = I + N1_i*VV
                mult_icholL1, logcholL1 = invert_trimat(
                    sla.cholesky(L1, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logL1 = 2*logcholL1

                L2 = I + N2_j*VV
                mult_icholL2, logcholL2 = invert_trimat(
                    sla.cholesky(L2, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logL2 = 2*logcholL2

                Ltar = I + (N1_i + N2_j)*VV
                mult_icholLtar, logcholLtar = invert_trimat(
                    sla.cholesky(Ltar, lower=False, overwrite_a=True),
                    right_inv=True, return_logdet=True)[:2]
                logLtar = 2*logcholLtar
                
                VWF1 = np.dot(F1[i,:], WV)
                VWF2 = np.dot(F2[j,:], WV)
                
                gamma_non_1 = mult_icholL1(VWF1)
                gamma_non_2 = mult_icholL2(VWF2)
                
                Qnon_1 = np.sum(gamma_non_1*gamma_non_1, axis=1)[:, None]
                Qnon_2 = np.sum(gamma_non_2*gamma_non_2, axis=1)
                
                gamma_tar_1 = mult_icholLtar(VWF1)
                gamma_tar_2 = mult_icholLtar(VWF2)
                
                Qtar_1 = np.sum(gamma_tar_1*gamma_tar_1, axis=1)[:, None]
                Qtar_2 = np.sum(gamma_tar_2*gamma_tar_2, axis=1)
                
                scores_ij = 2*np.dot(gamma_tar_1, gamma_tar_2.T)
                scores_ij += (Qtar_1 - Qnon_1 + Qtar_2 - Qnon_2)
                scores_ij += (logL1 + logL2 - logLtar)
                scores[i,j] = scores_ij
                
        scores *= 0.5

                

