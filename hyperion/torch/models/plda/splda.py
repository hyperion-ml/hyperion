"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import time
import logging

import torch
import torch.nn as nn

from ...utils.math import invert_trimat
from .plda_base import PLDABase


class SPLDA(PLDABase):
    def __init__(
        self,
        x_dim=None,
        y_dim=None,
        mu=None,
        V=None,
        W=None,
        num_classes=0,
        x_ref=None,
        p_tar=0.05,
        margin_multi=0.3,
        margin_tar=0.3,
        margin_non=0.3,
        margin_warmup_epochs=10,
        adapt_margin=False,
        adapt_gamma=0.99,
        lnorm=False,
        var_floor=1e-5,
        prec_floor=1e-5,
        preprocessor=None,
    ):
        super().__init__(
            x_dim=x_dim,
            mu=mu,
            num_classes=num_classes,
            x_ref=x_ref,
            p_tar=p_tar,
            margin_multi=margin_multi,
            margin_tar=margin_tar,
            margin_non=margin_non,
            margin_warmup_epochs=margin_warmup_epochs,
            adapt_margin=adapt_margin,
            adapt_gamma=adapt_gamma,
            lnorm=lnorm,
            var_floor=var_floor,
            prec_floor=prec_floor,
            preprocessor=preprocessor,
        )

        if V is None:
            assert y_dim is not None
            V = torch.randn(y_dim, self.x_dim, dtype=torch.get_default_dtype())
        else:
            V = torch.as_tensor(V, dtype=torch.get_default_dtype())
            y_dim = V.shape[0]
            assert V.shape[1] == self.x_dim

        self.y_dim = y_dim
        self.V = nn.Parameter(V)

        if W is None:
            W = torch.eye(self.x_dim, dtype=torch.get_default_dtype())
        else:
            W = torch.as_tensor(W, dtype=torch.get_default_dtype())
            assert W.shape[0] == W.shape[1]
            assert W.shape[0] == self.x_dim

        # W_eigval, W_eigvec = torch.linalg.eigh(W)
        # W_eigval, W_eigvec = torch.symeig(W, eigenvectors=True)
        # self.W_eigval = nn.Parameter(W_eigval)
        # self.W_eigvec = nn.Parameter(W_eigvec)
        eigval, U = torch.symeig(W, eigenvectors=True)
        U = U * torch.sqrt(eigval)
        self.U = nn.Parameter(U)

    def __str__(self):
        return (
            "{}(x_dim={}, y_dim={}, num_classes={}, p_tar={}, "
            "margin_multi={}, margin_tar={}, margin_non={}, "
            "margin_warmup_epochs={}, "
            "adapt_margin={}, adapt_gamma={}, "
            "lnorm={}, preprocessor={}"
            "var_floor={}, prec_floor={} "
        ).format(
            self.__class__.__name__,
            self.x_dim,
            self.y_dim,
            self.num_classes,
            self.p_tar,
            self.margin_multi,
            self.margin_tar,
            self.margin_non,
            self.margin_warmup_epochs,
            self.adapt_margin,
            self.adapt_gamma,
            self.lnorm,
            self.preprocessor,
            self.var_floor,
            self.prec_floor,
        )

    # @property
    # def W(self):
    #     W_eigval = self.W_eigval.clamp(min=self.prec_floor)
    #     return torch.matmul(self.W_eigvec * W_eigval, self.W_eigvec.t())

    @property
    def W(self):
        return torch.matmul(self.U, self.U.t())

    # @property
    # def iW(self):
    #     iW_eigval = (1 / self.W_eigval.clamp(min=self.prec_floor)).clamp(min=self.var_floor)
    #     return torch.matmul(self.W_eigvec * iW_eigval, self.W_eigvec.t())

    def _compute_aux_L(self):
        WV = torch.matmul(self.W, self.V.t())
        VV = torch.matmul(self.V, WV)
        I = torch.eye(self.y_dim)
        return I, WV, VV

    def _compute_aux_llr_1vs1(self):
        I, WV, VV = self._compute_aux_L()
        Lnon = I + VV
        cholLnon = torch.cholesky(Lnon, upper=True)
        mult_icholLnon, logcholLnon = invert_trimat(
            cholLnon, lower=False, right_inv=True, return_logdet=True
        )
        logLnon = 2 * logcholLnon

        Ltar = I + 2 * VV
        cholLtar = torch.cholesky(Ltar, upper=True)
        mult_icholLtar, logcholLtar = invert_trimat(
            cholLtar, lower=False, right_inv=True, return_logdet=True
        )
        logLtar = 2 * logcholLtar

        return WV, mult_icholLnon, mult_icholLtar, logLnon, logLtar

    # @staticmethod
    # def _llr_from_Qs(Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon_1, logLnon_2):
    #     Q1 = (Qtar_1 - Qnon_1).unsqueeze(dim=-1)
    #     Q2 = Qtar_2-Qnon_2
    #     bias = logLnon_1+logLnon_2-logLtar
    #     scores = 0.5* (2 * Qtar_12 + Q1 + Q2 + bias)
    #     return scores

    # @staticmethod
    # def _llr_from_Qs2(Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon_1, logLnon_2):
    #     Q1 = (Qtar_1 - Qnon_1).unsqueeze(dim=-1)
    #     Q2 = Qtar_2-Qnon_2
    #     bias = logLnon_1+logLnon_2-logLtar
    #     scores = 0.5* (2 * Qtar_12 + (Q1 + Q2) + bias)
    #     return scores

    # @staticmethod
    # def _llr_from_Qs1(Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon_1, logLnon_2):
    #     Qtar_1 = Qtar_1.unsqueeze(dim=-1)
    #     Qnon_1 = Qnon_1.unsqueeze(dim=-1)
    #     scores = 2 * Qtar_12
    #     scores += (Qtar_1-Qnon_1+Qtar_2-Qnon_2)
    #     scores += (logLnon_1+logLnon_2-logLtar)
    #     scores *= 0.5
    #     return scores

    # @staticmethod
    # def _llr_from_Qs2(Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon_1, logLnon_2):
    #     Qtar_1 = Qtar_1.unsqueeze(dim=-1)
    #     Qnon_1 = Qnon_1.unsqueeze(dim=-1)
    #     scores = 2 * Qtar_12
    #     scores += ((Qtar_1-Qnon_1)+(Qtar_2-Qnon_2))
    #     scores += (logLnon_1+logLnon_2-logLtar)
    #     scores *= 0.5
    #     return scores

    @staticmethod
    def _llr_from_Qs(
        Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon_1, logLnon_2
    ):
        Q1 = (Qtar_1 - Qnon_1).unsqueeze(dim=-1)
        Q2 = Qtar_2 - Qnon_2
        scores = 2 * Qtar_12
        scores += Q1 + Q2
        scores += logLnon_1 + logLnon_2 - logLtar
        scores *= 0.5
        return scores

    @staticmethod
    def _llr_compQ(VWF, icholL):
        gamma = icholL(VWF)
        return gamma, torch.sum(gamma * gamma, dim=1)

    def llr_1vs1(self, x1, x2, aux_comps=None, preproc=True):
        if self.preprocessor is not None and preproc:
            x1 = self.preprocessor(x1)
            x2 = self.preprocessor(x2)

        if self.lnorm:
            x1 = self.l2_norm(x1)
            x2 = self.l2_norm(x2)

        if aux_comps is None:
            aux_comps = self._compute_aux_llr_1vs1()

        WV, icholLnon, icholLtar, logLnon, logLtar = aux_comps
        VWF1 = torch.matmul(x1 - self.mu, WV)
        VWF2 = torch.matmul(x2 - self.mu, WV)
        gamma_non_1, Qnon_1 = self._llr_compQ(VWF1, icholLnon)
        gamma_non_2, Qnon_2 = self._llr_compQ(VWF2, icholLnon)
        gamma_tar_1, Qtar_1 = self._llr_compQ(VWF1, icholLtar)
        gamma_tar_2, Qtar_2 = self._llr_compQ(VWF2, icholLtar)
        Qtar_12 = torch.matmul(gamma_tar_1, gamma_tar_2.t())
        return self._llr_from_Qs(
            Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon, logLnon
        )

    def llr_self(self, x, aux_comps=None, preproc=True):
        if self.preprocessor is not None and preproc:
            x = self.preprocessor(x)

        if self.lnorm:
            x = self.l2_norm(x)

        if aux_comps is None:
            aux_comps = self._compute_aux_llr_1vs1()

        WV, icholLnon, icholLtar, logLnon, logLtar = aux_comps
        VWF1 = torch.matmul(x - self.mu, WV)
        gamma_non_1, Qnon_1 = self._llr_compQ(VWF1, icholLnon)
        gamma_tar_1, Qtar_1 = self._llr_compQ(VWF1, icholLtar)
        Qtar_11 = torch.matmul(gamma_tar_1, gamma_tar_1.t())
        return self._llr_from_Qs(
            Qtar_11, Qtar_1, Qtar_1, Qnon_1, Qnon_1, logLtar, logLnon, logLnon
        )

    def llr_1vs1_and_self(self, x1, x2, aux_comps=None, preproc=True):
        if self.preprocessor is not None and preproc:
            x1 = self.preprocessor(x1)
            x2 = self.preprocessor(x2)

        if self.lnorm:
            x1 = self.l2_norm(x1)
            x2 = self.l2_norm(x2)

        if aux_comps is None:
            aux_comps = self._compute_aux_llr_1vs1()
        WV, icholLnon, icholLtar, logLnon, logLtar = aux_comps

        VWF1 = torch.matmul(x1 - self.mu, WV)
        VWF2 = torch.matmul(x2 - self.mu, WV)
        gamma_non_1, Qnon_1 = self._llr_compQ(VWF1, icholLnon)
        gamma_non_2, Qnon_2 = self._llr_compQ(VWF2, icholLnon)
        gamma_tar_1, Qtar_1 = self._llr_compQ(VWF1, icholLtar)
        gamma_tar_2, Qtar_2 = self._llr_compQ(VWF2, icholLtar)

        Qtar_12 = torch.matmul(gamma_tar_1, gamma_tar_2.t())
        llr_1vs1 = self._llr_from_Qs(
            Qtar_12, Qtar_1, Qtar_2, Qnon_1, Qnon_2, logLtar, logLnon, logLnon
        )

        Qtar_11 = torch.matmul(gamma_tar_1, gamma_tar_1.t())
        llr_self = self._llr_from_Qs(
            Qtar_11, Qtar_1, Qtar_1, Qnon_1, Qnon_1, logLtar, logLnon, logLnon
        )
        return llr_1vs1, llr_self

    def get_config(self):
        config = {"y_dim": self.y_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def llr_1vs1_1(self, x1, x2, aux_comps=None):
    #     t=time.time()
    #     if aux_comps is None:
    #         aux_comps = self._compute_aux_llr_1vs1()

    #     WV, mult_icholLnon, mult_icholLtar, logLnon, logLtar = aux_comps
    #     logging.info('   time1={}'.format(time.time()-t))
    #     t=time.time()
    #     VWF1 = torch.matmul(x1-self.mu, WV)
    #     VWF2 = torch.matmul(x2-self.mu, WV)
    #     logging.info('   time2={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_non_1 = mult_icholLnon(VWF1)
    #     logging.info('   time3={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_non_2 = mult_icholLnon(VWF2)
    #     logging.info('   time4={}'.format(time.time()-t))
    #     t=time.time()
    #     Qnon_1 = torch.sum(gamma_non_1*gamma_non_1, dim=1)
    #     Qnon_2 = torch.sum(gamma_non_2*gamma_non_2, dim=1)
    #     logging.info('   time5={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_tar_1 = mult_icholLtar(VWF1)
    #     logging.info('   time6={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_tar_2 = mult_icholLtar(VWF2)
    #     logging.info('   time7={}'.format(time.time()-t))
    #     t=time.time()
    #     Qtar_1 = torch.sum(gamma_tar_1*gamma_tar_1, dim=1)
    #     Qtar_2 = torch.sum(gamma_tar_2*gamma_tar_2, dim=1)

    #     Qtar_12 = torch.matmul(gamma_tar_1, gamma_tar_2.t())
    #     logging.info('   time8={}'.format(time.time()-t))
    #     return self._llr_from_Qs(Qtar_12, Qtar_1, Qtar_2,
    #                              Qnon_1, Qnon_2,
    #                              logLtar, logLnon, logLnon)

    # def llr_1vs1_and_self(self, x1, x2, aux_comps=None):
    #     t=time.time()
    #     if self.lnorm:
    #         x1 = self.l2_norm(x1)
    #         x2 = self.l2_norm(x2)
    #     logging.info('   time0={}'.format(time.time()-t))

    #     t=time.time()
    #     if aux_comps is None:
    #         aux_comps = self._compute_aux_llr_1vs1()
    #     WV, icholLnon, icholLtar, logLnon, logLtar = aux_comps
    #     logging.info('   time1={}'.format(time.time()-t))
    #     t=time.time()
    #     VWF1 = torch.matmul(x1-self.mu, WV)
    #     VWF2 = torch.matmul(x2-self.mu, WV)
    #     logging.info('   time2={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_non_1, Qnon_1 = self._llr_compQ(VWF1, icholLnon)
    #     logging.info('   time3={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_non_2, Qnon_2 = self._llr_compQ(VWF2, icholLnon)
    #     logging.info('   time4={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_tar_1, Qtar_1 = self._llr_compQ(VWF1, icholLtar)
    #     logging.info('   time5={}'.format(time.time()-t))
    #     t=time.time()
    #     gamma_tar_2, Qtar_2 = self._llr_compQ(VWF2, icholLtar)
    #     logging.info('   time6={}'.format(time.time()-t))
    #     t=time.time()
    #     Qtar_12 = torch.matmul(gamma_tar_1, gamma_tar_2.t())
    #     logging.info('   time7={}'.format(time.time()-t))
    #     t=time.time()
    #     llr_1vs1 =  self._llr_from_Qs(Qtar_12, Qtar_1, Qtar_2,
    #                                   Qnon_1, Qnon_2,
    #                                   logLtar, logLnon, logLnon)
    #     logging.info('   time8={}'.format(time.time()-t))
    #     t=time.time()
    #     Qtar_11 = torch.matmul(gamma_tar_1, gamma_tar_1.t())
    #     logging.info('   time9={}'.format(time.time()-t))
    #     t=time.time()
    #     llr_self =  self._llr_from_Qs(Qtar_11, Qtar_1, Qtar_1,
    #                                   Qnon_1, Qnon_1,
    #                                   logLtar, logLnon, logLnon)
    #     logging.info('   time10={}'.format(time.time()-t))
    #     return llr_1vs1, llr_self
