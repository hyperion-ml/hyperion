"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np
from scipy.special import gammaln

from ...hyp_defs import float_cpu
from ..np_model import NPModel
from ...utils.math import (
    int2onehot,
    logdet_pdmat,
    invert_pdmat,
    softmax,
    fullcov_varfloor,
)
from .linear_gbe import LinearGBE


class LinearGBEUP(LinearGBE):
    def __init__(
        self,
        mu=None,
        W=None,
        update_mu=True,
        update_W=True,
        x_dim=1,
        num_classes=None,
        balance_class_weight=True,
        beta=None,
        nu=None,
        prior=None,
        prior_beta=None,
        prior_nu=None,
        post_beta=None,
        post_nu=None,
        **kwargs
    ):

        super(LinearGBEUP, self).__init__(
            mu=mu,
            W=W,
            update_mu=update_mu,
            update_W=update_W,
            x_dim=x_dim,
            num_classes=num_classes,
            balance_class_weight=balance_class_weight,
            beta=beta,
            nu=nu,
            prior=prior,
            prior_beta=prior_beta,
            prior_nu=prior_nu,
            post_beta=post_beta,
            post_nu=post_nu,
            **kwargs
        )

    def eval_linear(self, x):
        x_m = x[:, : x.shape[-1] / 2]
        x_s = x[:, x.shape[-1] / 2 :]
        try:
            S = invert_pdmat(self.W, return_inv=True)[-1]
        except:
            #            self.W += np.mean(np.diag(self.W))/1000*np.eye(x.shape[-1]/2)
            S = invert_pdmat(self.W, return_inv=True)[-1]

        logp = np.zeros((len(x), self.num_classes), dtype=float_cpu())
        for i in range(x.shape[0]):
            W_i = invert_pdmat(S + np.diag(x_s[i]), return_inv=True)[-1]
            A, b = self._compute_Ab_i(self.mu, W_i)
            logp[i] = np.dot(x_m[i], A) + b
        return logp

    def eval_llk(self, x):
        raise NotImplementedError
        logp = np.dot(x, self.A) + self.b
        K = 0.5 * logdet_pdmat(self.W) - 0.5 * self.x_dim * np.log(2 * np.pi)
        K += -0.5 * np.sum(np.dot(x, self.W) * x, axis=1, keepdims=True)
        logp += K
        return logp

    def eval_predictive(self, x):
        raise NotImplementedError
        K = self.W / self.nu
        c = self.nu + 1 - self.x_dim
        r = self.beta / (self.beta + 1)

        # T(mu, L, c) ; L = c r K

        logg = (
            gammaln((c + self.x_dim) / 2)
            - gammaln(c / 2)
            - 0.5 * self.x_dim * np.log(c * np.pi)
        )

        # 0.5*log|L| = 0.5*log|K| + 0.5*d*log(c r)
        logK = logdet_pdmat(K)
        logL_div_2 = 0.5 * logK + 0.5 * self.x_dim * r

        # delta2_0 = (x-mu)^T W (x-mu)
        delta2_0 = np.sum(np.dot(x, self.W) * x, axis=1, keepdims=True) - 2 * (
            np.dot(x, self.A) + self.b
        )
        # delta2 = (x-mu)^T L (x-mu) = c r delta0 / nu
        # delta2/c = r delta0 / nu
        delta2_div_c = r * delta2_0 / self.nu

        D = -0.5 * (c + self.x_dim) * np.log(1 + delta2_div_c)
        logging.debug(self.nu)
        logging.debug(c)
        logging.debug(self.x_dim)
        logging.debug(logg)
        logging.debug(logL_div_2.shape)
        logging.debug(D.shape)

        logp = logg + logL_div_2 + D
        return logp

    def fit(self, x, class_ids=None, p_theta=None, sample_weight=None):
        x_m = x[:, : x.shape[-1] / 2]
        x_s = x[:, x.shape[-1] / 2 :]
        x = x_m
        assert class_ids is not None or p_theta is not None

        do_map = True if self.prior is not None else False
        if do_map:
            self._load_prior()

        self.x_dim = x.shape[-1]
        if self.num_classes is None:
            if class_ids is not None:
                self.num_classes = np.max(class_ids) + 1
            else:
                self.num_classes = p_theta.shape[-1]

        if class_ids is not None:
            p_theta = int2onehot(class_ids, self.num_classes)

        if sample_weight is not None:
            p_theta = sample_weight[:, None] * p_theta

        N = np.sum(p_theta, axis=0)

        F = np.dot(p_theta.T, x)

        if self.update_mu:
            xbar = F / N[:, None]
            if do_map:
                alpha_mu = (N / (N + self.prior.beta))[:, None]
                self.mu = (1 - alpha_mu) * self.prior.mu + alpha_mu * xbar
                self.beta = N + self.prior.beta
            else:
                self.mu = xbar
                self.beta = N
        else:
            xbar = self.mu

        if self.update_W:
            if do_map:
                nu0 = self.prior.nu
                S0 = invert_pdmat(self.prior.W, return_inv=True)[-1]
                if self.balance_class_weight:
                    alpha_W = (N / (N + nu0 / self.num_classes))[:, None]
                    S = (self.num_classes - np.sum(alpha_W)) * S0
                else:
                    S = nu0 * S0
            else:
                nu0 = 0
                S = np.zeros((x.shape[1], x.shape[1]), dtype=float_cpu())

            for k in range(self.num_classes):
                delta = x - xbar[k]
                S_k = np.dot(p_theta[:, k] * delta.T, delta)
                if do_map and self.update_mu:
                    mu_delta = xbar[k] - self.prior.mu[k]
                    S_k += N[k] * (1 - alpha_mu[k]) * np.outer(mu_delta, mu_delta)

                if self.balance_class_weight:
                    S_k /= N[k] + nu0 / self.num_classes

                S += S_k

            if self.balance_class_weight:
                S /= self.num_classes
            else:
                S /= nu0 + np.sum(N)

            x_s_mean = np.diag(np.mean(x_s, axis=0))
            S = fullcov_varfloor(S, np.sqrt(x_s_mean) * 1.1)
            S -= x_s_mean

            self.W = invert_pdmat(S, return_inv=True)[-1]
            self.nu = np.sum(N) + nu0

        self._change_post_r()
        self._compute_Ab()

    @staticmethod
    def _compute_Ab_i(mu, W):
        A = np.dot(W, mu.T)
        b = -0.5 * np.sum(mu.T * A, axis=0)
        return A, b
