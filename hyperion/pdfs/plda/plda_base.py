"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

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
    def compute_py_g_x(self, D):
        pass

    def fit(
        self,
        x,
        class_ids=None,
        ptheta=None,
        sample_weight=None,
        x_val=None,
        class_ids_val=None,
        ptheta_val=None,
        sample_weight_val=None,
        epochs=20,
        ml_md="ml+md",
        md_epochs=None,
    ):

        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta)
        else:
            D = self.compute_stats_hard(x, class_ids)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(x_val, ptheta_val)
            else:
                D_val = self.compute_stats_hard(x_val, class_ids_val)

        if not self.is_init:
            self.initialize(D)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):

            stats = self.Estep(D)
            elbo[epoch] = self.elbo(stats)
            if x_val is not None:
                stats_val = self.Estep(D_val)
                elbo_val[epoch] = self.elbo(stats_val)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

        elbo_norm = elbo / np.sum(D[0])
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(D_val[0])
            return elbo, elbo_norm, elbo_val, elbo_val_norm

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
    def llr_1vs1(self, x1, x2):
        pass

    @abstractmethod
    def llr_NvsM_book(self, D1, D2):
        pass

    def fit_adapt_weighted_avg_model(
        self,
        x,
        class_ids=None,
        ptheta=None,
        sample_weight=None,
        x_val=None,
        class_ids_val=None,
        ptheta_val=None,
        sample_weight_val=None,
        epochs=20,
        ml_md="ml+md",
        md_epochs=None,
        plda0=None,
        w_mu=1,
        w_B=0.5,
        w_W=0.5,
    ):

        assert self.is_init
        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta)
        else:
            D = self.compute_stats_hard(x, class_ids)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(x_val, ptheta_val)
            else:
                D_val = self.compute_stats_hard(x_val, class_ids_val)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):

            stats = self.Estep(D)
            elbo[epoch] = self.elbo(stats)
            if x_val is not None:
                stats_val = self.Estep(D_val)
                elbo_val[epoch] = self.elbo(stats_val)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

            self.weighted_avg_model(plda0, w_mu, w_B, w_W)

        elbo_norm = elbo / np.sum(D[0])
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(D_val[0])
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    def fit_adapt(
        self,
        x,
        class_ids=None,
        ptheta=None,
        sample_weight=None,
        x0=None,
        class_ids0=None,
        ptheta0=None,
        sample_weight0=None,
        x_val=None,
        class_ids_val=None,
        ptheta_val=None,
        sample_weight_val=None,
        epochs=20,
        ml_md="ml+md",
        md_epochs=None,
    ):

        assert self.is_init
        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta)
        else:
            D = self.compute_stats_hard(x, class_ids)

        if x0 is not None:
            assert not (class_ids0 is None and ptheta0 is None)
            if class_ids0 is None:
                D0 = self.compute_stats_soft(x0, ptheta0)
            else:
                D0 = self.compute_stats_hard(x0, class_ids0)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(x_val, ptheta_val)
            else:
                D_val = self.compute_stats_hard(x_val, class_ids_val)

        elbo = np.zeros((epochs,), dtype=float_cpu())
        elbo_val = np.zeros((epochs,), dtype=float_cpu())
        for epoch in range(epochs):

            stats = self.Estep(D)
            stats0 = self.Estep(D0)
            elbo[epoch] = self.elbo(stats)
            if x_val is not None:
                stats_val = self.Estep(D_val)
                elbo_val[epoch] = self.elbo(stats_val)

            if use_ml:
                self.MstepML(stats)
            if use_md and (md_epochs is None or epoch in md_epochs):
                self.MstepMD(stats)

        elbo_norm = elbo / np.sum(D[0])
        if x_val is None:
            return elbo, elbo_norm
        else:
            elbo_val_norm = elbo_val / np.sum(D_val[0])
            return elbo, elbo_norm, elbo_val, elbo_val_norm

    @staticmethod
    def compute_stats_soft(x, p_theta, sample_weight=None, scal_factor=None):
        if sample_weight is not None:
            p_theta = sample_weight[:, None] * p_theta
        if scal_factor is not None:
            p_theta *= scal_factor
        N = np.sum(p_theta, axis=0)
        F = np.dot(p_theta.T, x)
        wx = np.sum(p_theta, axis=1, keepdims=True) * x
        S = np.dot(x.T, wx)
        return N, F, S

    @staticmethod
    def compute_stats_hard(x, class_ids, sample_weight=None, scale_factor=None):
        x_dim = x.shape[1]
        num_classes = np.max(class_ids) + 1
        N = np.zeros((num_classes,), dtype=float_cpu())
        F = np.zeros((num_classes, x_dim), dtype=float_cpu())
        if sample_weight is not None:
            wx = sample_weight[:, None] * x
        else:
            wx = x

        for i in range(num_classes):
            idx = class_ids == i
            if sample_weight is None:
                N[i] = np.sum(idx).astype(float_cpu())
                F[i] = np.sum(x[idx], axis=0)
            else:
                N[i] = np.sum(sample_weight[idx])
                F[i] = np.sum(wx[idx], axis=0)

        S = np.dot(x.T, wx)
        if scale_factor is not None:
            N *= scale_factor
            F *= scale_factor
            S *= scale_factor

        return N, F, S

    @staticmethod
    def compute_stats_hard_v0(x, class_ids, sample_weight=None, scal_factor=None):
        x_dim = x.shape[1]
        num_classes = np.max(class_ids) + 1
        p_theta = np.zeros((x.shape[0], num_classes), dtype=float_cpu())
        p_theta[np.arange(x.shape[0]), class_ids] = 1
        return PLDABase.compute_stats_soft(x, p_theta, sample_weight, scal_factor)

    @staticmethod
    def center_stats(D, mu):
        N, F, S = D
        Fc = F - np.outer(N, mu)
        Fmu = np.outer(np.sum(F, axis=0), mu)
        Sc = S - Fmu - Fmu.T + np.sum(N) * np.outer(mu, mu)
        return N, Fc, Sc

    def llr_NvsM(self, x1, x2, ids1=None, ids2=None, method="vavg-lnorm"):
        if method == "savg":
            return self.llr_NvsM_savg(x1, ids1, x2, ids2)

        D1 = x1 if ids1 is None else self.compute_stats_hard(x1, class_ids=ids1)
        D2 = x2 if ids2 is None else self.compute_stats_hard(x2, class_ids=ids2)

        if method == "book":
            return self.llr_NvsM_book(D1, D2)
        if method == "vavg":
            return self.llr_NvsM_vavg(D1, D2, do_lnorm=False)
        if method == "vavg-lnorm":
            return self.llr_NvsM_vavg(D1, D2, do_lnorm=True)

    def llr_NvsM_vavg(self, D1, D2, do_lnorm=True):
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        x2 = D2[1] / np.expand_dims(D2[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_NvsM_savg(self, x1, ids1, x2, ids2):
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores_Nvs1 = F / N[:, None]
        N, F, _ = self.compute_stats_hard(scores_Nvs1.T, ids2)
        scores = F.T / N
        return scores

    def llr_Nvs1(self, x1, x2, ids1=None, method="vavg-lnorm"):
        if method == "savg":
            return self.llr_Nvs1_savg(x1, ids1, x2)

        D1 = x1 if ids1 is None else self.compute_stats_hard(x1, class_ids=ids1)

        if method == "book":
            D2 = self.compute_stats_hard(x2, np.arange(x2.shape[0]))
            return self.llr_NvsM_book(D1, D2)
        if method == "vavg":
            return self.llr_Nvs1_vavg(D1, x2, do_lnorm=False)
        if method == "vavg-lnorm":
            return self.llr_Nvs1_vavg(D1, x2, do_lnorm=True)

    def llr_Nvs1_vavg(self, D1, x2, do_lnorm=True):
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_Nvs1_savg(self, x1, ids1, x2):
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores = F / N[:, None]
        return scores

    @abstractmethod
    def sample(self, num_classes, num_samples_per_class, rng=None, seed=1024):
        pass

    def get_config(self):
        config = {"y_dim": self.y_dim, "update_mu": self.update_mu}
        base_config = super(PLDABase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def weigthed_avg_params(self, mu, w_mu):
        self.mu = w_mu * mu + (1 - w_mu) * self.mu

    @abstractmethod
    def weigthed_avg_model(self, plda):
        pass
