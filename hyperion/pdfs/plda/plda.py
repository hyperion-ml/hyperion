"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from scipy import linalg as sla

from ...hyp_defs import float_cpu
from ...utils.math import invert_pdmat, invert_trimat, logdet_pdmat
from .plda_base import PLDABase


class PLDA(PLDABase):
    def __init__(
        self,
        y_dim=None,
        z_dim=None,
        mu=None,
        V=None,
        U=None,
        D=None,
        floor_iD=1e-5,
        update_mu=True,
        update_V=True,
        update_U=True,
        update_D=True,
        **kwargs
    ):
        super(PLDA, self).__init__(y_dim=y_dim, mu=mu, update_mu=update_mu, **kwargs)
        self.z_dim = z_dim
        if V is not None:
            self.y_dim = V.shape[0]
        if U is not None:
            self.z_dim = U.shape[0]
        self.V = V
        self.U = U
        self.D = D
        self.floor_iD = floor_iD
        self.update_V = update_V
        self.update_U = update_U
        self.update_D = update_D

        # aux. vars
        self._DU = None
        self._Jt = None
        self._Lz = None
        self._mult_iLz = None
        self._log_Lz = None
        self._W = None
        self._VW = None
        self._VWV = None

    def validate(self):
        assert self.mu.shape[0] >= self.V.shape[0]
        assert self.mu.shape[0] == self.V.shape[1]
        assert self.mu.shape[0] >= self.U.shape[0]
        assert self.mu.shape[0] == self.U.shape[1]
        assert self.mu.shape[0] == self.D.shape[0]

    @property
    def is_init(self):
        if self._is_init:
            return True
        if (
            self.mu is not None
            and self.V is not None
            and self.U is not None
            and self.D is not None
        ):
            self.validate()
            if self._VWV is None:
                self.compute_aux()
            self._is_init = True
        return self._is_init

    def compute_aux(self):
        DV = self.V * self.D
        DU = self.U * self.D
        self._DU = DU
        self._J = np.dot(self.V, DU.T)
        self._Lz = np.eye(self.z_dim, dtype=float_cpu()) + np.dot(DU, self.U.T)
        self._mult_iLz, _, self._log_Lz = invert_pdmat(
            self._Lz, right_inv=True, return_logdet=True
        )
        DUiLz = self._mult_iLz(DU.T)
        self._W = np.diag(self.D) - np.dot(DUiLz, DU)
        self._VW = DV.T - np.dot(DUiLz, self._J.T)
        self._VWV = np.dot(self.V, self._VW)

    def initialize(self, D):
        N, F, S = D
        self.x_dim = F.shape[1]
        M = F.shape[0]
        N_tot = np.sum(N)

        Vytilde = F / N[:, None]
        mu = np.mean(Vytilde, axis=0)

        Vy = Vytilde - mu
        U, s, Vt = sla.svd(Vy, full_matrices=False, overwrite_a=True)
        V = s[: self.y_dim, None] * Vt[: self.y_dim, :]

        NVytilde = N[:, None] * Vytilde
        C = (S - np.dot(NVytilde.T, Vytilde)) / N_tot
        w, U = sla.eigh(C)
        U = np.fliplr(U * np.sqrt(w))[:, : self.z_dim].T

        iD = np.diag(C - np.dot(U.T, U)).copy()
        iD[iD < self.floor_iD] = self.floor_iD

        self.mu = mu
        self.V = V
        self.U = U
        self.D = 1 / iD
        self.compute_aux()

    def compute_py_g_x(
        self, D, return_cov=False, return_logpy_0=False, return_acc=False
    ):

        assert self.is_init

        N, F, S = D
        Fc = F - self.mu

        M = F.shape[0]
        y_dim = self.y_dim

        compute_inv = return_cov or return_acc
        return_tuple = compute_inv or return_logpy_0

        N_is_int = False
        if np.all(np.ceil(N) == N):
            N_is_int = True

        I = np.eye(y_dim, dtype=float_cpu())
        gamma = np.dot(Fc, self._VW)
        if N_is_int:
            iterator = np.unique(N)
        else:
            iterator = range(M)

        y = np.zeros((M, y_dim), dtype=float_cpu())
        if return_cov:
            Sigma_y = np.zeros((M, y_dim, y_dim), dtype=float_cpu())
        else:
            Sigma_y = None

        if return_logpy_0:
            logpy = -0.5 * y_dim * np.log(2 * np.pi) * np.ones((M,), dtype=float_cpu())

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

            L_i = I + N_i * self._VWV
            r = invert_pdmat(
                L_i,
                right_inv=True,
                return_logdet=return_logpy_0,
                return_inv=compute_inv,
            )

            mult_iL = r[0]
            if return_logpy_0:
                logL = r[2]
            if compute_inv:
                iL = r[-1]

            y[i, :] = mult_iL(gamma[i, :])

            if return_cov:
                Sigma_y[i, :, :] = iL

            if return_logpy_0:
                logpy[i] += 0.5 * (logL - np.sum(y[i, :] * gamma[i, :], axis=-1))

            if return_acc:
                Py += M_i * iL
                Ry += N_i * M_i * iL

        if not return_tuple:
            return y

        r = [y]
        if return_cov:
            r += [Sigma_y]
        if return_logpy_0:
            r += [logpy]
        if return_acc:
            r += [Ry, Py]
        return tuple(r)

    def Estep(self, D):
        N, F, S = D
        y, logpy, Ry, Py = self.compute_py_g_x(D, return_logpy_0=True, return_acc=True)

        M = F.shape[0]
        N_tot = np.sum(N)
        F_tot = np.sum(F, axis=0)

        # Cy
        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F.T, y)

        # Cz
        A = np.dot(S - np.dot(F_tot.T, self.mu), self._DU.T) - np.dot(Cy, self._J)
        Cz = self._mult_iLz(A)

        # Ry Ry1, Py
        Niy = y * N[:, None]
        Ry1 = np.sum(Niy, axis=0)
        Ry += np.dot(Niy.T, y)
        Py += np.dot(y.T, y)

        # acc logpy
        logpy_acc = np.sum(logpy)

        # Rz, Pz
        _, Fc, Sc = self.center_stats(D, self.mu)
        Fc_acc = np.sum(Fc, axis=0)
        Rz1 = self._mult_iLz(np.dot(Fc_acc, self._DU.T) - np.dot(Ry1, self._J))
        Cbary = Cy - np.outer(self.mu, Ry1)
        Ryz = self._mult_iLz(np.dot(Cbary.T, self._DU.T) - np.dot(Ry, self._J))
        A = np.dot(np.dot(self._DU, Cbary), self._J)
        B = (
            np.dot(np.dot(self._DU, Sc), self._DU.T)
            - A
            - A.T
            + np.dot(np.dot(self._J.T, Ry), self._J)
        )
        B = self._mult_iLz(B)
        Rz = self._mult_iLz(B.T).T + N_tot * self._mult_iLz(
            np.eye(self.z_dim, dtype=float_cpu())
        )

        stats = (
            N_tot,
            M,
            F_tot,
            S,
            logpy_acc,
            y_acc,
            Ry1,
            Ry,
            Cy,
            Py,
            Rz1,
            Rz,
            Ryz,
            Cz,
        )
        return stats

    def elbo(self, stats):
        N, M, F, S, logpy_x = stats[:5]

        logD = np.sum(np.log(self.D))
        Fmu = np.outer(F, self.mu)
        Shat = S - Fmu - Fmu.T + N * np.outer(self.mu, self.mu)

        logpx_y = 0.5 * (
            -N * self.x_dim * np.log(2 * np.pi)
            + N * (logD - self._log_Lz)
            - np.inner(self._W.ravel(), Shat.ravel())
        )
        logpy = -0.5 * M * self.y_dim * np.log(2 * np.pi)

        elbo = logpx_y + logpy - logpy_x
        return elbo

    def MstepML(self, stats):
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py, Rz1, Rz, Ryz, Cz = stats

        if self.update_mu and not self.update_V and not self.update_U:
            self.mu = (F - np.dot(Ry1, self.V) - np.dot(Rz1, self.U)) / N

        if not self.update_mu and self.update_V and not self.update_U:
            iRy_mult = invert_pdmat(Ry, right_inv=False)[0]
            C = Cy.T - np.outer(Ry1, self.mu) - np.dot(Ryz, self.U)
            self.V = iRy_mult(C)

        if not self.update_mu and not self.update_V and self.update_U:
            iRz_mult = invert_pdmat(Rz, right_inv=False)[0]
            C = Cz.T - np.dot(Ryz.T, self.V) - np.outer(Rz1, self.mu)
            self.U = iRz_mult(C)

        if not self.update_mu and self.update_V and self.update_U:
            a = np.hstack((Ry, Ryz))
            b = np.hstack((Ryz.T, Rz))
            Rytilde = np.vstack((a, b))
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            a = Cy.T - np.outer(Ry1, self.mu)
            b = Cz.T - np.outer(Rz1, self.mu)
            C = np.vstack((a, b))
            Vtilde = iRytilde_mult(C)
            self.V = Vtilde[: self.y_dim]
            self.U = Vtilde[self.y_dim :]

        if self.update_mu and not self.update_V and self.update_U:
            a = np.hstack((Rz, Rz1[:, None]))
            b = np.hstack((Rz1, N))
            Rytilde = np.vstack((a, b))
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            a = Cz.T - np.outer(Ryz, self.V)
            b = F[:, None] - np.outer(Ry1, self.V)
            C = np.vstack((a, b))
            Vtilde = iRytilde_mult(C)
            self.U = Vtilde[:-1]
            self.mu = Vtilde[-1]

        if not self.update_mu and self.update_V and not self.update_U:
            a = np.hstack((Ry, Ry1[:, None]))
            b = np.hstack((Ry1, N))
            Rytilde = np.vstack((a, b))
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            a = Cy.T - np.dot(Ryz, self.U)
            b = F[:, None] - np.dot(Rz1, self.U)
            C = np.vstack((a, b))
            Vtilde = iRytilde_mult(C)
            self.V = Vtilde[:-1]
            self.U = Vtilde[-1]

        a = np.hstack((Ry, Ryz, Ry1[:, None]))
        b = np.hstack((Ryz.T, Rz, Rz1[:, None]))
        c = np.hstack((Ry1, Rz1, N))
        Rytilde = np.vstack((a, b, c))
        Cytilde = np.hstack((Cy, Cz, F[:, None]))

        if self.update_mu and self.update_V and self.update_U:
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            Vtilde = iRytilde_mult(Cytilde.T)
            self.V = Vtilde[: self.y_dim, :]
            self.U = Vtilde[self.y_dim : -1]
            self.mu = Vtilde[-1]

        if self.update_D:
            Vtilde = np.vstack((self.V, self.U, self.mu))
            CVt = np.dot(Cytilde, Vtilde)
            iD = np.diag(
                (S - CVt - CVt.T + np.dot(np.dot(Vtilde.T, Rytilde), Vtilde)) / N
            ).copy()
            iD[iD < self.floor_iD] = self.floor_iD
            self.D = 1 / iD

        self.compute_aux()

    def MstepMD(self, stats):
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py, Rz1, Rz, Ryz, Cz = stats
        mu_y = y_acc / M
        Cov_y = Py / M - np.outer(mu_y, mu_y)
        chol_Cov_y = sla.cholesky(Cov_y, lower=False, overwrite_a=True)

        R = Ry - np.outer(Ry1, Ry1) / N
        mult_iR = invert_pdmat(R, right_inv=True)[0]
        H = mult_iR(Ryz.T - np.outer(Rz1, Ry1) / N)
        mu_z = (Rz1 - np.dot(Ry1, H.T)) / N
        RzyH = np.dot(Ryz.T, H.T)
        Cov_z = (Rz - RzyH - RzyH.T + np.dot(np.dot(H, Ry), H.T)) / N - np.outer(
            mu_z, mu_z
        )
        chol_Cov_z = sla.cholesky(Cov_z, lower=False, overwrite_a=True)

        if self.update_mu:
            self.mu += np.dot(mu_y, self.V + np.dot(H.T, self.U)) + np.dot(mu_z, self.U)

        if self.update_V:
            self.V = np.dot(chol_Cov_y, self.V + np.dot(H.T, self.U))

        if self.update_U:
            self.U = np.dot(chol_Cov_z, self.U)

        self.compute_aux()

    def get_config(self):
        config = {
            "update_D": self.update_D,
            "update_U": self.update_U,
            "update_V": self.update_V,
            "floor_iD": self.floor_iD,
        }
        base_config = super(PLDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"mu": self.mu, "V": self.V, "U": self.U, "D": self.D}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "V", "U", "D"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def log_probx_g_y(self, x, y):
        iW = np.diag(1 / self.D) + np.dot(self.U.T, self.U)
        mult_W, _, logiW = invert_pdmat(iW, return_logdet=True)
        delta = x - self.mu - np.dot(y, self.V)
        logp = (
            -x.shape[-1] * np.log(2 * np.pi)
            - logiW
            - np.sum(mult_W(delta) * delta, axis=-1)
        )
        logp /= 2
        return logp

    def log_probx_g_yz(self, x, y, z):
        logD = np.sum(np.log(self.D))
        delta = x - self.mu - np.dot(y, self.V) - np.dot(z, self.U)
        logp = (
            -x.shape[-1] * np.log(2 * np.pi)
            + logD
            - np.sum(self.D * delta ** 2, axis=-1)
        )
        logp /= 2
        return logp

    def llr_1vs1(self, x1, x2):

        assert self.is_init
        WV = self._VW
        VV = self._VWV
        I = np.eye(self.y_dim, dtype=float_cpu())

        Lnon = I + VV
        mult_icholLnon, logcholLnon = invert_trimat(
            sla.cholesky(Lnon, lower=False, overwrite_a=True),
            right_inv=True,
            return_logdet=True,
        )[:2]
        logLnon = 2 * logcholLnon

        Ltar = I + 2 * VV
        mult_icholLtar, logcholLtar = invert_trimat(
            sla.cholesky(Ltar, lower=False, overwrite_a=True),
            right_inv=True,
            return_logdet=True,
        )[:2]
        logLtar = 2 * logcholLtar

        VWF1 = np.dot(x1 - self.mu, WV)
        VWF2 = np.dot(x2 - self.mu, WV)

        gamma_non_1 = mult_icholLnon(VWF1)
        gamma_non_2 = mult_icholLnon(VWF2)

        Qnon_1 = np.sum(gamma_non_1 * gamma_non_1, axis=1)[:, None]
        Qnon_2 = np.sum(gamma_non_2 * gamma_non_2, axis=1)

        gamma_tar_1 = mult_icholLtar(VWF1)
        gamma_tar_2 = mult_icholLtar(VWF2)

        Qtar_1 = np.sum(gamma_tar_1 * gamma_tar_1, axis=1)[:, None]
        Qtar_2 = np.sum(gamma_tar_2 * gamma_tar_2, axis=1)

        scores = 2 * np.dot(gamma_tar_1, gamma_tar_2.T)
        scores += Qtar_1 - Qnon_1 + Qtar_2 - Qnon_2
        scores += 2 * logLnon - logLtar
        scores *= 0.5
        return scores

    def llr_NvsM_book(self, D1, D2):

        assert self.is_init

        N1, F1, _ = D1
        N2, F2, _ = D2

        WV = self._WV
        VV = self._VWV
        I = np.eye(self.y_dim, dtype=float_cpu())

        F1 -= N1[:, None] * self.mu
        F2 -= N2[:, None] * self.mu

        scores = np.zeros((len(N1), len(N2)), dtype=float_cpu())
        for N1_i in np.unique(N1):
            for N2_j in np.unique(N2):
                i = np.where(N1 == N1_i)[0]
                j = np.where(N2 == N2_j)[0]

                L1 = I + N1_i * VV
                mult_icholL1, logcholL1 = invert_trimat(
                    sla.cholesky(L1, lower=False, overwrite_a=True),
                    right_inv=True,
                    return_logdet=True,
                )[:2]
                logL1 = 2 * logcholL1

                L2 = I + N2_j * VV
                mult_icholL2, logcholL2 = invert_trimat(
                    sla.cholesky(L2, lower=False, overwrite_a=True),
                    right_inv=True,
                    return_logdet=True,
                )[:2]
                logL2 = 2 * logcholL2

                Ltar = I + (N1_i + N2_j) * VV
                mult_icholLtar, logcholLtar = invert_trimat(
                    sla.cholesky(Ltar, lower=False, overwrite_a=True),
                    right_inv=True,
                    return_logdet=True,
                )[:2]
                logLtar = 2 * logcholLtar

                VWF1 = np.dot(F1[i, :], WV)
                VWF2 = np.dot(F2[j, :], WV)

                gamma_non_1 = mult_icholL1(VWF1)
                gamma_non_2 = mult_icholL2(VWF2)

                Qnon_1 = np.sum(gamma_non_1 * gamma_non_1, axis=1)[:, None]
                Qnon_2 = np.sum(gamma_non_2 * gamma_non_2, axis=1)

                gamma_tar_1 = mult_icholLtar(VWF1)
                gamma_tar_2 = mult_icholLtar(VWF2)

                Qtar_1 = np.sum(gamma_tar_1 * gamma_tar_1, axis=1)[:, None]
                Qtar_2 = np.sum(gamma_tar_2 * gamma_tar_2, axis=1)

                scores_ij = 2 * np.dot(gamma_tar_1, gamma_tar_2.T)
                scores_ij += Qtar_1 - Qnon_1 + Qtar_2 - Qnon_2
                scores_ij += logL1 + logL2 - logLtar
                scores[np.ix_(i, j)] = scores_ij

        scores *= 0.5
        return scores

    def sample(self, num_classes, num_samples_per_class, rng=None, seed=1024):
        if rng is None:
            rng = np.random.RandomState(seed=seed)

        x_dim = self.mu.shape[0]

        z1 = rng.normal(size=(num_classes * num_samples_per_class, x_dim)).astype(
            dtype=float_cpu(), copy=False
        )
        z1 /= self.D

        z2 = rng.normal(size=(num_classes * num_samples_per_class, self.z_dim)).astype(
            dtype=float_cpu(), copy=False
        )
        z2 = np.dot(z2, self.U)
        y = rng.normal(size=(num_classes, self.y_dim)).astype(
            dtype=float_cpu(), copy=False
        )
        y = np.dot(y, self.V) + self.mu
        y = np.repeat(y, num_samples_per_class, axis=0)

        return y + z1 + z2

    def weighted_avg_params(self, mu, V, U, D, w_mu, w_B, w_W):

        super(PLDA, self).weigthed_avg_params(mu, w_mu)
        if w_B > 0:
            Sb0 = np.dot(self.V.T, self.V)
            Sb = np.dot(V.T, V)
            Sb = w_B * Sb + (1 - w_B) * Sb0
            w, V = sla.eigh(Sb, overwrite_a=True)
            V = np.sqrt(w) * V
            V = V[:, -self.y_dim :]
            self.V = V.T

        if w_W > 0:
            Sw0 = np.dot(self.U.T, self.U) + np.diag(1 / self.D)
            Sw = np.dot(U.T, U) + np.diag(1 / D)
            Sw = w_W * Sw + (1 - w_W) * Sw0
            w, U = sla.eigh(Sw, overwrite_a=False)
            U = np.sqrt(w) * U
            U = U[:, -self.z_dim :]
            self.U = U.T
            iD = np.diag(Sw - np.dot(self.U.T, self.U)).copy()
            # print(Sw[:10,:10])
            # print(np.dot(self.U.T, self.U))
            # print(iD[:10])
            iD[iD < self.floor_iD] = self.floor_iD
            self.D = 1 / iD

        # if w_W > 0:
        #     Sw0 = np.dot(self.U.T, self.U)
        #     Sw = np.dot(U.T, U)
        #     Sw = w_W*Sw + (1-w_W)*Sw0
        #     w, U = sla.eigh(Sw, overwrite_a=True)
        #     U = np.sqrt(w)*U
        #     U = U[:,-self.z_dim:]
        #     self.U = U.T

        # if w_D > 0:
        #     Sd0 = 1/self.D
        #     Sd = 1/D
        #     Sd = w_D*Sd + (1-w_D)*Sd0
        #     self.D = 1/Sd

    def weighted_avg_model(self, plda, w_mu, w_B, w_W):
        self.weighted_avg_params(plda.mu, plda.V, plda.U, plda.D, w_mu, w_B, w_W)
