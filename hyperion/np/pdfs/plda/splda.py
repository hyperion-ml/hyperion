"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
from scipy import linalg as sla

from ....hyp_defs import float_cpu
from ....utils.math import invert_pdmat, invert_trimat, logdet_pdmat
from .plda_base import PLDABase


class SPLDA(PLDABase):
    """Class for Simplied Probabilistic Discriminant Analysis (SPLDA).
     .. math::
        \mathbf{x}_{ij} = \mu + \mathbf{V} \mathbf{y}_i + \varepsilon_{ij}

    Attributes:
      y_dim: speaker factor dimension.
      mu: class-independent mean.
      V: speaker factor loading matrix.
      W: within-class precision.
      fullcov_W: whether W is full-precision matrix or not.
      update_mu: whether to update mu or not when training the model.
      update_V: whether to update V or not when training the model.
      update_W: whether to update W or not when training the model.
      x_dim: data dimension.
    """

    def __init__(
        self,
        y_dim=None,
        mu=None,
        V=None,
        W=None,
        fullcov_W=True,
        update_mu=True,
        update_V=True,
        update_W=True,
        **kwargs
    ):
        super().__init__(y_dim=y_dim, mu=mu, update_mu=update_mu, **kwargs)
        if V is not None:
            self.y_dim = V.shape[0]
        self.V = V
        self.W = W
        self.fullcov_W = fullcov_W
        self.update_V = update_V
        self.update_W = update_W

    def validate(self):
        """Validates the model parameters."""
        assert self.mu.shape[0] >= self.V.shape[0]
        assert self.mu.shape[0] == self.V.shape[1]
        assert self.mu.shape[0] == self.W.shape[0]
        assert self.mu.shape[0] == self.W.shape[1]

    @property
    def is_init(self):
        """Returns True if the model has been initialized."""
        if self._is_init:
            return True
        if self.mu is not None and self.V is not None and self.W is not None:
            self.validate()
            self._is_init = True
        return self._is_init

    def initialize(self, D):
        """initializes the model.

        Args:
          D: tuple of sufficient statistics (N, F, S)
        """
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
        if self.fullcov_W:
            W = invert_pdmat(C, return_inv=True)[-1]
        else:
            W = 1 / np.diag(C)

        self.mu = mu
        self.V = V
        self.W = W

    def compute_py_g_x(
        self, D, return_cov=False, return_logpy_0=False, return_acc=False
    ):
        """Computes the posterior P(y|x)

        Args:
          D: tuple of sufficient statistics (N, F, S)
          return_cov: whether or not to return the posterior covariances.
          return_logpy_0: whether or not to return log P(y=0|x).
          return_acc: whether or not to return Ry and Py accumulators.

        Returns:
          Speaker factor posterior means with shape (num_speakers, y_dim)
          Speaker factor posterior convariances with shape (num_speakers, y_dim, y_dim)
          log P(y=0|x) with shape (num_spakers,)
          Ry accumlator for ML step with shape (y_dim, y_dim)
          Py accumlator for MD step with shape (y_dim, y_dim)
        """
        N, F, S = D
        Fc = F - self.mu

        M = F.shape[0]
        y_dim = self.y_dim

        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)

        compute_inv = return_cov or return_acc
        return_tuple = compute_inv or return_logpy_0

        N_is_int = False
        if np.all(np.ceil(N) == N):
            N_is_int = True

        I = np.eye(y_dim, dtype=float_cpu())
        gamma = np.dot(Fc, WV)
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

            L_i = I + N_i * VV
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
        """Expectation step.

        Args:
          D: tuple with sufficient statistics (N, F, S)

        Returns:
          Tuple of statistics with accumlated expectations.
        """
        N, F, S = D
        y, logpy, Ry, Py = self.compute_py_g_x(D, return_logpy_0=True, return_acc=True)

        M = F.shape[0]
        N_tot = np.sum(N)
        F_tot = np.sum(F, axis=0)

        y_acc = np.sum(y, axis=0)
        Cy = np.dot(F.T, y)

        Niy = y * N[:, None]
        Ry1 = np.sum(Niy, axis=0)
        Ry += np.dot(Niy.T, y)
        Py += np.dot(y.T, y)

        logpy_acc = np.sum(logpy)

        stats = (N_tot, M, F_tot, S, logpy_acc, y_acc, Ry1, Ry, Cy, Py)
        return stats

    def elbo(self, stats):
        """Computes the objective function.

        Args:
          stats: tuple of expectations computed at the Estep.

        Returns:
         log P(X)
        """
        N, M, F, S, logpy_x = stats[:5]

        logW = logdet_pdmat(self.W)
        Fmu = np.outer(F, self.mu)
        Shat = S - Fmu - Fmu.T + N * np.outer(self.mu, self.mu)

        logpx_y = 0.5 * (
            -N * self.x_dim * np.log(2 * np.pi)
            + N * logW
            - np.inner(self.W.ravel(), Shat.ravel())
        )
        logpy = -0.5 * M * self.y_dim * np.log(2 * np.pi)

        elbo = logpx_y + logpy - logpy_x
        return elbo

    def MstepML(self, stats):
        """Maximum likelihood estimation step.

        Args:
          stats: tuple of expectations computed at the Estep.

        """
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py = stats

        a = np.hstack((Ry, Ry1[:, None]))
        b = np.hstack((Ry1, N))
        Rytilde = np.vstack((a, b))

        Cytilde = np.hstack((Cy, F[:, None]))

        if self.update_mu and not self.update_V:
            self.mu = (F - np.dot(Ry1, self.V)) / N

        if not self.update_mu and self.update_V:
            iRy_mult = invert_pdmat(Ry, right_inv=False)[0]
            self.V = iRy_mult(Cy.T - np.outer(Ry1, self.mu))

        if self.update_mu and self.update_V:
            iRytilde_mult = invert_pdmat(Rytilde, right_inv=False)[0]
            Vtilde = iRytilde_mult(Cytilde.T)
            self.V = Vtilde[:-1, :]
            self.mu = Vtilde[-1, :]

        if self.update_W:
            if self.update_mu and self.update_V:
                iW = (S - np.dot(Cy, self.V) - np.outer(F, self.mu)) / N
            else:
                Vtilde = np.vstack((self.V, self.mu))
                CVt = np.dot(Cytilde, Vtilde)
                iW = (S - CVt - CVt.T + np.dot(np.dot(Vtilde.T, Rytilde), Vtilde)) / N
            if self.fullcov_W:
                self.W = invert_pdmat(iW, return_inv=True)[-1]
            else:
                self.W = np.diag(1 / np.diag(iW))

    def MstepMD(self, stats):
        """Minimum divergence estimation step.

        Args:
          stats: tuple of expectations computed at the Estep.

        """
        N, M, F, S, _, y_acc, Ry1, Ry, Cy, Py = stats
        mu_y = y_acc / M

        if self.update_mu:
            self.mu += np.dot(mu_y, self.V)

        if self.update_V:
            Cov_y = Py / M - np.outer(mu_y, mu_y)
            chol_Cov_y = sla.cholesky(Cov_y, lower=False, overwrite_a=True)
            self.V = np.dot(chol_Cov_y, self.V)

    def get_config(self):
        """Returns the model configuration dict."""
        config = {
            "update_W": self.update_W,
            "update_V": self.update_V,
            "fullcov_W": self.fullcov_W,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        """Saves the model paramters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "V": self.V, "W": self.W}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "V", "W"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def log_probx_g_y(self, x, y):
        """Computes logP(X|Y)

        Args:
          x: data samples with shape (num_samples, x_dim).
          y: speaker factors for each sample with shape (num_samples, y_dim).

        Returns:
          log P(X|Y) array with shape (num_samples,)
        """
        logW = logdet_pdmat(self.W)
        delta = x - self.mu - np.dot(y, self.V)
        logp = (
            -x.shape[-1] * np.log(2 * np.pi)
            + logW
            - np.sum(np.dot(delta, self.W) * delta, axis=-1)
        )
        logp /= 2
        return logp

    def llr_1vs1(self, x1, x2):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of one enrollment and one test segments.

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_enroll_segmens, x_dim).

        Returns:
          Score matrix with shape (num_enrollment_segments, num_test_segments).
        """
        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)
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
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side
        evaluated with the exact formula (by the book).

        Args:
          D1: tuple of sufficient statistics for the enrollment sides (N1, F1, S1).
          D2: tuple of sufficient statistics for the test sides (N2, F2, S2).

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
        N1, F1, _ = D1
        N2, F2, _ = D2

        WV = np.dot(self.W, self.V.T)
        VV = np.dot(self.V, WV)
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
        """Draws samples from the PLDA model.

        Args:
          num_classes: number of classes to sample.
          num_samples_per_class: number of samples to sample per each class.
          rng: random number generator.
          seed: random seed used if rng is None.

        Returns:
          Generated samples with shape (num_samples, x_dim).
        """
        if rng is None:
            rng = np.random.RandomState(seed=seed)

        Sw = invert_pdmat(self.W, return_inv=True)[-1]
        chol_Sw = sla.cholesky(Sw, lower=False)

        x_dim = self.mu.shape[0]
        z = rng.normal(size=(num_classes * num_samples_per_class, x_dim)).astype(
            dtype=float_cpu(), copy=False
        )
        z = np.dot(z, chol_Sw)
        y = rng.normal(size=(num_classes, self.y_dim)).astype(
            dtype=float_cpu(), copy=False
        )
        y = np.dot(y, self.V) + self.mu
        y = np.repeat(y, num_samples_per_class, axis=0)

        return y + z

    def weighted_avg_params(self, mu, V, W, w_mu, w_B, w_W):
        """Performs weighted average of the model parameters
        and some given parameters.

        Args:
          mu: other mean vector
          w_mu: weight of the given mean vector.

        """
        super().weigthed_avg_params(mu, w_mu)
        if w_B > 0:
            Sb0 = np.dot(self.V.T, self.V)
            Sb = np.dot(V.T, V)
            Sb = w_B * Sb + (1 - w_B) * Sb0
            w, V = sla.eigh(Sb, overwrite_a=True)
            w = w[-self.y_dim :]
            V = np.sqrt(w) * V[:, -self.y_dim :]
            self.V = V.T

        if w_W > 0:
            Sw0 = invert_pdmat(self.W, return_inv=True)[-1]
            Sw = invert_pdmat(W, return_inv=True)[-1]
            Sw = w_W * Sw + (1 - w_W) * Sw0
            self.W = invert_pdmat(Sw, return_inv=True)[-1]

    def weighted_avg_model(self, plda, w_mu, w_B, w_W):
        """Performs weighted average of the model parameters
        and those of another model given as input.

        Args:
          plda: other PLDA model.

        """
        self.weighted_avg_params(plda.mu, plda.V, plda.W, w_mu, w_B, w_W)

    def project(self, T, delta_mu=None):
        """Transforms the PLDA parameters given an affine transformation
        of the data.

        Args:
          T: data projection matrix.
          delta_mu: data shift vector.

        Returns:
          Projected PLDA model.
        """
        mu = self.mu
        if mu is not None:
            mu -= delta_mu
        mu = np.dot(mu, T)
        V = np.dot(self.V, T)
        Sw = invert_pdmat(self.W, return_inv=True)[-1]
        Sw = np.dot(T.T, np.dot(Sw, T))
        W = invert_pdmat(Sw, return_inv=True)[-1]

        return SPLDA(mu=mu, V=V, W=W, fullcov_W=True)
