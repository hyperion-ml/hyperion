"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ....hyp_defs import float_cpu
from ..core.pdf import PDF
from ...transforms import LNorm


class PLDABase(PDF):
    """Abstract Base class for different versions of
    Probabilistic Linear Discriminant Analysis (PLDA) models.

    Attributes:
      y_dim: speaker factor dimension.
      mu: class-independent mean.
      update_mu: whether to update mu or not when training the model.
      x_dim: data dimension.
    """

    def __init__(self, y_dim=None, mu=None, update_mu=True, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.y_dim = y_dim
        self.update_mu = update_mu
        if mu is not None:
            self.x_dim = mu.shape[0]

    def initialize(self, D):
        """initializes the model.

        Args:
          D: tuple of sufficient statistics (N, F, S)
        """
        pass

    def compute_py_g_x(self, D):
        """Computes the posterior P(y|x)

        Args:
          D: tuple of sufficient statistics (N, F, S)
        """
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
        """Trains the model.

        Args:
          x: train data matrix with shape (num_samples, x_dim).
          class_ids: class identifiers [0, num_clases-1] for training data.
          ptheta: probability of belonging to a class with shape (num_samples, num_classes) for training data.
          sample_weight: weight of each sample in the training loss shape (num_samples,).
          x_val: validation data matrix with shape (num_val_samples, x_dim).
          class_ids_val: class identifiers [0, num_clases-1] for val data.
          ptheta_val: probability of belonging to a class with shape (num_samples, num_classes) for val. data.
          sample_weight_val: weight of each sample in the val. loss.
          epochs: number of EM steps.
          ml_md: whether to do maximum likelihood estimation ("ml"), minimum divergence ("md") or both ("ml+md").
          md_epochs: in which epochs to do MD estimation, if None, MD is done in all epochs.

        Returns:
          log p(X) of the training data.
          log p(x) per sample.
          log p(X) of the val. data, if present.
          log p(x) of the val. data per sample, if present.
        """

        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta, sample_weight=sample_weight)
        else:
            D = self.compute_stats_hard(x, class_ids, sample_weight=sample_weight)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(
                    x_val, ptheta_val, sample_weight=sample_weight_val
                )
            else:
                D_val = self.compute_stats_hard(
                    x_val, class_ids_val, sample_weight=sample_weight_val
                )

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

    def Estep(self, x):
        """Expectation step."""
        pass

    def MstepML(self, x):
        """Maximum likelihood step."""
        pass

    def MstepMD(self, x):
        """Minimum Divergence step."""
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
        """Adapts a PLDA model to new data. The adapted model is weighted averaged with the prior after each epoch.

        Args:
          x: train data matrix with shape (num_samples, x_dim).
          class_ids: class identifiers [0, num_clases-1] for training data.
          ptheta: probability of belonging to a class with shape (num_samples, num_classes) for training data.
          sample_weight: weight of each sample in the training loss shape (num_samples,).
          x_val: validation data matrix with shape (num_val_samples, x_dim).
          class_ids_val: class identifiers [0, num_clases-1] for val data.
          ptheta_val: probability of belonging to a class with shape (num_samples, num_classes) for val. data.
          sample_weight_val: weight of each sample in the val. loss.
          epochs: number of EM steps.
          ml_md: whether to do maximum likelihood estimation ("ml"), minimum divergence ("md") or both ("ml+md").
          md_epochs: in which epochs to do MD estimation, if None, MD is done in all epochs.
          plda0: prior model.
          w_mu: weigth of the prior on the mean.
          w_B: weight of the prior on the between-class precision.
          w_W: weight of the prior on the within-class precision.

        Returns:
          log p(X) of the training data.
          log p(x) per sample.
          log p(X) of the val. data, if present.
          log p(x) of the val. data per sample, if present.
        """

        assert self.is_init
        use_ml = False if ml_md == "md" else True
        use_md = False if ml_md == "ml" else True

        assert not (class_ids is None and ptheta is None)
        if class_ids is None:
            D = self.compute_stats_soft(x, ptheta, sample_weight=sample_weight)
        else:
            D = self.compute_stats_hard(x, class_ids, sample_weight=sample_weight)

        if x_val is not None:
            assert not (class_ids_val is None and ptheta_val is None)
            if class_ids_val is None:
                D_val = self.compute_stats_soft(
                    x_val, ptheta_val, sample_weight=sample_weight_val
                )
            else:
                D_val = self.compute_stats_hard(
                    x_val, class_ids_val, sample_weight=sample_weight_val
                )

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

    @staticmethod
    def compute_stats_soft(x, p_theta, sample_weight=None, scal_factor=None):
        """Computes sufficient statistics need by PLDA model using soft class assigments.

        Args:
          x: input data with shape (num_samples, x_dim)
          p_theta: soft class assigments with shape (num_samples, num_classes)
          sample_weight: weight of each sample for training with shape (num_samples, )
          scal_factor: scaling factor for sufficient statistics (Themos factor)

        Returns:
          N: zero order stats with shape (num_classes,)
          F: first order stats with shape (num_classes, x_dim)
          S: Accumulated second order stats with sahpe (x_dim, x_dim)
        """
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
        """Computes sufficient statistics need by PLDA model using soft class assigments.

        Args:
          x: input data with shape (num_samples, x_dim)
          class_ids: integer [0, num_classes-1] vector indicating the class of each sample.
          sample_weight: weight of each sample for training with shape (num_samples, )
          scal_factor: scaling factor for sufficient statistics (Themos factor)

        Returns:
          N: zero order stats with shape (num_classes,)
          F: first order stats with shape (num_classes, x_dim)
          S: Accumulated second order stats with sahpe (x_dim, x_dim)
        """
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
        """Centers the sufficient statistics by the PLDA mean.

        Args:
           D: tupe with sufficient stats (N, F, S).
           mu: mean vector.

        Returns:
          Centered N, F, S
        """
        N, F, S = D
        Fc = F - np.outer(N, mu)
        Fmu = np.outer(np.sum(F, axis=0), mu)
        Sc = S - Fmu - Fmu.T + np.sum(N) * np.outer(mu, mu)
        return N, Fc, Sc

    def llr_1vs1(self, x1, x2):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of one enrollment and one test segments.

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_enroll_segmens, x_dim).

        Returns:
          Score matrix with shape (num_enrollment_segments, num_test_segments).
        """
        pass

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
        pass

    def llr_NvsM(self, x1, x2, ids1=None, ids2=None, method="vavg-lnorm"):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_enroll_segmens, x_dim).
          ids1: integer array mapping from segments to
                enrollment-sides in [0, num_enroll_sides-1]
          ids2: integer array mapping from segments to
                test-sides in [0, num_test_sides-1]
          method: evaluation method in ["book" (exact formula),
            "vavg" (vector averaging), "vavg-lnorm" (vector averagin + lnorm),
            "savg" (score averaging)]

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
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
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side
        evaluated with vector averaging.

        Args:
          D1: tuple of sufficient statistics for the enrollment sides (N1, F1, S1).
          D2: tuple of sufficient statistics for the test sides (N2, F2, S2).
          do_lnorm: whether or not to do length norm. after vector averaging.

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        x2 = D2[1] / np.expand_dims(D2[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_NvsM_savg(self, x1, ids1, x2, ids2):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_enroll_segmens, x_dim).
          ids1: integer array mapping from segments to
                enrollment-sides in [0, num_enroll_sides-1]
          ids2: integer array mapping from segments to
                test-sides in [0, num_test_sides-1]

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores_Nvs1 = F / N[:, None]
        N, F, _ = self.compute_stats_hard(scores_Nvs1.T, ids2)
        scores = F.T / N
        return scores

    def llr_Nvs1(self, x1, x2, ids1=None, method="vavg-lnorm"):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_test_segmens, x_dim).
          ids1: integer array mapping from segments to
                enrollment-sides in [0, num_enroll_sides-1]
          method: evaluation method in ["book" (exact formula),
            "vavg" (vector averaging), "vavg-lnorm" (vector averagin + lnorm),
            "savg" (score averaging)]

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
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
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side
        evaluated with vector averaging.

        Args:
          D1: tuple of sufficient statistics for the enrollment sides (N1, F1, S1).
          x2: test vectors with shape (num_test_segmens, x_dim).
          do_lnorm: whether or not to do length norm. after vector averaging.

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
        x1 = D1[1] / np.expand_dims(D1[0], axis=-1)
        if do_lnorm:
            lnorm = LNorm()
            x1 = lnorm.predict(x1)
            x2 = lnorm.predict(x2)

        return self.llr_1vs1(x1, x2)

    def llr_Nvs1_savg(self, x1, ids1, x2):
        """log-likelihood ratio between target and non-target hypothesis for
        the case of N segments/enrollment-side and M segments/test-side

        Args:
          x1: enrollment vectors with shape (num_enroll_segmens, x_dim).
          x2: test vectors with shape (num_enroll_segmens, x_dim).
          ids1: integer array mapping from segments to
                enrollment-sides in [0, num_enroll_sides-1]

        Returns:
          Score matrix with shape (num_enrollment_sides, num_test_sides).
        """
        scores_1vs1 = self.llr_1vs1(x1, x2)
        N, F, _ = self.compute_stats_hard(scores_1vs1, ids1)
        scores = F / N[:, None]
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
        pass

    def get_config(self):
        """Returns the model configuration dict."""
        config = {"y_dim": self.y_dim, "update_mu": self.update_mu}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def weigthed_avg_params(self, mu, w_mu):
        """Performs weighted average of the model parameters
        and some given parameters.

        Args:
          mu: other mean vector
          w_mu: weight of the given mean vector.

        """
        self.mu = w_mu * mu + (1 - w_mu) * self.mu

    def weigthed_avg_model(self, plda):
        """Performs weighted average of the model parameters
        and those of another model given as input.

        Args:
          plda: other PLDA model.

        """
        pass
