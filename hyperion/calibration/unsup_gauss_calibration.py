"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import sys
import numpy as np

from ..pdfs.mixtures.diag_gmm_tiedcovs import DiagGMMTiedCovs as GMM
from .gauss_calibration import GaussCalibration


class UnsupGaussCalibration(GaussCalibration):
    """Class for unsupervised Gaussian calibration.
       The model assumes that targer and non-target score distributions are Gaussians
       with shared covariance.
       The model is trained using a mixture of two Gaussians using EM algorithm.

    Attributes:
      mu1: mean of the target score distribution.
      mu2: mean of the non-target score distribution.
      sigma2: shared variance of the target and non-target score distributions.
      prior: prior prob. for target trials. It is the weight of the target component of the GMM.
      init_prior: initial weight given to the target component of the GMM, when initializing the EM algorithm.
    """

    def __init__(
        self, mu1=None, mu2=None, sigma2=None, prior=0.5, init_prior=0.5, **kwargs
    ):
        super().__init__(mu1, mu2, sigma2, prior, **kwargs)
        self.init_prior = init_prior

    def fit(self, x):
        """Estimates the parameters of the model.

        Args:
          x: score numpy tensor (num_scores,).
        """

        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)

        if self.is_init():
            mu1 = self.mu1
            mu2 = self.mu2
            sigma2 = np.expand_dims(self.sigma2, axis=-1)
            pi = np.array([self.prior, 1 - self.prior])
        else:
            mu1 = np.max(x, axis=0, keepdims=True)
            mu2 = np.mean(x, axis=0, keepdims=True)
            sigma2 = np.std(x, axis=0, keepdims=True) ** 2
            pi = np.array([self.init_prior, 1 - self.init_prior])

        mu = np.vstack((mu1, mu2))
        gmm = GMM(mu=mu, Lambda=1 / sigma2, pi=pi)
        gmm.fit(x, epochs=20)

        self.mu1 = gmm.mu[0, 0]
        self.mu2 = gmm.mu[1, 0]
        self.sigma2 = gmm.Sigma[0]
        self.prior = gmm.pi[0]

        self._compute_scale_bias()
