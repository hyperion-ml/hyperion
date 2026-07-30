"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Optional

import numpy as np

from ..pdfs.mixtures.gmm_tied_diag_cov import GMMTiedDiagCov as GMM
from .gauss_calibration import GaussCalibration


class UnsupGaussCalibration(GaussCalibration):
    """Class for unsupervised Gaussian calibration.
       The model assumes that target and non-target score distributions are Gaussians
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
        self,
        mu1: Optional[float] = None,
        mu2: Optional[float] = None,
        sigma2: Optional[float] = None,
        prior: float = 0.5,
        init_prior: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(mu1=mu1, mu2=mu2, sigma2=sigma2, prior=prior, **kwargs)
        self.init_prior: float = init_prior

    def fit(self, x: np.ndarray) -> None:
        """Estimates the parameters of the model.

        Args:
          x: score numpy tensor (num_scores,).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            raise ValueError(f"x must have shape (N,) or (N,1), got shape={x.shape}")
        if x.ndim == 1:
            if x.size == 0:
                raise ValueError("x must be non-empty")
            x = np.expand_dims(x, axis=-1)
        elif x.ndim == 2:
            if x.shape[0] == 0:
                raise ValueError("x must be non-empty")
            if x.shape[1] != 1:
                raise ValueError(f"x must have shape (N,1), got shape={x.shape}")
        else:
            raise ValueError(f"x must have shape (N,) or (N,1), got shape={x.shape}")

        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")

        if self.is_init():
            assert (
                self.mu1 is not None
                and self.mu2 is not None
                and self.sigma2 is not None
            )
            if not np.isfinite(self.sigma2) or self.sigma2 <= 0:
                raise ValueError(
                    f"sigma2 must be finite and > 0 before EM, got {self.sigma2}"
                )
            if not np.isfinite(self.prior) or not (0 < self.prior < 1):
                raise ValueError(
                    f"prior must be finite and in (0, 1), got {self.prior}"
                )
            mu1 = self.mu1
            mu2 = self.mu2
            sigma2 = np.expand_dims(self.sigma2, axis=-1)
            pi = np.array([self.prior, 1 - self.prior])
        else:
            if not np.isfinite(self.init_prior) or not (0 < self.init_prior < 1):
                raise ValueError(
                    f"init_prior must be finite and in (0, 1), got {self.init_prior}"
                )
            mu1 = np.max(x, axis=0, keepdims=True)
            mu2 = np.mean(x, axis=0, keepdims=True)
            sigma2 = np.std(x, axis=0, keepdims=True) ** 2
            pi = np.array([self.init_prior, 1 - self.init_prior])

        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
            raise ValueError(f"sigma2 must be finite and > 0 before EM, got {sigma2}")
        if (
            not np.all(np.isfinite(pi))
            or np.any(pi <= 0)
            or not np.isclose(np.sum(pi), 1)
        ):
            raise ValueError(f"pi must be finite, positive, and sum to 1, got {pi}")

        mu = np.vstack((mu1, mu2))
        gmm = GMM(mu=mu, Lambda=1 / sigma2, pi=pi)
        gmm.fit(x, epochs=20)

        self.mu1 = float(gmm.mu[0, 0])
        self.mu2 = float(gmm.mu[1, 0])
        self.sigma2 = float(gmm.Sigma[0])
        self.prior = float(gmm.pi[0])
        if not np.isfinite(self.sigma2) or self.sigma2 <= 0:
            raise ValueError(
                f"sigma2 must be finite and > 0 after EM, got {self.sigma2}"
            )
        if not np.isfinite(self.prior) or not (0 < self.prior < 1):
            raise ValueError(
                f"prior must be finite and in (0, 1) after EM, got {self.prior}"
            )

        self._compute_scale_bias()
