"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Optional, Tuple, Union

import numpy as np

from ...hyp_defs import float_cpu
from ...utils.math_funcs import (
    fullcov_varfloor,
    int2onehot,
    invert_pdmat,
)
from .linear_gbe import LinearGBE


class LinearGBEUP(LinearGBE):
    """Linear Gaussian Back-end with uncertainty propagation.

    This variant assumes each input vector concatenates feature means and
    diagonal feature variances, i.e. ``x = [x_mean, x_var]`` with total shape
    ``(num_samples, 2 * x_dim)``. During scoring, each trial uncertainty
    ``diag(x_var)`` is added to the class covariance before evaluating linear
    scores.

    Attributes:
      mu: Class means with shape ``(num_classes, x_dim)``.
      W: Shared within-class precision with shape ``(x_dim, x_dim)``.
      update_mu: If True, class means are updated in ``fit``.
      update_W: If True, shared precision is updated in ``fit``.
      x_dim: Mean-feature dimension (excluding concatenated variance features).
      num_classes: Number of classes.
      balance_class_weight: If True, balances class contributions when estimating
        ``W``.
      beta: Gaussian-Wishart beta parameter per class.
      nu: Wishart degrees-of-freedom parameter.
      prior: Prior ``LinearGBE`` used for MAP adaptation.
      prior_beta: Optional override for prior beta relevance factor.
      prior_nu: Optional override for prior nu relevance factor.
      post_beta: Optional fixed posterior beta relevance factor.
      post_nu: Optional fixed posterior nu relevance factor.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.classifiers.linear_gbe_up import LinearGBEUP
      >>> x_mean = np.array([[0.1, 1.2], [1.0, -0.2], [0.3, 0.4], [1.2, 0.1]])
      >>> x_var = 0.05 * np.ones_like(x_mean)
      >>> x = np.hstack((x_mean, x_var))
      >>> y = np.array([0, 1, 2, 1], dtype=np.int64)
      >>> model = LinearGBEUP(num_classes=3, update_mu=True, update_W=True)
      >>> model.fit(x, class_ids=y)
      >>> scores = model.predict(x, eval_method="linear", normalize=False)
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        update_mu: bool = True,
        update_W: bool = True,
        x_dim: int = 1,
        num_classes: Optional[int] = None,
        balance_class_weight: bool = False,
        beta: Optional[np.ndarray] = None,
        nu: Optional[float] = None,
        prior: Optional[Union["LinearGBE", str]] = None,
        prior_beta: Optional[float] = 16,
        prior_nu: Optional[float] = 16,
        post_beta: Optional[float] = None,
        post_nu: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a LinearGBEUP model.

        Args:
          mu: Class means with shape ``(num_classes, x_dim)``.
          W: Shared within-class precision matrix with shape ``(x_dim, x_dim)``.
          update_mu: If True, update ``mu`` in ``fit``.
          update_W: If True, update ``W`` in ``fit``.
          x_dim: Input mean feature dimension (not counting concatenated variance
            features).
          num_classes: Number of classes.
          balance_class_weight: If True, re-balance each class contribution when
            estimating ``W``.
          beta: Gaussian-Wishart beta parameter per class.
          nu: Wishart degrees-of-freedom parameter.
          prior: Prior ``LinearGBE`` instance or path to a serialized model.
          prior_beta: Optional override for the prior beta relevance factor.
          prior_nu: Optional override for the prior nu relevance factor.
          post_beta: Optional fixed posterior beta relevance factor.
          post_nu: Optional fixed posterior nu relevance factor.
          **kwargs: Extra arguments forwarded to ``LinearGBE``.
        """

        super().__init__(
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
            **kwargs,
        )

    @staticmethod
    def _split_mean_var(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Splits concatenated mean/variance features.

        Args:
          x: Input matrix with shape ``(num_samples, 2 * x_dim)``.

        Returns:
          Tuple ``(x_mean, x_var)`` each with shape ``(num_samples, x_dim)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        feat_dim = x.shape[-1]
        if feat_dim % 2 != 0:
            raise ValueError(
                f"x second dimension must be even (mean+var), got shape={x.shape}"
            )
        half = feat_dim // 2
        return x[:, :half], x[:, half:]

    def eval_linear(self, x: np.ndarray) -> np.ndarray:
        """Evaluates class unnormalized log-likelihoods with uncertainty propagation.

        Args:
          x: Input features ``[x_mean, x_var]`` with shape ``(num_samples, 2*x_dim)``.

        Returns:
          Unnormalized log-likelihoods with shape ``(num_samples, num_classes)``.
        """
        x_m, x_s = self._split_mean_var(x)
        if np.any(x_s < 0):
            raise ValueError("x variance part must be non-negative")
        S = invert_pdmat(self.W, return_inv=True)[-1]

        logp = np.zeros((len(x), self.num_classes), dtype=float_cpu())
        for i in range(x.shape[0]):
            W_i = invert_pdmat(S + np.diag(x_s[i]), return_inv=True)[-1]
            A, b = self._compute_Ab_i(self.mu, W_i)
            logp[i] = np.dot(x_m[i], A) + b
        return logp

    def eval_llk(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for uncertainty-propagated inputs."""
        raise NotImplementedError("eval_llk is not implemented in LinearGBEUP")

    def eval_predictive(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for uncertainty-propagated inputs."""
        raise NotImplementedError("eval_predictive is not implemented in LinearGBEUP")

    def fit(
        self,
        x: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        p_theta: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the parameters of the model with uncertainty propagation.

        Args:
          x: Input features ``[x_mean, x_var]`` with shape ``(num_samples, 2*x_dim)``.
          class_ids: Integer vector with class ids in ``[0, num_classes)``.
          p_theta: Alternative to ``class_ids``, posterior class probabilities
            with shape ``(num_samples, num_classes)``.
          sample_weight: Per-sample weighting with shape ``(num_samples,)``.
        """
        x_m, x_s = self._split_mean_var(x)
        if np.any(x_s < 0):
            raise ValueError("x variance part must be non-negative")
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
    def _compute_Ab_i(mu: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes trial-specific linear scoring parameters.

        Args:
          mu: Class means with shape ``(num_classes, x_dim)``.
          W: Trial-specific precision matrix with shape ``(x_dim, x_dim)``.

        Returns:
          Tuple ``(A, b)`` for linear scoring.
        """
        A = np.dot(W, mu.T)
        b = -0.5 * np.sum(mu.T * A, axis=0)
        return A, b
