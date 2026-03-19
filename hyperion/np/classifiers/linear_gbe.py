"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from scipy.special import gammaln

from ...hyp_defs import float_cpu
from ...utils.math_funcs import int2onehot, invert_pdmat, logdet_pdmat, softmax
from ..hyper_np_model import HyperNPModel


class LinearGBE(HyperNPModel):
    """Linear Gaussian Back-end.

    Attributes:
      mu: mean of the classes (num_classes, x_dim)
      W: Within-class precision, shared for all classes (x_dim, x_dim)
      update_mu: if True, it updates the means when calling the fit function.
      update_W: if True, it updates the precision when calling the fit function.
      x_dim: dimension of the input features.
      num_classes: number of classes.
      balance_class_weight: if True, all classes have the same weight in the estimation of W.
      beta: beta param of Gaussian-Wishart distribution.
      nu: nu (deegres of freedom) param of Wishart distribution.
      prior: LinearGBE object containing a prior mean, precision, beta, nu (used for adaptation).
      prior_beta: if given, it overwrites beta in the prior object.
      prior_nu: if given, it overwrites nu in the prior object.
      post_beta: if given, it fixes the value of beta in the posterior, overwriting the beta computed by the fit function.
      post_nu: if given, it fixes the value of nu in the posterior, overwriting the beta computed by the fit function.
      labels: list of class labels.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.classifiers.linear_gbe import LinearGBE
      >>> x = np.array([[0.1, 1.2], [1.0, -0.2], [0.3, 0.4], [1.2, 0.1]])
      >>> y = np.array([0, 1, 2, 1], dtype=np.int64)
      >>> model = LinearGBE(num_classes=3, update_mu=True, update_W=True)
      >>> model.fit(x, class_ids=y)
      >>> scores = model.predict(x, eval_method="linear", normalize=False)
      >>> log_post = model.predict(x, eval_method="linear", normalize=True)
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
        labels: Optional[Union[np.ndarray, List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a LinearGBE model.

        Args:
          mu: Class means with shape ``(num_classes, x_dim)``.
          W: Shared within-class precision matrix with shape ``(x_dim, x_dim)``.
          update_mu: If True, update ``mu`` in ``fit``.
          update_W: If True, update ``W`` in ``fit``.
          x_dim: Input feature dimension.
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
          labels: Optional class labels.
          **kwargs: Extra arguments forwarded to ``HyperNPModel``.
        """

        super().__init__(**kwargs)
        if mu is not None:
            num_classes = mu.shape[0]
            x_dim = mu.shape[1]

        self.mu = mu
        self.W = W
        self.update_mu = update_mu
        self.update_W = update_W
        self.x_dim = x_dim
        self.num_classes = num_classes
        self.balance_class_weight = balance_class_weight
        self.A = None
        self.b = None
        self.prior = prior
        self.beta = beta
        self.nu = nu
        self.prior_beta = prior_beta
        self.prior_nu = prior_nu
        self.post_beta = post_beta
        self.post_nu = post_nu

        self.set_labels(labels)
        self._compute_Ab()

    def set_labels(self, labels: Optional[Union[np.ndarray, List[Any]]]) -> None:
        """Sets class labels.

        Args:
          labels: Class labels as a list/array or ``None``.
        """
        if isinstance(labels, np.ndarray):
            labels = list(labels)

        self.labels = labels

    def get_config(self) -> Dict[str, Any]:
        """Gets model hyperparameters.

        Returns:
          Dictionary with the hyperparameters of the model.
        """
        config = {
            "update_mu": self.update_mu,
            "update_W": self.update_W,
            "x_dim": self.x_dim,
            "num_classes": self.num_classes,
            "balance_class_weight": self.balance_class_weight,
            "prior_beta": self.prior_beta,
            "prior_nu": self.prior_nu,
            "post_beta": self.post_beta,
            "post_nu": self.post_nu,
            "labels": self.labels,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _load_prior(self) -> None:
        """Loads and configures the prior model for MAP adaptation."""
        if isinstance(self.prior, str):
            self.prior = LinearGBE.load(self.prior)
        num_classes = self.prior.mu.shape[0]
        if self.prior_beta is not None:
            self.prior.beta = self.prior_beta * np.ones(
                (num_classes,), dtype=float_cpu()
            )
        if self.prior_nu is not None:
            self.prior.nu = num_classes * self.prior_nu

    def _change_post_r(self) -> None:
        """Applies fixed posterior relevance factors if configured."""
        if self.post_beta is not None:
            self.beta = self.post_beta * np.ones((self.num_classes,), dtype=float_cpu())
        if self.post_nu is not None:
            self.nu = self.num_classes * self.post_nu

    def eval_linear(self, x: np.ndarray) -> np.ndarray:
        """Evals the class unnormalized log-likelihoods. which reduces to a linear function.

        Args:
          x: input features (num_trials, x_dim).

        Returns:
          Log-likelihoods (num_trials, num_classes).
        """
        return np.dot(x, self.A) + self.b

    def eval_llk(self, x: np.ndarray) -> np.ndarray:
        """Evals the class log-likelihoods

        Args:
          x: input features (num_trials, x_dim).

        Returns:
          Log-likelihoods (num_trials, num_classes).
        """

        logp = np.dot(x, self.A) + self.b
        K = 0.5 * logdet_pdmat(self.W) - 0.5 * self.x_dim * np.log(2 * np.pi)
        K += -0.5 * np.sum(np.dot(x, self.W) * x, axis=1, keepdims=True)
        logp += K
        return logp

    def eval_predictive(self, x: np.ndarray) -> np.ndarray:
        """Evals the log-predictive distribution, taking into account the uncertainty in mu and W.
            It involves evaluating the Student-t distributions. For this we need to give priors
            to the model parameters.

        Args:
          x: input features (num_trials, x_dim).

        Returns:
          Log-likelihoods (num_trials, num_classes).
        """

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

    def predict(
        self, x: np.ndarray, eval_method: str = "linear", normalize: bool = False
    ) -> np.ndarray:
        """Evaluates the Gaussian back-end.

        Args:
          x: input features (num_trials, x_dim).
          eval_method: evaluation method can be linear (evaluates linear function),
                       llk (evaluates exact log-likelihood),
                       or predictive (evaluates the predictive distribution).
          normalize: if True, normalize log-likelihoods transforming them into log-posteriors.

        Returns:
          Log-LLK or log-posterior scores (num_trials, num_classes).
        """
        if eval_method == "linear":
            logp = self.eval_linear(x)
        elif eval_method == "llk":
            logp = self.eval_llk(x)
        elif eval_method == "predictive":
            logp = self.eval_predictive(x)
        else:
            raise ValueError("wrong eval method %s" % eval_method)

        if normalize:
            logp = np.log(softmax(logp, axis=1))

        return logp

    def __call__(
        self, x: np.ndarray, eval_method: str = "linear", normalize: bool = False
    ) -> np.ndarray:
        """Evaluates the Gaussian back-end.

        Args:
          x: input features (num_trials, x_dim).
          eval_method: evaluation method can be linear (evaluates linear function),
                       llk (evaluates exact log-likelihood),
                       or predictive (evaluates the predictive distribution).
          normalize: if True, normalize log-likelihoods transforming them into log-posteriors.

        Returns:
          Log-LLK or log-posterior scores (num_trials, num_classes).
        """
        return self.predict(x, eval_method, normalize)

    def fit(
        self,
        x: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        p_theta: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the parameters of the model.

        Args:
          x: input features (num_samples, x_dim)
          class_ids: integer vector (num_samples,) with elements in [0, num_classes)
                     indicating the class of each example.
          p_theta: alternative to class_ids, it is a matrix (num_samples, num_classes)
                   indicating the prob. for example i to belong to class j.
          sample_weight: indicates the weight of each sample in the estimation of the parameters (num_samples,).
        """
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

            self.W = invert_pdmat(S, return_inv=True)[-1]
            self.nu = np.sum(N) + nu0

        self._change_post_r()
        self._compute_Ab()

    def save_params(self, f: Any) -> None:
        """Saves model parameters to an open HDF5 handle.

        Args:
          f: HDF5 file handle.
        """
        params = {"mu": self.mu, "W": self.W, "beta": self.beta, "nu": self.nu}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "LinearGBE":
        """Loads model parameters from an open HDF5 handle.

        Args:
          f: HDF5 file handle.
          config: Configuration dictionary.

        Returns:
          Initialized ``LinearGBE`` instance.
        """
        param_list = ["mu", "W", "beta", "nu"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def _compute_Ab(self) -> None:
        """Computes the rotation and bias parameters for the linear scoring."""
        if self.mu is not None and self.W is not None:
            self.A = np.dot(self.W, self.mu.T)
            self.b = -0.5 * np.sum(self.mu.T * self.A, axis=0)

    @staticmethod
    def filter_class_args(**kwargs: Any) -> Dict[str, Any]:
        """Extracts the hyperparams of the class from a dictionary.

        Returns:
          Hyperparameter dictionary to initialize the class.
        """
        valid_args = (
            "update_mu",
            "update_W",
            "balance_class_weight",
            "prior",
            "prior_beta",
            "prior_nu",
            "post_beta",
            "post_nu",
            "name",
        )
        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return d

    filter_train_args = filter_class_args

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """It adds the arguments corresponding to the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--update-mu",
            default=True,
            action=ActionYesNo,
            help="whether to update class means during fit",
        )
        parser.add_argument(
            "--update-W",
            default=True,
            action=ActionYesNo,
            help="whether to update the shared within-class precision during fit",
        )
        parser.add_argument(
            "--balance-class-weight",
            default=False,
            action=ActionYesNo,
            help="whether to balance each class contribution when estimating W",
        )
        parser.add_argument(
            "--prior",
            default=None,
            help="path to a prior LinearGBE model for MAP adaptation",
        )
        parser.add_argument(
            "--prior-beta",
            default=16,
            type=float,
            help="prior relevance factor for class means (MAP adaptation)",
        )
        parser.add_argument(
            "--prior-nu",
            default=16,
            type=float,
            help="prior relevance factor for precision/covariance (MAP adaptation)",
        )
        parser.add_argument(
            "--post-beta",
            default=None,
            type=float,
            help="fixed posterior relevance factor for class means",
        )
        parser.add_argument(
            "--post-nu",
            default=None,
            type=float,
            help="fixed posterior relevance factor for precision/covariance",
        )

        parser.add_argument("--name", default="lgbe", help="model identifier")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    @staticmethod
    def filter_eval_args(**kwargs: Any) -> Dict[str, Any]:
        """Extracts the evaluation time hyperparams of the class from a dictionary.

        Returns:
          Hyperparameters to evaluate the class.
        """
        valid_args = ("model_file", "normalize", "eval_method")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """It adds the arguments needed to evaluate the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--normalize",
            default=False,
            action=ActionYesNo,
            nargs="?",
            help="whether to normalize scores into log-posteriors",
        )
        parser.add_argument(
            "--eval-method",
            default="linear",
            choices=["linear", "llk", "predictive"],
            help="evaluation method: linear function, full Gaussian llk, or predictive distribution",
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args
