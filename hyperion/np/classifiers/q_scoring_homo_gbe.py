"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from jsonargparse import ActionParser, ArgumentParser

from ...hyp_defs import float_cpu
from ...utils.math_funcs import int2onehot, softmax
from ..hyper_np_model import HyperNPModel


class QScoringHomoGBE(HyperNPModel):
    """Q-scoring homogeneous Gaussian back-end.

    This model expects each input sample as concatenated mean and diagonal
    variance terms: ``x = [x_mean, x_var]`` with shape ``(num_samples, 2*x_dim)``.

    Attributes:
      mu: Class means with shape ``(num_classes, x_dim)``.
      W: Shared per-dimension precision terms with shape ``(x_dim,)``.
      N: Effective class counts with shape ``(num_classes,)``.
      balance_class_weight: If True, balances class contributions when estimating
        ``W``.
      prior: Prior model (or model path) used for MAP adaptation.
      prior_N: Optional relevance factor applied to prior class counts.
      post_N: Optional fixed posterior relevance factor.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        N: Optional[np.ndarray] = None,
        balance_class_weight: bool = False,
        prior: Optional[Union["QScoringHomoGBE", str]] = None,
        prior_N: Optional[float] = None,
        post_N: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a ``QScoringHomoGBE`` model.

        Args:
          mu: Class means with shape ``(num_classes, x_dim)``.
          W: Shared per-dimension precision terms with shape ``(x_dim,)``.
          N: Effective class counts with shape ``(num_classes,)``.
          balance_class_weight: If True, balances class contributions in ``fit``.
          prior: Prior model instance or path to a serialized model.
          prior_N: Optional prior relevance factor for MAP adaptation.
          post_N: Optional fixed posterior relevance factor.
          **kwargs: Extra arguments forwarded to ``HyperNPModel``.
        """

        super().__init__(**kwargs)

        self.mu = mu
        self.W = W
        self.N = N
        self.balance_class_weight = balance_class_weight
        self.prior = prior
        self.prior_N = prior_N
        self.post_N = post_N

    @property
    def x_dim(self) -> Optional[int]:
        """Input mean-feature dimensionality."""
        return None if self.mu is None else self.mu.shape[1]

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes."""
        return None if self.mu is None else self.mu.shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Gets model hyperparameters.

        Returns:
          Dictionary with the hyperparameters of the model.
        """
        config = {
            "balance_class_weight": self.balance_class_weight,
            "prior_N": self.prior_N,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _load_prior(self) -> None:
        """Loads and configures the prior model for MAP adaptation."""
        if isinstance(self.prior, str):
            self.prior = QScoringHomoGBE.load(self.prior)
        num_classes = self.prior.mu.shape[0]
        if self.prior_N is not None:
            mean_prior_N = np.mean(self.prior.N)
            if mean_prior_N <= 0:
                raise ValueError(
                    "prior.N mean must be > 0 to apply prior_N scaling, "
                    f"got mean={mean_prior_N}"
                )
            self.prior.W = 1 + self.prior_N / mean_prior_N * (self.prior.W - 1)
            self.prior.N = self.prior_N * np.ones((num_classes,), dtype=float_cpu())

    def _change_post_N(self) -> None:
        """Applies a fixed posterior relevance factor if configured."""
        if self.post_N is not None:
            logging.debug(self.N)
            logging.debug(self.W)
            mean_N = np.mean(self.N)
            if mean_N <= 0:
                raise ValueError(
                    "posterior N mean must be > 0 to apply post_N scaling, "
                    f"got mean={mean_N}"
                )
            self.W = 1 + self.post_N / mean_N * (self.W - 1)
            self.N = self.post_N * np.ones((self.num_classes,), dtype=float_cpu())
            logging.debug(self.N)
            logging.debug(self.W)

    @staticmethod
    def _split_mean_var(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Splits concatenated mean/variance features and validates variances.

        Args:
          x: Input matrix with shape ``(num_samples, 2*x_dim)``.

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
        x_mean = x[:, :half]
        x_var = x[:, half:]
        if np.any(x_var <= 0):
            raise ValueError("x variance part must be strictly positive")
        return x_mean, x_var

    def fit(
        self,
        x: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        p_theta: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model parameters.

        Args:
          x: Input features ``[x_mean, x_var]`` with shape ``(num_samples, 2*x_dim)``.
          class_ids: Integer class ids in ``[0, num_classes)`` with shape
            ``(num_samples,)``.
          p_theta: Alternative to ``class_ids``, class posteriors with shape
            ``(num_samples, num_classes)``.
          sample_weight: Optional sample weights with shape ``(num_samples,)``.
        """
        assert class_ids is not None or p_theta is not None

        do_map = True if self.prior is not None else False
        if do_map:
            self._load_prior()

        mu_x, s_x = self._split_mean_var(x)
        x_dim = mu_x.shape[-1]
        if self.num_classes is None:
            if class_ids is not None:
                num_classes = np.max(class_ids) + 1
            else:
                num_classes = p_theta.shape[-1]
        else:
            num_classes = self.num_classes

        if class_ids is not None:
            p_theta = int2onehot(class_ids, num_classes)

        if sample_weight is not None:
            p_theta = sample_weight[:, None] * p_theta

        prec_x = 1 / s_x

        N = np.sum(p_theta, axis=0)
        eta = np.dot(p_theta.T, prec_x * mu_x)
        prec = 1 + np.dot(p_theta.T, prec_x - 1)
        if self.prior is not None:
            eta += self.prior.W * self.prior.mu
            prec += self.prior.W - 1
            N += self.prior.N

        if np.any(prec <= 0):
            raise ValueError(
                "Invalid class-conditional precision in fit: all entries in "
                "`prec` must be > 0."
            )
        C = 1 / prec
        self.mu = C * eta
        self.N = N

        if self.balance_class_weight:
            prec = 1 + np.mean(prec - 1, axis=0)
        else:
            prec = 1 + np.sum(prec_x - 1, axis=0) / num_classes
        self.W = prec

        self._change_post_N()

    def predict(self, x: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Evaluates class scores.

        Args:
          x: Input features ``[x_mean, x_var]`` with shape ``(num_trials, 2*x_dim)``.
          normalize: If True, converts scores into log-posteriors.

        Returns:
          Scores with shape ``(num_trials, num_classes)``.
        """
        if self.mu is None or self.W is None:
            raise ValueError(
                "Model parameters are not initialized. Train or load model first."
            )

        mu_x, s_x = self._split_mean_var(x)
        prec_x = 1 / s_x

        eta_e = self.mu * self.W
        L_e = self.W
        eta_t = prec_x * mu_x
        L_t = prec_x

        L_et = L_t + L_e - 1  # (batch x dim)
        if np.any(L_et <= 0):
            raise ValueError(
                "Invalid precision combination: L_t + L_e - 1 must be > 0 "
                "for all samples/dimensions."
            )

        C_et = 1 / L_et  # (batch x dim)
        C_e = C_et - 1 / L_e  # (batch x dim)
        C_t = C_et - 1 / L_t  # (batch x dim)

        r_e = np.sum(np.log(L_e), axis=0, keepdims=True) + np.dot(
            eta_e * eta_e, C_e.T
        )  # (num_classes x batch)
        r_t = np.sum(np.log(L_t), axis=1, keepdims=True) + np.sum(
            C_t * eta_t**2, axis=1, keepdims=True
        )  # (batch x 1)
        r_et = -np.sum(np.log(L_et), axis=1, keepdims=True) + 2 * np.dot(
            eta_t * C_et, eta_e.T
        )  # (batch x num_classes)
        logp = 0.5 * (r_et + r_e.T + r_t)

        if normalize:
            logp = np.log(softmax(logp, axis=1))

        return logp

    def save_params(self, f: Any) -> None:
        """Saves model parameters to an open HDF5 handle."""
        params = {"mu": self.mu, "W": self.W, "N": self.N}

        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "QScoringHomoGBE":
        """Loads model parameters from an open HDF5 handle."""
        param_list = ["mu", "W", "N"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    @staticmethod
    def filter_train_args(
        prefix: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Filters training arguments for model construction."""

        valid_args = ("balance_class_weight", "prior", "prior_N", "post_N", "name")

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return d

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds training arguments to a jsonargparse parser."""
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--balance-class-weight",
            default=False,
            action="store_true",
            help="Balances the weight of each class when computing W",
        )
        parser.add_argument(
            "--prior", default=None, help="prior file for MAP adaptation"
        )
        parser.add_argument(
            "--prior-N", default=None, type=float, help="relevance factor for prior"
        )
        parser.add_argument(
            "--post-N",
            default=None,
            type=float,
            help="relevance factor for posterior",
        )

        parser.add_argument("--name", default="q_scoring", help="model name")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_train_args = add_class_args

    @staticmethod
    def filter_eval_args(prefix: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Filters evaluation arguments for model evaluation."""
        valid_args = ("model_file", "normalize")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds evaluation arguments to a jsonargparse parser."""
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--model-file", required=True, help=("model file"))
        parser.add_argument(
            "--normalize",
            default=False,
            action="store_true",
            help=("normalizes the ouput probabilities to sum to one"),
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_eval_args = add_eval_args
