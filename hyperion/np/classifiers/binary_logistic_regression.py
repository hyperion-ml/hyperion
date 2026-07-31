"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .logistic_regression import LogisticRegression


class BinaryLogisticRegression(LogisticRegression):
    """Binary logistic regression.

    This is a wrapper that add functionalities to sklearn logistic regression.
    Contrary to sklearn, this class produces well-calibrated likelihood ratios.
    Thus, this is suitable for score calibration.

    Loss function:
      For training samples ``(x_n, t_n)`` with binary label ``t_n in {0, 1}``,
      the optimized objective is:

      ``L(a, b) = sum_n w_n * CE(t_n, p_n) + lambda_reg * R(a)``

      where:
      ``CE(t_n, p_n) = -t_n * log(p_n) - (1 - t_n) * log(1 - p_n)``
      ``w_n = s_n * (pi_{t_n} / N_{t_n})``

      ``y_n = a^T x_n + b``
      ``z_n = y_n + log(pi_1 / pi_0)``
      ``p_n = 1 / (1 + exp(-z_n))``

      where ``s_n`` is optional ``sample_weight`` (or ``1`` if not provided),
      ``pi_1 = prior``, ``pi_0 = 1 - prior``, and ``N_k`` is the number of
      training samples in class ``k``. ``R(a)`` is either ``||a||_2^2``
      (for ``penalty='l2'``) or ``||a||_1`` (for ``penalty='l1'``), depending
      on the selected solver/penalty.

    Attributes:

      * ``A``: Scale coefficients with shape ``(num_features, 1)``.
      * ``b``: Bias vector with shape ``(1,)``.
      * ``penalty``: ``"l1"`` or ``"l2"`` regularization. The
        ``newton-cg``, ``sag``, and ``lbfgs`` solvers support only L2;
        ``liblinear`` and ``saga`` also support L1.
      * ``lambda_reg``: Positive regularization strength.
      * ``use_bias``: Whether to add an intercept to the decision function.
      * ``bias_scaling``: Synthetic-feature scale used by ``liblinear`` when
        ``use_bias`` is enabled. Increasing it reduces regularization on the
        intercept weight.
      * ``prior``: Prior probability of the positive class.
      * ``random_state``: Optional random generator used by ``sag`` and
        ``liblinear``.
      * ``solver``: Optimization solver. ``liblinear`` is generally suitable
        for small data sets, while ``sag`` and ``saga`` suit larger data sets
        whose features have comparable scales.
      * ``max_iter``, ``dual``, ``tol``, ``verbose``, ``warm_start``: Solver
        convergence and reuse controls passed to the underlying estimator.
      * ``lr_seed``: Random seed used by Hyperion's optimizer wrapper.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.classifiers.binary_logistic_regression import (
      ...     BinaryLogisticRegression,
      ... )
      >>> x = np.array([[0.1, 1.2], [1.0, -0.2], [0.3, 0.4], [1.2, 0.1]])
      >>> y = np.array([0, 1, 0, 1], dtype=np.int64)
      >>> blr = BinaryLogisticRegression(prior=0.5, solver="lbfgs")
      >>> blr.fit(x, y)
      >>> llr = blr.predict(x, eval_type="logit")
      >>> post = blr.predict(x, eval_type="post")
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        penalty: str = "l2",
        lambda_reg: float = 1e-5,
        use_bias: bool = True,
        bias_scaling: float = 1,
        prior: float = 0.5,
        random_state: Any = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        dual: bool = False,
        tol: float = 0.0001,
        verbose: int = 0,
        warm_start: bool = True,
        lr_seed: int = 1024,
        **kwargs: Any,
    ) -> None:

        priors = {0: 1 - prior, 1: prior}
        super().__init__(
            A=A,
            b=b,
            penalty=penalty,
            lambda_reg=lambda_reg,
            use_bias=use_bias,
            bias_scaling=bias_scaling,
            priors=priors,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            dual=dual,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            multi_class="ovr",
            lr_seed=lr_seed,
            **kwargs,
        )

    @property
    def prior(self) -> float:
        """Prior probability for a positive sample."""
        return self.priors[1]

    def get_config(self) -> Dict[str, Any]:
        """Gets configuration hyperparams.
        Returns:
          Dictionary with config hyperparams.
        """

        config = {"prior": self.prior}
        base_config = super().get_config()
        del base_config["priors"]
        return dict(list(base_config.items()) + list(config.items()))

    def predict(self, x: np.ndarray, eval_type: str = "logit") -> np.ndarray:
        """Evaluates the logistic regression.

        It provides well calibrated likelihood ratios or posteriors.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          eval_type: evaluation method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Output scores (num_samples,)
        """
        valid_eval_types = ("logit", "log-post", "post")
        if eval_type not in valid_eval_types:
            raise ValueError(
                f"invalid eval_type={eval_type!r}, valid values are {valid_eval_types}"
            )

        if x.ndim == 1:
            x = x[:, None]

        y = np.dot(x, self.A).ravel() + self.b
        z = y + np.log(self.prior / (1 - self.prior))

        if eval_type == "log-post":
            y = -np.logaddexp(0.0, -z)
        if eval_type == "post":
            y = np.exp(-np.logaddexp(0.0, -z))

        return y

    def __call__(self, x: np.ndarray, eval_type: str = "logit") -> np.ndarray:
        """Evaluates the logistic regression.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          eval_type: evaluation method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Output scores (num_samples,)
        """
        return self.predict(x, eval_type)

    @staticmethod
    def filter_class_args(**kwargs: Any) -> Dict[str, Any]:
        """Extracts the hyperparams of the class from a dictionary.

        Returns:
          Hyperparameter dictionary to initialize the class.
        """
        valid_args = (
            "penalty",
            "lambda_reg",
            "use_bias",
            "bias_scaling",
            "no_use_bias",
            "prior",
            "lr_seed",
            "solver",
            "max_iter",
            "dual",
            "tol",
            "verbose",
            "warm_start",
            "no_warm_start",
            "name",
        )
        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        if "no_use_bias" in d:
            d["use_bias"] = not d["no_use_bias"]
        if "no_warm_start" in d:
            d["warm_start"] = not d["no_warm_start"]

        return d

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """It adds the arguments corresponding to the class to jsonargparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--penalty",
            default="l2",
            choices=["l2", "l1"],
            help="used to specify the norm used in the penalization",
        )
        parser.add_argument(
            "--lambda-reg", default=1e-5, type=float, help="regularization strength"
        )
        parser.add_argument(
            "--no-use-bias", default=False, action=ActionYesNo, help="Not use bias"
        )
        parser.add_argument(
            "--bias-scaling",
            default=1.0,
            type=float,
            help="useful only when the solver liblinear is used and use_bias is set to True",
        )
        parser.add_argument(
            "--lr-seed", default=1024, type=int, help="random number generator seed"
        )
        parser.add_argument(
            "--solver",
            default="lbfgs",
            choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            help="type of solver",
        )
        parser.add_argument(
            "--max-iter",
            default=100,
            type=int,
            help="only for the newton-cg, sag and lbfgs solvers",
        )
        parser.add_argument(
            "--dual",
            default=False,
            action=ActionYesNo,
            help=(
                "dual or primal formulation. "
                "Dual formulation is only implemented for l2 penalty with liblinear solver"
            ),
        )
        parser.add_argument(
            "--tol", default=1e-4, type=float, help="tolerance for stopping criteria"
        )
        parser.add_argument(
            "--verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )
        parser.add_argument(
            "--no-warm-start",
            default=False,
            action=ActionYesNo,
            help="don't use previous model to start",
        )

        parser.add_argument("--prior", default=0.5, type=float, help="Target prior")

        parser.add_argument("--name", default="lr", help="model name")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
