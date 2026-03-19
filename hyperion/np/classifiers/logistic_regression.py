"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from sklearn.linear_model import LogisticRegression as LR

from ...hyp_defs import float_cpu
from ...utils.math_funcs import softmax
from ..hyper_np_model import HyperNPModel


class LogisticRegression(HyperNPModel):
    """Multi-class logistic regression.

    This is a wrapper that add functionalities to sklearn logistic regression.

    Loss function:
      For training samples ``(x_n, y_n)`` with effective per-sample weights
      ``w_n`` (including optional ``sample_weight`` and prior/count-based class
      reweighting), the optimized objective is:

      ``L(W, b) = sum_n w_n * CE(y_n, p(y|x_n; W, b)) + lambda_reg * R(W)``

      where:
      ``CE(y_n, p_n) = - sum_k 1[y_n = k] * log(p_{n,k}) = -log(p_{n,y_n})``
      ``w_n = s_n * (pi_{y_n} / N_{y_n})``

      ``z_{n,k} = w_k^T x_n + b_k``
      ``p_{n,k} = exp(z_{n,k} + log(pi_k)) / sum_j exp(z_{n,j} + log(pi_j))``

      where ``s_n`` is optional ``sample_weight`` (or ``1`` if not provided),
      ``pi_k`` is the class prior for class ``k``, and ``N_k`` is the number of
      training samples in class ``k``. ``R(W)`` is either
      ``||W||_2^2`` (for ``penalty='l2'``) or ``||W||_1`` (for
      ``penalty='l1'``), depending on the selected solver/penalty.

    Attributes:
      A: Scaling Coefficients (num_feats, num_classes)
      b: biases (num_classes, )
      penalty: str, ‘l1’ or ‘l2’, default: ‘l2’ ,
                 Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
                  New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)
      lambda_reg: float, default: 1e-5
                     Regularization strength; must be a positive float.
      use_bias: bool, default: True
                   Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
      bias_scaling: float, default 1.
                       Useful only when the solver ‘liblinear’ is used and use_bias is set to True.
                       In this case, x becomes [x, bias_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.
                       Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) bias_scaling has to be increased.
      priors: dict, default: None
                Prior probabilities in the form {class_label: prior_probability}.
                Class labels must be integer class ids and prior probabilities must
                be finite values in (0, 1) that sum to 1.
                If None, priors are set uniformly at fit time.
      random_state: default_rng instance or None, optional, default: None
                    Used when solver == ‘sag’ or ‘liblinear’.
      solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
                 default: ‘lbfgs’ Algorithm to use in the optimization problem.
                 For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
                 ‘saga’ are faster for large ones.
                 For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
                 handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
                 ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas
                 ‘liblinear’ and ‘saga’ handle L1 penalty.
                 Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.
                 New in version 0.17: Stochastic Average Gradient descent solver.
                 New in version 0.19: SAGA solver.
      max_iter: int, default: 100
                   Useful only for the newton-cg, sag and lbfgs solvers.
                   Maximum number of iterations taken for the solvers to converge.
      dual: bool, default: False
               Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
      tol: float, default: 1e-4
              Tolerance for stopping criteria.
      multi_class: str, {‘ovr’, ‘multinomial’}, default: ‘multinomial’
                     Multiclass option can be either ‘ovr’ or ‘multinomial’. If the option chosen is ‘ovr’, then a binary problem is fit for each label. Else the loss minimised is the multinomial loss fit across the entire probability distribution. Does not work for liblinear solver.
                     New in version 0.18: Stochastic Average Gradient descent solver for ‘multinomial’ case.
      verbose: int, default: 0
                  For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
      warm_start: bool, default: True
                     When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
                     New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.
      num_jobs: int, default: 1
                 Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the ``solver``is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. If given a value of -1, all cores are used.
      lr_seed: seed for numpy random.

    Example:
      >>> import numpy as np
      >>> from hyperion.np.classifiers.logistic_regression import LogisticRegression
      >>> x = np.array([[0.1, 1.2], [1.0, -0.2], [0.3, 0.4], [1.2, 0.1]])
      >>> y = np.array([0, 1, 2, 1], dtype=np.int64)
      >>> priors = {0: 0.2, 1: 0.5, 2: 0.3}
      >>> lr = LogisticRegression(priors=priors, multi_class="multinomial")
      >>> lr.fit(x, y)
      >>> scores = lr.predict(x, eval_type="logit")
      >>> post = lr.predict(x, eval_type="post")
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        penalty: str = "l2",
        lambda_reg: float = 1e-5,
        use_bias: bool = True,
        bias_scaling: Optional[float] = 1,
        priors: Optional[Dict[int, float]] = None,
        random_state: Any = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        dual: bool = False,
        tol: float = 0.0001,
        multi_class: str = "multinomial",
        verbose: int = 0,
        warm_start: bool = True,
        num_jobs: int = 1,
        lr_seed: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if random_state is None:
            # random_state = np.random.default_rng(seed=lr_seed)
            random_state = np.random.RandomState(seed=lr_seed)

        if bias_scaling is None:
            if use_bias and solver == "liblinear":
                bias_scaling = 100
            else:
                bias_scaling = 1

        self.use_bias = use_bias
        self.bias_scaling = bias_scaling
        if priors is None:
            self.priors = None
        else:
            priors_norm = {}
            for k, v in priors.items():
                if isinstance(k, (int, np.integer)):
                    k_int = int(k)
                elif isinstance(k, str):
                    try:
                        k_int = int(k)
                    except ValueError as err:
                        raise ValueError(
                            f"invalid prior class key {k!r}, expected integer class id"
                        ) from err
                    if str(k_int) != k:
                        raise ValueError(
                            f"invalid prior class key {k!r}, expected canonical integer string"
                        )
                else:
                    raise ValueError(
                        f"invalid prior class key {k!r}, expected integer class id"
                    )
                if k_int < 0:
                    raise ValueError(
                        f"invalid prior class key {k!r}, class ids must be >= 0"
                    )
                if k_int in priors_norm:
                    raise ValueError(
                        f"duplicate prior class key after int conversion: {k!r} -> {k_int}"
                    )
                v_float = float(v)
                if not np.isfinite(v_float) or not (0 < v_float < 1):
                    raise ValueError(
                        f"invalid prior value for class {k_int}: {v!r}, expected finite value in (0, 1)"
                    )
                priors_norm[k_int] = v_float
            if len(priors_norm) == 0:
                raise ValueError("priors cannot be empty")
            self.priors = priors_norm
        self.lambda_reg = lambda_reg
        self.multi_class = multi_class
        self.lr = LR(
            penalty=penalty,
            C=1 / lambda_reg,
            dual=dual,
            tol=tol,
            fit_intercept=use_bias,
            intercept_scaling=bias_scaling,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=num_jobs,
        )

        if A is not None:
            self.lr.coef_ = A.T

        if b is not None:
            self.lr.intercept_ = b / self.bias_scaling

    @property
    def A(self) -> np.ndarray:
        return self.lr.coef_.T

    @A.setter
    def A(self, value: np.ndarray) -> None:
        self.lr.coef_ = value.T

    @property
    def b(self) -> np.ndarray:
        return self.lr.intercept_ * self.bias_scaling

    @b.setter
    def b(self, value: Any) -> None:
        if isinstance(value, float):
            value = [value]

        if not isinstance(value, np.ndarray):
            value = np.asarray(value)

        self.lr.intercept_ = value / self.bias_scaling

    def get_config(self) -> Dict[str, Any]:
        """Gets configuration hyperparams.
        Returns:
          Dictionary with config hyperparams.
        """
        config = {
            "use_bias": self.use_bias,
            "bias_scaling": self.bias_scaling,
            "priors": self.priors,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _priors_to_array(self, num_classes: int) -> np.ndarray:
        if self.priors is None:
            raise ValueError("priors must be set")
        expected_keys = set(range(num_classes))
        prior_keys = set(self.priors.keys())
        if prior_keys != expected_keys:
            missing = sorted(expected_keys - prior_keys)
            extra = sorted(prior_keys - expected_keys)
            raise ValueError(
                f"priors keys must match class ids 0..{num_classes-1}, "
                f"missing={missing}, extra={extra}"
            )
        priors = np.asarray(
            [self.priors[i] for i in range(num_classes)], dtype=float_cpu()
        )
        if (
            not np.all(np.isfinite(priors))
            or np.any(priors <= 0)
            or np.any(priors >= 1)
        ):
            raise ValueError("priors must contain finite values in (0, 1)")
        if not np.isclose(np.sum(priors), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(f"priors must sum to 1, got sum={np.sum(priors)}")
        return priors

    def predict(self, x: np.ndarray, eval_type: str = "logit") -> np.ndarray:
        """Evaluates the logistic regression.

        Args:
          x: input features (num_samples, feat_dim),
             it can be (num_samples,) if feat_dim=1.
          eval_type: evaluation method: logit (log-likelihood ratio),
                     log-post (log-posteriors), post (posteriors)

        Returns:
          Output scores (num_samples, num_classes)
        """
        if x.ndim == 1:
            x = x[:, None]

        y = np.dot(x, self.A) + self.b

        if eval_type in ("log-post", "post") and self.priors is None:
            raise ValueError(
                "priors must be set before using eval_type='log-post' or 'post'"
            )

        if self.priors is not None:
            if len(self.priors) != y.shape[1]:
                raise ValueError(
                    f"len(priors)={len(self.priors)} must match num_classes={y.shape[1]}"
                )
            priors = self._priors_to_array(y.shape[1])
        else:
            priors = None

        if eval_type == "log-post":
            y = np.log(softmax(y + np.log(priors), axis=1) + 1e-10)
        if eval_type == "post":
            y = softmax(y + np.log(priors))

        return y

    def __call__(self, x: np.ndarray, eval_type: str = "logit") -> np.ndarray:
        """Evaluates the logistic regression.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          eval_type: evaluation method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Output scores (num_samples, num_classes)
        """
        return self.predict(x, eval_type)

    def fit(
        self,
        x: np.ndarray,
        class_ids: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Estimates the parameters of the model.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          class_ids: class integer [0, num_classes-1] identifier (num_samples,)
          sample_weight: weight of each sample in the estimation (num_samples,)
        """
        if x.ndim == 1:
            x = x[:, None]
        num_classes = np.max(class_ids) + 1
        counts = np.bincount(class_ids)
        assert num_classes == len(counts)

        if self.priors is None:
            prior_value = 1 / num_classes
            self.priors = {i: prior_value for i in range(num_classes)}
        if len(self.priors) != num_classes:
            raise ValueError(
                f"len(priors)={len(self.priors)} must match num_classes={num_classes}"
            )
        priors = self._priors_to_array(num_classes)

        class_weights = np.zeros((num_classes,), dtype=float_cpu())
        valid = counts > 0
        class_weights[valid] = priors[valid] / counts[valid]

        if sample_weight is None:
            sample_weight = class_weights[class_ids]
        else:
            sample_weight = sample_weight * class_weights[class_ids]

        self.lr.fit(x, class_ids, sample_weight=sample_weight)

        if self.multi_class == "ovr":
            # adjust bias to produce log-llk ratios
            if len(self.lr.intercept_) == 1:
                prior = self.priors[1]
                self.lr.intercept_ -= np.log(prior / (1 - prior)) / self.bias_scaling
            else:
                self.lr.intercept_ -= np.log(priors / (1 - priors)) / self.bias_scaling
        else:
            # adjust bias to produce log-llk
            self.lr.intercept_ -= np.log(priors) / self.bias_scaling

    def save_params(self, f: Any) -> None:
        params = {"A": self.A, "b": self.b}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "LogisticRegression":
        param_list = ["A", "b"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

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
            "priors",
            "lr_seed",
            "solver",
            "max_iter",
            "dual",
            "tol",
            "multi_class",
            "verbose",
            "warm_start",
            "no_warm_start",
            "num_jobs",
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
        """It adds the arguments corresponding to the class to jsonarparse.
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
                "Dual formulation is only implemented for "
                "l2 penalty with liblinear solver"
            ),
        )
        parser.add_argument(
            "--tol", default=1e-4, type=float, help="tolerance for stopping criteria"
        )
        parser.add_argument(
            "--multi-class",
            default="ovr",
            choices=["ovr", "multinomial"],
            help=(
                "ovr fits a binary problem for each class else "
                "it minimizes the multinomial loss."
                "Does not work for liblinear solver"
            ),
        )
        parser.add_argument(
            "--verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )
        parser.add_argument(
            "--num-jobs", default=1, type=int, help="number of cores for ovr"
        )
        parser.add_argument(
            "--no-warm-start",
            default=False,
            action=ActionYesNo,
            help="don't use previous model to start",
        )

        parser.add_argument("--name", default="lr", help="model name")

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
        valid_args = ("model_file", "eval_type")
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

        parser.add_argument("--model-file", required=True, help=("model file"))
        parser.add_argument(
            "--eval-type",
            default="logit",
            choices=["logit", "log-post", "post"],
            help=("type of evaluation"),
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    # for backward compatibility
    filter_train_args = filter_class_args
    add_argparse_args = add_class_args
    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args
