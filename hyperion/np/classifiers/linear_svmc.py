"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from sklearn.svm import LinearSVC as SVC

from ...utils.math_funcs import softmax
from ..hyper_np_model import HyperNPModel


class LinearSVMC(HyperNPModel):
    """Linear Support Vector Machine for Classification.

    Attributes:
      A: Linear transformation coefficients with shape ``(feat_dim, num_classes)``.
      b: Bias vector with shape ``(num_classes,)``.
      use_bias: If True, fit an intercept term.
      bias_scaling: Intercept scaling used by ``sklearn.svm.LinearSVC``.
      balance_class_weight: If True and ``class_weight`` is None, it sets
        ``class_weight="balanced"``.
      svm: Internal ``sklearn.svm.LinearSVC`` estimator.
      labels: Optional list of class labels.
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        penalty: Literal["l1", "l2"] = "l2",
        C: float = 1.0,
        loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
        use_bias: bool = True,
        bias_scaling: float = 1,
        class_weight: Optional[Union[Dict[int, float], str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 100,
        dual: bool = True,
        tol: float = 0.0001,
        multi_class: Literal["ovr", "crammer_singer"] = "ovr",
        verbose: int = 0,
        balance_class_weight: bool = False,
        lr_seed: int = 1024,
        labels: Optional[Union[np.ndarray, List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes a ``LinearSVMC`` model.

        Args:
          A: Linear transformation coefficients with shape
            ``(feat_dim, num_classes)``.
          b: Bias vector with shape ``(num_classes,)``.
          penalty: Penalty norm, ``"l1"`` or ``"l2"``.
          C: Regularization parameter. The strength of regularization is
            inversely proportional to ``C`` and must be strictly positive.
          loss: Loss type, ``"hinge"`` or ``"squared_hinge"``.
          use_bias: If True, includes an intercept term.
          bias_scaling: Intercept scaling factor. In this case the model uses
            ``[x, bias_scaling]`` (a synthetic feature appended to the input).
            The intercept becomes ``bias_scaling * synthetic_feature_weight``.
            The synthetic feature weight is regularized as other features, so
            larger values can reduce the relative effect of regularization.
          class_weight: Dictionary or ``"balanced"``. If a dict is provided,
            class ``i`` uses effective regularization ``class_weight[i] * C``.
            If ``"balanced"``, class weights are automatically set inversely
            proportional to class frequencies.
          random_state: Integer seed or ``np.random.RandomState`` used by sklearn.
          max_iter: Maximum number of optimization iterations.
          dual: Dual or primal formulation.
          tol: Tolerance for stopping criteria.
          multi_class: ``"ovr"`` or ``"crammer_singer"``. ``"ovr"`` trains
            one-vs-rest classifiers. ``"crammer_singer"`` optimizes a joint
            multi-class objective and ignores ``loss``, ``penalty``, and
            ``dual``.
          verbose: Verbosity level.
          balance_class_weight: If True and ``class_weight`` is None, sets
            ``class_weight="balanced"``.
          lr_seed: RNG seed used only when ``random_state`` is None.
          labels: Optional class labels.
          **kwargs: Extra arguments forwarded to ``HyperNPModel``.
        """

        super().__init__(**kwargs)

        if class_weight is None and balance_class_weight:
            class_weight = "balanced"

        if random_state is None:
            # random_state = np.random.default_rng(seed=lr_seed)
            random_state = np.random.RandomState(seed=lr_seed)
        elif isinstance(random_state, np.random.Generator):
            raise TypeError(
                "random_state as np.random.Generator is not supported; "
                "use int seed or np.random.RandomState"
            )

        self.use_bias = use_bias
        self.bias_scaling = bias_scaling
        self.balance_class_weight = balance_class_weight
        self.svm = SVC(
            penalty=penalty,
            C=C,
            loss=loss,
            dual=dual,
            tol=tol,
            fit_intercept=use_bias,
            intercept_scaling=bias_scaling,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
        )

        if A is not None:
            self.svm.coef_ = A.T

        if b is not None:
            self.svm.intercept_ = b

        self.set_labels(labels)

    @property
    def A(self) -> np.ndarray:
        """Linear transformation coefficients."""
        return self.svm.coef_.T

    @property
    def b(self) -> np.ndarray:
        """Bias vector."""
        return self.svm.intercept_

    def set_labels(self, labels: Optional[Union[np.ndarray, List[Any]]]) -> None:
        """Sets class labels.

        Args:
          labels: Labels as list/array or ``None``.
        """
        if isinstance(labels, np.ndarray):
            labels = list(labels)

        self.labels = labels

    def get_config(self) -> Dict[str, Any]:
        """Gets configuration hyperparams.

        Returns:
          Dictionary with config hyperparams.
        """
        config = {
            "use_bias": self.use_bias,
            "bias_scaling": self.bias_scaling,
            "balance_class_weight": self.balance_class_weight,
            "labels": self.labels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def predict(
        self,
        x: np.ndarray,
        eval_type: Literal[
            "logit", "bin-log-post", "bin-post", "cat-log-post", "cat-post"
        ] = "logit",
    ) -> np.ndarray:
        """Evaluates the SVM.

        Args:
          x: Input features with shape ``(num_samples, feat_dim)``.
             It can also be ``(num_samples,)`` when ``feat_dim=1``.
          eval_type: Evaluation method:
            ``"logit"`` returns linear scores (logits),
            ``"bin-log-post"`` returns binary log-posteriors,
            ``"bin-post"`` returns binary posteriors,
            ``"cat-log-post"`` returns categorical log-posteriors,
            ``"cat-post"`` returns categorical posteriors.

        Returns:
          Output scores with shape ``(num_samples, num_classes)``.
        """
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim != 2:
            raise ValueError(f"x must be 1D or 2D, got shape={x.shape}")

        s = np.dot(x, self.A) + self.b

        if eval_type == "bin-log-post":
            return -np.logaddexp(0.0, -s)
        if eval_type == "bin-post":
            return np.exp(-np.logaddexp(0.0, -s))
        if eval_type == "cat-post":
            return softmax(s)
        if eval_type == "cat-log-post":
            return np.log(softmax(s))
        if eval_type == "logit":
            return s
        raise ValueError(f"Invalid eval_type={eval_type}")

    def __call__(
        self,
        x: np.ndarray,
        eval_type: Literal[
            "logit", "bin-log-post", "bin-post", "cat-log-post", "cat-post"
        ] = "logit",
    ) -> np.ndarray:
        """Evaluates the SVM.

        Args:
          x: Input features with shape ``(num_samples, feat_dim)``.
             It can also be ``(num_samples,)`` when ``feat_dim=1``.
          eval_type: Evaluation method:
            ``"logit"`` returns linear scores (logits),
            ``"bin-log-post"`` returns binary log-posteriors,
            ``"bin-post"`` returns binary posteriors,
            ``"cat-log-post"`` returns categorical log-posteriors,
            ``"cat-post"`` returns categorical posteriors.

        Returns:
          Output scores with shape ``(num_samples, num_classes)``.
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
          x: Input features with shape ``(num_samples, feat_dim)``.
             It can also be ``(num_samples,)`` when ``feat_dim=1``.
          class_ids: Integer class identifiers in ``[0, num_classes-1]``
            with shape ``(num_samples,)``.
          sample_weight: Optional sample weights with shape ``(num_samples,)``.
        """
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim != 2:
            raise ValueError(f"x must be 1D or 2D, got shape={x.shape}")

        self.svm.fit(x, class_ids, sample_weight=sample_weight)

    def save_params(self, f: Any) -> None:
        """Saves model parameters to an open HDF5 handle."""
        params = {"A": self.A, "b": self.b}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: Any, config: Dict[str, Any]) -> "LinearSVMC":
        """Loads model parameters from an open HDF5 handle."""
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
            "C",
            "loss",
            "use_bias",
            "bias_scaling",
            "class_weight",
            "lr_seed",
            "max_iter",
            "dual",
            "tol",
            "multi_class",
            "verbose",
            "balance_class_weight",
            "name",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    filter_train_args = filter_class_args

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
            "--c",
            dest="C",
            default=1.0,
            type=float,
            help="inverse of regularization strength",
        )
        parser.add_argument(
            "--loss",
            default="squared_hinge",
            choices=["hinge", "squared_hinge"],
            help="type of loss",
        )

        parser.add_argument(
            "--use-bias",
            default=True,
            action=ActionYesNo,
            nargs="?",
            help="Use bias",
        )
        parser.add_argument(
            "--bias-scaling",
            default=1.0,
            type=float,
            help=(
                "useful only when the solver liblinear is used "
                "and use_bias is set to True"
            ),
        )
        parser.add_argument(
            "--lr-seed", default=1024, type=int, help="random number generator seed"
        )
        parser.add_argument(
            "--max-iter",
            default=100,
            type=int,
            help="only for the newton-cg, sag and lbfgs solvers",
        )
        parser.add_argument(
            "--dual",
            default=True,
            action=ActionYesNo,
            nargs="?",
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
            choices=["ovr", "crammer_singer"],
            help=(
                "ovr fits a binary problem for each class else "
                "it minimizes the multinomial loss."
            ),
        )
        parser.add_argument(
            "--verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )

        parser.add_argument(
            "--balance-class-weight",
            default=False,
            action=ActionYesNo,
            help="Balances the weight of each class when computing W",
        )

        parser.add_argument("--name", default="svc", help="model name")
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
        valid_args = ("eval_type",)
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """It adds the arguments needed to evaluate the class to jsonargparse.

        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--eval-type",
            default="logit",
            choices=["logit", "bin-log-post", "bin-post", "cat-log-post", "cat-post"],
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
