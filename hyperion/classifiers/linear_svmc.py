"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np

from sklearn.svm import LinearSVC as SVC

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..utils.math import softmax


class LinearSVMC(HypModel):
    def __init__(
        self,
        A=None,
        b=None,
        penalty="l2",
        C=1.0,
        loss="squared_hinge",
        use_bias=True,
        bias_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=100,
        dual=True,
        tol=0.0001,
        multi_class="ovr",
        verbose=0,
        balance_class_weight=True,
        lr_seed=1024,
        **kwargs
    ):

        super().__init__(**kwargs)

        if class_weight is None and balance_class_weight:
            class_weight = "balanced"

        if random_state is None:
            random_state = np.random.RandomState(seed=lr_seed)

        self.use_bias = use_bias
        self.bias_scaling = bias_scaling
        self.balance_class_weight = balance_class_weight
        logging.debug(class_weight)
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

    @property
    def A(self):
        return self.svm.coef_.T

    @property
    def b(self):
        return self.svm.intercept_ * self.bias_scaling

    def get_config(self):
        config = {
            "use_bias": self.use_bias,
            "bias_scaling": self.bias_scaling,
            "balance_class_weight": self.balance_class_weight,
        }
        base_config = super(LinearSVMC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def predict(self, x, eval_type="logit"):

        s = np.dot(x, self.A) + self.b

        if eval_type == "bin-logpost":
            return np.log(1 + np.exp(-s))
        if eval_type == "bin-post":
            return 1 / (1 + np.exp(-s))
        if eval_type == "cat-post":
            return softmax(s)
        if eval_type == "cat-logpost":
            return np.log(softmax(s))

        return s

    def fit(self, x, class_ids, sample_weight=None):
        self.svm.fit(x, class_ids, sample_weight=sample_weight)

    def save_params(self, f):
        params = {"A": self.A, "b": self.b}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["A", "b"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    @staticmethod
    def filter_train_args(prefix=None, **kwargs):

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

    @staticmethod
    def add_class_train_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "."
            p2 = prefix + "."

        parser.add_argument(
            p1 + "penalty",
            default="l2",
            choices=["l2", "l1"],
            help="used to specify the norm used in the penalization",
        )
        parser.add_argument(
            p1 + "c",
            dest=(p2 + "C"),
            default=1.0,
            type=float,
            help="inverse of regularization strength",
        )
        parser.add_argument(
            p1 + "loss",
            default="squared_hinge",
            choices=["hinge", "squared_hinge"],
            help="type of loss",
        )

        parser.add_argument(
            p1 + "no-use-bias",
            dest=(p2 + "use_bias"),
            default=True,
            action="store_false",
            help="Not use bias",
        )
        parser.add_argument(
            p1 + "bias-scaling",
            default=1.0,
            type=float,
            help=(
                "useful only when the solver liblinear is used "
                "and use_bias is set to True"
            ),
        )
        parser.add_argument(
            p1 + "lr-seed", default=1024, type=int, help="random number generator seed"
        )
        parser.add_argument(
            p1 + "max-iter",
            default=100,
            type=int,
            help="only for the newton-cg, sag and lbfgs solvers",
        )
        parser.add_argument(
            p1 + "no-dual",
            dest=(p2 + "dual"),
            default=True,
            action="store_false",
            help=(
                "dual or primal formulation. "
                "Dual formulation is only implemented for "
                "l2 penalty with liblinear solver"
            ),
        )
        parser.add_argument(
            p1 + "tol", default=1e-4, type=float, help="tolerance for stopping criteria"
        )
        parser.add_argument(
            p1 + "multi-class",
            default="ovr",
            choices=["ovr", "crammer_singer"],
            help=(
                "ovr fits a binary problem for each class else "
                "it minimizes the multinomial loss."
            ),
        )
        parser.add_argument(
            p1 + "verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )

        parser.add_argument(
            p1 + "balance-class-weight",
            default=False,
            action="store_true",
            help="Balances the weight of each class when computing W",
        )

        parser.add_argument(p1 + "name", default="svc", help="model name")

    @staticmethod
    def filter_eval_args(prefix, **kwargs):
        valid_args = ("model_file", "eval_type")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "."
            p2 = prefix + "."

        parser.add_argument(p1 + "model-file", required=True, help=("model file"))
        parser.add_argument(
            p1 + "eval-type",
            default="logit",
            choices=["logit", "bin-logpost", "bin-post", "cat-logpost", "cat-post"],
            help=("type of evaluation"),
        )

    add_argparse_train_args = add_class_train_args
    add_argparse_eval_args = add_class_eval_args
