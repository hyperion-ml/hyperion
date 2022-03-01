"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np

from .logistic_regression import LogisticRegression


class BinaryLogisticRegression(LogisticRegression):
    """Binary logistic regression.

    This is a wrapper that add functionalities to sklearn logistic regression.
    Contrary to sklearn, this class produces well-calibrated likelihood ratios.
    Thus, this is suitable for score calibration.

    Attributes:
      A: Scaling Coefficients (num_feats, 1)
      b: biases (1, )
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
      priors: prior prob for having a positive sample.
      random_state: RandomState instance or None, optional, default: None
                    Used when solver == ‘sag’ or ‘liblinear’.
      solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
                 default: ‘liblinear’ Algorithm to use in the optimization problem.
                 For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
                 ‘saga’ are faster for large ones.
                 ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas
                 ‘liblinear’ and ‘saga’ handle L1 penalty.
                 Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.
                 New in version 0.17: Stochastic Average Gradient descent solver.
                 New in version 0.19: SAGA solver.
      max_iter: int, default: 100
                   Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
      dual: bool, default: False
               Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
      tol: float, default: 1e-4
              Tolerance for stopping criteria.
      verbose: int, default: 0
                  For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
      warm_start: bool, default: False
                     When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
                     New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.
      lr_seed: seed for numpy random.
    """

    def __init__(
        self,
        A=None,
        b=None,
        penalty="l2",
        lambda_reg=1e-6,
        use_bias=True,
        bias_scaling=1,
        prior=0.5,
        random_state=None,
        solver="liblinear",
        max_iter=100,
        dual=False,
        tol=0.0001,
        verbose=0,
        warm_start=True,
        lr_seed=1024,
        **kwargs
    ):

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
            lr_seed=1024,
            **kwargs
        )

    @property
    def prior(self):
        """Prior probability for a positive sample."""
        return self.priors[1]

    def get_config(self):
        """Gets configuration hyperparams.
        Returns:
          Dictionary with config hyperparams.
        """

        config = {"prior": self.prior}
        base_config = super().get_config()
        del base_config["priors"]
        return dict(list(base_config.items()) + list(config.items()))

    def predict(self, x, eval_type="logit"):
        """Evaluates the logistic regression.

        It provides well calibrated likelihood ratios or posteriors.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          eval_type: evaluationg method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Ouput scores (num_samples,)
        """
        if x.ndim == 1:
            x = x[:, None]

        y = np.dot(x, self.A).ravel() + self.b

        if eval_type == "log-post":
            y = -np.log(1 + np.exp(-(y + np.log(self.prior / (1 - self.prior)))))
        if eval_type == "post":
            y = 1 / (1 + np.exp(-(y + np.log(self.prior / (1 - self.prior)))))

        return y

    def __call__(self, x, eval_type="logit"):
        """Evaluates the logistic regression.

        Args:
          x: input features (num_samples, feat_dim), it can be (num_samples,) if feat_dim=1.
          eval_type: evaluationg method: logit (log-likelihood ratio), log-post (log-posteriors), post (posteriors)

        Returns:
          Ouput scores (num_samples,)
        """
        return self.predict(x, eval_type)

    @staticmethod
    def filter_class_args(**kwargs):
        """Extracts the hyperparams of the class from a dictionary.

        Returns:
          Hyperparamter dictionary to initialize the class.
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
    def add_class_args(parser, prefix=None):
        """It adds the arguments corresponding to the class to jsonarparse.
        Args:
          parser: jsonargparse object
          prefix: argument prefix.
        """
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "penalty",
            default="l2",
            choices=["l2", "l1"],
            help="used to specify the norm used in the penalization",
        )
        parser.add_argument(
            p1 + "lambda-reg", default=1e-5, type=float, help="regularization strength"
        )
        parser.add_argument(
            p1 + "no-use-bias", default=False, action="store_true", help="Not use bias"
        )
        parser.add_argument(
            p1 + "bias-scaling",
            default=1.0,
            type=float,
            help="useful only when the solver liblinear is used and use_bias is set to True",
        )
        parser.add_argument(
            p1 + "lr-seed", default=1024, type=int, help="random number generator seed"
        )
        parser.add_argument(
            p1 + "solver",
            default="lbfgs",
            choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            help="type of solver",
        )
        parser.add_argument(
            p1 + "max-iter",
            default=100,
            type=int,
            help="only for the newton-cg, sag and lbfgs solvers",
        )
        parser.add_argument(
            p1 + "dual",
            default=False,
            action="store_true",
            help=(
                "dual or primal formulation. "
                "Dual formulation is only implemented for l2 penalty with liblinear solver"
            ),
        )
        parser.add_argument(
            p1 + "tol", default=1e-4, type=float, help="tolerance for stopping criteria"
        )
        parser.add_argument(
            p1 + "verbose",
            default=0,
            type=int,
            help="For the liblinear and lbfgs solvers",
        )
        parser.add_argument(
            p1 + "no-warm-start",
            default=False,
            action="store_true",
            help="don't use previous model to start",
        )

        parser.add_argument(p1 + "prior", default=0.1, type=float, help="Target prior")

        parser.add_argument(p1 + "name", default="lr", help="model name")

    add_argparse_args = add_class_args
