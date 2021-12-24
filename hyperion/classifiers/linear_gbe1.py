"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..utils.math import int2onehot, logdet_pdmat, invert_pdmat, softmax


class LinearGBE(HypModel):
    def __init__(
        self,
        mu=None,
        W=None,
        update_mu=True,
        update_W=True,
        x_dim=1,
        num_classes=None,
        balance_class_weight=True,
        do_map=False,
        r_mu=16,
        r_W=16,
        **kwargs
    ):
        super(LinearGBE, self).__init__(**kwargs)
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
        self.do_map = do_map
        self.r_mu = r_mu
        self.r_W = r_W

        self._compute_Ab()

    def get_config(self):
        config = {
            "update_mu": self.update_mu,
            "update_W": self.update_W,
            "x_dim": self.x_dim,
            "num_classes": self.num_classes,
            "balance_class_weight": self.balance_class_weight,
            "do_map": self.do_map,
            "r_mu": self.r_mu,
            "r_W": self.r_W,
        }
        base_config = super(LinearGBE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def predict(self, x, normalize=False, return_full_llk=False):
        logp = np.dot(x, self.A) + self.b

        if return_full_llk:
            K = 0.5 * logdet_pdmat(self.W) - 0.5 * self.x_dim * np.log(2 * np.pi)
            K += -0.5 * np.sum(np.dot(x, self.W) * x, axis=1, keepdims=True)
            logp += K

        if normalize:
            logp = np.log(softmax(logp, axis=1))

        return logp

    def fit(self, x, class_ids=None, p_theta=None, sample_weight=None):

        assert class_ids is not None or p_theta is not None

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

        mu0 = self.mu
        xbar = mu0
        if self.update_mu:
            xbar = F / N[:, None]
            if self.do_map:
                alpha = (N / (N + self.r_mu))[:, None]
                self.mu = (1 - alpha) * mu0 + alpha * xbar
            else:
                self.mu = xbar

        if self.update_W:
            if self.do_map:
                r_W = self.r_W
                alpha = (N / (N + r_W))[:, None]
                S0 = invert_pdmat(self.W, return_inv=True)[-1]
                if self.balance_class_weight:
                    S = (self.num_classes - np.sum(alpha)) * S0
                else:
                    S = self.num_classes * self.r_W * S0
            else:
                r_W = 0
                S = np.zeros((x.shape[1], x.shape[1]), dtype=float_cpu())

            for k in range(self.num_classes):
                delta = x - xbar[k]
                S_k = np.dot(p_theta[:, k] * delta.T, delta)
                if self.do_map:
                    mu_delta = xbar[k] - mu0[k]
                    S_k += self.r_W * alpha[k] * np.outer(mu_delta, mu_delta)

                if self.balance_class_weight:
                    S_k /= N[k] + r_W

                S += S_k

            if self.balance_class_weight:
                S /= self.num_classes
            else:
                S /= self.num_classes * r_w + np.sum(N)

            self.W = invert_pdmat(S, return_inv=True)[-1]

        self._compute_Ab()

    def save_params(self, f):
        params = {"mu": self.mu, "W": self.W}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "W"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    def _compute_Ab(self):
        if self.mu is not None and self.W is not None:
            self.A = np.dot(self.W, self.mu.T)
            self.b = -0.5 * np.sum(self.mu.T * self.A, axis=0)

    @staticmethod
    def filter_args(**kwargs):

        valid_args = (
            "update_mu",
            "update_W",
            "balance_class_weight",
            "do_map",
            "r_mu",
            "r_W",
            "name",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    filter_train_args = filter_args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "."
            p2 = prefix + "."

        parser.add_argument(
            p1 + "no-update-mu",
            default=True,
            action="store_false",
            help="not update mu",
        )
        parser.add_argument(
            p1 + "no-update-W",
            dest=(p2 + "update_W"),
            default=True,
            action="store_false",
            help="not update W",
        )
        parser.add_argument(
            p1 + "balance-class-weight",
            dest=(p2 + "balance_class_weight"),
            default=False,
            action="store_true",
            help="Balances the weight of each class when computing W",
        )
        parser.add_argument(
            p1 + "do-map",
            dest=(p2 + "do_map"),
            default=False,
            action="store_true",
            help="does MAP adaptation",
        )
        parser.add_argument(
            p1 + "r-mu",
            dest=(p2 + "r_mu"),
            default=16,
            type=float,
            help="relevance factor for the means",
        )
        parser.add_argument(
            p1 + "r-w",
            dest=(p2 + "r_W"),
            default=16,
            type=float,
            help="relevance factor for the variances",
        )

        parser.add_argument(
            p1 + "name", dest=(p2 + "name"), default="lgbe", help="model name"
        )

    @staticmethod
    def filter_eval_args(**kwargs):
        valid_args = ("model_file", "normalize", "return_full_llk")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_argparse_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
            p2 = ""
        else:
            p1 = "--" + prefix + "."
            p2 = prefix + "."

        parser.add_argument(
            p1 + "model-file",
            dest=(p2 + "model_file"),
            required=True,
            help=("model file"),
        )
        parser.add_argument(
            p1 + "normalize",
            dest=(p2 + "normalize"),
            default=False,
            action="store_true",
            help=("normalizes the ouput probabilities to sum to one"),
        )
        parser.add_argument(
            p1 + "return-full-llk",
            dest=(p2 + "return_full_llk"),
            default=False,
            action="store_true",
            help=("evaluates full gaussian likelihood instead of linear function"),
        )

    add_argparse_args = add_class_args
    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args
