"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np
from scipy.special import gammaln

from ..hyp_defs import float_cpu
from ..hyp_model import HypModel
from ..utils.math import int2onehot, logdet_pdmat, invert_pdmat, softmax


class QScoringHomoGBE(HypModel):
    def __init__(
        self,
        mu=None,
        W=None,
        N=None,
        balance_class_weight=True,
        prior=None,
        prior_N=None,
        post_N=None,
        **kwargs
    ):

        super(QScoringHomoGBE, self).__init__(**kwargs)

        self.mu = mu
        self.W = W
        self.N = N
        self.balance_class_weight = balance_class_weight
        self.prior = prior
        self.prior_N = prior_N
        self.post_N = post_N

    @property
    def x_dim(self):
        return None if self.mu is None else self.mu.shape[1]

    @property
    def num_classes(self):
        return None if self.mu is None else self.mu.shape[0]

    def get_config(self):
        config = {
            "balance_class_weight": self.balance_class_weight,
            "prior_N": self.prior_N,
        }

        base_config = super(QScoringHomoGBE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _load_prior(self):
        if isinstance(self.prior, str):
            self.prior = QScoringHomoGBE.load(self.prior)
        num_classes = self.prior.mu.shape[0]
        if self.prior_N is not None:
            self.prior.W = 1 + self.prior_N / np.mean(self.prior.N) * (self.W - 1)
            self.prior.N = self.prior_N * np.ones((num_classes,), dtype=float_cpu())

    def _change_post_N(self):
        if self.post_N is not None:
            logging.debug(self.N)
            logging.debug(self.W)
            self.W = 1 + self.post_N / np.mean(self.N) * (self.W - 1)
            self.N = self.post_N * np.ones((self.num_classes,), dtype=float_cpu())
            logging.debug(self.N)
            logging.debug(self.W)

    def fit(self, x, class_ids=None, p_theta=None, sample_weight=None):
        assert class_ids is not None or p_theta is not None

        do_map = True if self.prior is not None else False
        if do_map:
            self._load_prior()

        x_dim = int(x.shape[-1] / 2)
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

        mu_x = x[:, :x_dim]
        s_x = x[:, x_dim:]

        prec_x = 1 / s_x

        N = np.sum(p_theta, axis=0)
        eta = np.dot(p_theta.T, prec_x * mu_x)
        prec = 1 + np.dot(p_theta.T, prec_x - 1)
        if self.prior is not None:
            eta += self.prior.W * self.prior.mu
            prec += self.prior.W - 1
            N += self.prior.N

        C = 1 / prec
        self.mu = C * eta
        self.N = N

        if self.balance_class_weight:
            prec = 1 + np.mean(prec - 1, axis=0)
        else:
            prec = 1 + np.sum(prec_x - 1, axis=0) / num_classes
        self.W = prec

        self._change_post_N()

    def predict(self, x, normalize=False):

        mu_x = x[:, : self.x_dim]
        s_x = x[:, self.x_dim :]
        prec_x = 1 / s_x

        eta_e = self.mu * self.W
        L_e = self.W
        eta_t = prec_x * mu_x
        L_t = prec_x

        L_et = L_t + L_e - 1  # (batch x dim)

        C_et = 1 / L_et  # (batch x dim)
        C_e = C_et - 1 / L_e  # (batch x dim)
        C_t = C_et - 1 / L_t  # (batch x dim)

        r_e = np.sum(np.log(L_e), axis=0, keepdims=True) + np.dot(
            eta_e * eta_e, C_e.T
        )  # (num_classes x batch)
        r_t = np.sum(np.log(L_t), axis=1, keepdims=True) + np.sum(
            C_t * eta_t ** 2, axis=1, keepdims=True
        )  # (batch x 1)
        r_et = -np.sum(np.log(L_et), axis=1, keepdims=True) + 2 * np.dot(
            eta_t * C_et, eta_e.T
        )  # (batch x num_classes)
        logp = 0.5 * (r_et + r_e.T + r_t)

        if normalize:
            logp = np.log(softmax(logp, axis=1))

        return logp

    def save_params(self, f):
        params = {"mu": self.mu, "W": self.W, "N": self.N}

        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "W", "N"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        kwargs = dict(list(config.items()) + list(params.items()))
        return cls(**kwargs)

    @staticmethod
    def filter_train_args(prefix=None, **kwargs):

        valid_args = ("balance_class_weight", "prior", "prior_N", "post_N", "name")

        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "balance-class-weight",
            default=False,
            action="store_true",
            help="Balances the weight of each class when computing W",
        )
        parser.add_argument(
            p1 + "prior", default=None, help="prior file for MAP adaptation"
        )
        parser.add_argument(
            p1 + "prior-N", default=None, type=float, help="relevance factor for prior"
        )
        parser.add_argument(
            p1 + "post-N",
            default=None,
            type=float,
            help="relevance factor for posterior",
        )

        parser.add_argument(p1 + "name", default="q_scoring", help="model name")

    add_argparse_train_args = add_class_args

    @staticmethod
    def filter_eval_args(prefix, **kwargs):
        valid_args = ("model_file", "normalize")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(p1 + "model-file", required=True, help=("model file"))
        parser.add_argument(
            p1 + "normalize",
            default=False,
            action="store_true",
            help=("normalizes the ouput probabilities to sum to one"),
        )

    add_argparse_eval_args = add_eval_args
