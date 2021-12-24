"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py

from numpy.linalg import matrix_rank
import scipy.linalg as la

from ..hyp_model import HypModel


class PCA(HypModel):
    """Class to do principal component analysis"""

    def __init__(
        self,
        mu=None,
        T=None,
        update_mu=True,
        update_T=True,
        pca_dim=None,
        pca_var_r=None,
        pca_min_dim=2,
        whiten=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        self.update_mu = update_mu
        self.update_T = update_T
        self.pca_dim = pca_dim
        self.pca_var_r = pca_var_r
        self.pca_min_dim = pca_min_dim
        self.whiten = whiten

    def predict(self, x):
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    @staticmethod
    def get_pca_dim_for_var_ratio(x, var_r=1, min_dim=2):
        if var_r == 1:
            rank = matrix_rank(x)
            if rank <= min_dim:
                # it may have failed, let's try the cov
                rank = matrix_rank(np.dot(x.T, x))
        else:
            sv = la.svd(x, compute_uv=False)
            Ecc = np.cumsum(sv ** 2)
            Ecc = Ecc / Ecc[-1]
            rank = np.where(Ecc > var_r)[0][0]

        rank = max(min_dim, rank)
        return rank

    def fit(self, x=None, sample_weight=None, mu=None, S=None):

        if x is not None:
            mu = np.mean(x, axis=0)
            delta = x - mu
            S = np.dot(delta.T, delta) / x.shape[0]

        if self.update_mu:
            self.mu = mu

        if self.update_T:
            d, V = la.eigh(S)
            d = np.flip(d)
            V = np.fliplr(V)

            # This makes the Transform unique
            p = V[0, :] < 0
            V[:, p] *= -1

            if self.pca_var_r is not None:
                var_acc = np.cumsum(d)
                var_r = var_acc / var_acc[-1]
                self.pca_dim = max(
                    np.where(var_r > self.pca_var_r)[0][0], self.pca_min_dim
                )

            if self.whiten:
                # the projected features will be whitened
                # do not whithen dimension with eigenvalue eq. to 0.
                is_zero = d <= 0
                if np.any(is_zero):
                    max_dim = np.where(is_zero)[0][0]
                    V = V[:, :max_dim] * 1 / np.sqrt(d[:max_dim])
                    if self.pca_dim is None:
                        self.pca_dim = max_dim
                    else:
                        self.pca_dim = min(max_dim, self.pca_dim)
                else:
                    V = V * 1 / np.sqrt(d)

            if self.pca_dim is not None:
                assert self.pca_dim <= V.shape[1]
                V = V[:, : self.pca_dim]

            self.T = V

    def get_config(self):
        config = {
            "update_mu": self.update_mu,
            "update_t": self.update_T,
            "pca_dim": self.pca_dim,
            "pca_var_r": self.pca_var_r,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f):
        params = {"mu": self.mu, "T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f, config):
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            T=params["T"],
            pca_dim=config["pca_dim"],
            name=config["name"],
        )

    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("update_mu", "update_T", "name", "pca_dim", "pca_var_r")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "update-mu",
            default=True,
            type=bool,
            help=("updates centering parameter"),
        )
        parser.add_argument(
            p1 + "update-T",
            default=True,
            type=bool,
            help=("updates whitening parameter"),
        )

        parser.add_argument(
            p1 + "pca-dim", default=None, type=int, help=("output dimension of PCA")
        )

        parser.add_argument(
            p1 + "pca-var-r",
            default=None,
            type=int,
            help=("proportion of variance to keep when choosing the PCA dimension"),
        )

        parser.add_argument("--name", dest="name", default="pca")

    add_argparse_args = add_class_args
