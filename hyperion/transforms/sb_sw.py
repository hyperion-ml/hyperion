"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py

import scipy.linalg as la
from sklearn.neighbors import BallTree

from ..hyp_model import HypModel
from ..hyp_defs import float_cpu


class SbSw(HypModel):
    """Class to compute between and within class matrices"""

    def __init__(self, Sb=None, Sw=None, mu=None, num_classes=0, **kwargs):
        super(SbSw, self).__init__(**kwargs)
        self.Sb = None
        self.Sw = None
        self.mu = None
        self.num_classes = num_classes

    def fit(self, x, class_ids, sample_weight=None, class_weights=None, normalize=True):
        dim = x.shape[1]
        if self.Sb is None:
            self.Sb = np.zeros((dim, dim))
            self.Sw = np.zeros((dim, dim))
            self.mu = np.zeros((dim,))
            self.num_classes = 0

        u_ids = np.unique(class_ids)
        self.num_classes += len(u_ids)

        for i in u_ids:
            idx = class_ids == i
            N_i = np.sum(idx)
            mu_i = np.mean(x[idx, :], axis=0)
            self.mu += mu_i
            x_i = x[idx, :] - mu_i
            self.Sb += np.outer(mu_i, mu_i)
            self.Sw += np.dot(x_i.T, x_i) / N_i

        if normalize:
            self.normalize()

    def normalize(self):
        self.mu /= self.num_classes
        self.Sb = self.Sb / self.num_classes - np.outer(self.mu, self.mu)
        self.Sw /= self.num_classes

    @classmethod
    def accum_stats(cls, stats):
        mu = np.zeros_like(stats[0].mu)
        Sb = np.zeros_like(stats[0].Sb)
        Sw = np.zeros_like(stats[0].Sw)
        num_classes = 0
        for s in stats:
            mu += s.mu
            Sb += s.Sb
            Sw += s.Sw
            num_classes += s.num_classes
        return cls(mu=mu, Sb=Sb, Sw=Sw, num_classes=num_classes)

    def save_params(self, f):
        params = {
            "mu": self.mu,
            "Sb": self.Sb,
            "Sw": self.Sw,
            "num_classes": self.num_classes,
        }
        self._save_params_from_dict(f, params)

    @classmethod
    def load(cls, file_path):
        with h5py.File(file_path, "r") as f:
            config = self.load_config_from_json(f["config"])
            param_list = ["mu", "Sb", "Sw", "num_classes"]
            params = cls._load_params_to_dict(f, config["name"], param_list)
            kwargs = dict(list(config.items()) + list(params.items()))
            return cls(**kwargs)


class NSbSw(SbSw):
    def __init__(self, K=10, alpha=1, **kwargs):
        super(NSbSw, self).__init__(**kwargs)
        self.K = K
        self.alpha = alpha

    def fit(self, x, class_ids, sample_weight=None, class_weights=None, normalize=True):
        dim = x.shape[1]
        self.Sb = np.zeros((dim, dim), dtype=float_cpu())
        self.Sw = np.zeros((dim, dim), dtype=float_cpu())
        self.mu = np.zeros((dim,), dtype=float_cpu())

        u_ids = np.unique(class_ids)
        self.num_classes = np.max(u_ids) + 1

        d = np.zeros((self.num_classes, x.shape[0]), dtype=float_cpu())
        delta = np.zeros((self.num_classes,) + x.shape, dtype=float_cpu())
        for i in u_ids:
            idx_i = class_ids == i

            mu_i = np.mean(x[idx_i, :], axis=0)
            self.mu += mu_i

            x_i = x[idx_i]
            tree = BallTree(x_i)
            d_i, NN_i = tree.query(x, k=self.K, dualtree=True, sort_results=True)
            d[i] = d_i[:, -1]
            for l in range(x.shape[0]):
                delta[i, l] = x[l] - np.mean(x_i[NN_i[l]], axis=0)

        d = d ** self.alpha
        for i in u_ids:
            idx_i = (class_ids == i).nonzero()[0]
            N_i = len(idx_i)
            w_i = 0
            Sb_i = np.zeros(self.Sb.shape, dtype=float_cpu())

            for j in range(self.num_classes):
                w_ij = np.minimum(d[i], d[j]) / (d[i] + d[j])
                for l in idx_i:
                    S = np.outer(delta[j, l], delta[j, l])
                    if i == j:
                        self.Sw += S / N_i
                    else:
                        Sb_i += w_ij[l] * S
                        w_i += w_ij[l]
            self.Sb += Sb_i / w_i

        if normalize:
            self.normalize()

    def normalize(self):
        self.mu /= self.num_classes
        self.Sb /= self.num_classes
        self.Sw /= self.num_classes

    def get_config(self):
        config = {"K": self.K, "alpha": self.alpha}
        base_config = super(NSbSw, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
