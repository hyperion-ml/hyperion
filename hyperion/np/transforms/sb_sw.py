"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from sklearn.neighbors import BallTree

from ...hyp_defs import float_cpu
from ..hyper_np_model import HyperNPModel


class SbSw(HyperNPModel):
    """Class to compute between and within class covariance matrices.

    Args:
      Sb: between-class cov. matrix.
      Sw: within-class cov. matrix.
      mu: data mean vector.
      num_classes: number of classes.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import SbSw

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((2000, 256))
      class_ids = rng.integers(0, 100, size=(2000,))

      sbsw = SbSw()
      sbsw.fit(x, class_ids)

      print(sbsw.mu.shape)  # (256,)
      print(sbsw.Sb.shape)  # (256, 256)
      print(sbsw.Sw.shape)  # (256, 256)
      ```
    """

    def __init__(
        self,
        Sb: Optional[np.ndarray] = None,
        Sw: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        num_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initializes Sb/Sw statistics container.

        Args:
          Sb: Initial between-class covariance matrix.
          Sw: Initial within-class covariance matrix.
          mu: Initial global mean vector.
          num_classes: Number of classes represented in the statistics.
          **kwargs: Extra arguments forwarded to `HyperNPModel`.
        """
        super().__init__(**kwargs)
        self.Sb = Sb
        self.Sw = Sw
        self.mu = mu
        self.num_classes = num_classes

    def reset(self, x_dim: Optional[int] = None, dtype: Any = float_cpu()) -> None:
        """Resets statistics accumulators.

        Args:
          x_dim: Feature dimension. If `None`, clears accumulators to `None`.
          dtype: Data type for accumulator arrays.
        """
        if x_dim is None:
            self.Sb = None
            self.Sw = None
            self.mu = None
        else:
            self.Sb = np.zeros((x_dim, x_dim), dtype=dtype)
            self.Sw = np.zeros((x_dim, x_dim), dtype=dtype)
            self.mu = np.zeros((x_dim,), dtype=dtype)
        self.num_classes = 0

    def fit(self, x: np.ndarray, class_ids: np.ndarray, normalize: bool = True) -> None:
        """Computes class-conditional first/second-order statistics.

        Args:
          x: Feature matrix with shape `(num_samples, x_dim)`.
          class_ids: Integer class labels with shape `(num_samples,)`.
          normalize: If `True`, normalizes accumulated statistics at the end.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if class_ids.ndim != 1:
            raise ValueError(
                f"class_ids must be a 1D array, got shape={class_ids.shape}"
            )
        if class_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "class_ids length must match number of samples in x, "
                f"got {class_ids.shape[0]} and {x.shape[0]}"
            )

        dim = x.shape[1]
        self.reset(dim)

        u_ids = np.unique(class_ids)
        self.num_classes = len(u_ids)

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

    def normalize(self) -> None:
        """Normalizes accumulated `mu`, `Sb`, and `Sw` by `num_classes`."""
        self.mu /= self.num_classes
        self.Sb = self.Sb / self.num_classes - np.outer(self.mu, self.mu)
        self.Sw /= self.num_classes

    @classmethod
    def accum_stats(cls, stats: List["SbSw"]) -> "SbSw":
        """Aggregates a list of `SbSw` statistics objects.

        Args:
          stats: List of precomputed `SbSw` instances.

        Returns:
          A new `SbSw` object with summed statistics.
        """
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

    def save_params(self, f: h5py.File) -> None:
        """Saves statistics tensors to an HDF5 file handle.

        Args:
          f: Output HDF5 file handle.
        """
        params = {
            "mu": self.mu,
            "Sb": self.Sb,
            "Sw": self.Sw,
            "num_classes": self.num_classes,
        }
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "SbSw":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "Sb", "Sw", "num_classes"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            Sb=params["Sb"],
            Sw=params["Sw"],
            num_classes=params["num_classes"],
            **config,
        )


class NSbSw(SbSw):
    """Class to compute nearest neighbour between and within class
    covariance matrices.
    https://www.isca-speech.org/archive/pdfs/interspeech_2014/sadjadi14_interspeech.pdf

    This is an NDA-style local statistics estimator. Instead of building
    global between/within scatter only from class means, it uses
    nearest-neighbor structure around each sample:

    1. For each class `i`, find the `K` nearest neighbors (inside class `i`)
       for every sample and compute local differences
       `delta_i(l) = x_l - mean(NN_i(x_l))`.
    2. Build class-dependent distances `d_i(l)` from the farthest neighbor in
       that local set.
    3. For each sample of class `i`, accumulate:
       - within-class term from `delta_i(l)`,
       - between-class term from other classes `j != i`, weighted by
         `w_ij(l) = min(d_i(l), d_j(l)) / (d_i(l) + d_j(l))`.
    4. Normalize accumulated `mu`, `Sb`, and `Sw` by number of classes.

    Compared to global Sb/Sw, this focuses more on local class boundaries and
    can improve discrimination when class structure is non-Gaussian or
    multi-modal.

    Args:
      K: number of neighbours.
      alpha: distance exponent that determines how fast the weight of the samples decays
        when they get far from the classification boundary.
      Sb: between-class cov. matrix.
      Sw: within-class cov. matrix.
      mu: data mean vector.
      num_classes: number of classes.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import NSbSw

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((1500, 256))
      class_ids = rng.integers(0, 80, size=(1500,))

      nsbsw = NSbSw(K=10, alpha=1.0)
      nsbsw.fit(x, class_ids)

      print(nsbsw.mu.shape)  # (256,)
      print(nsbsw.Sb.shape)  # (256, 256)
      print(nsbsw.Sw.shape)  # (256, 256)
      ```
    """

    def __init__(self, K: int = 10, alpha: float = 1, **kwargs: Any) -> None:
        """Initializes nearest-neighbor Sb/Sw estimator.

        Args:
          K: Number of nearest neighbors per class.
          alpha: Exponent applied to class-distance terms.
          **kwargs: Extra arguments forwarded to `SbSw`.
        """
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha

    def fit(self, x: np.ndarray, class_ids: np.ndarray, normalize: bool = True) -> None:
        """Computes nearest-neighbor based between/within class statistics.

        Args:
          x: Feature matrix with shape `(num_samples, x_dim)`.
          class_ids: Integer class labels with shape `(num_samples,)`.
          normalize: If `True`, normalizes accumulated statistics at the end.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if class_ids.ndim != 1:
            raise ValueError(
                f"class_ids must be a 1D array, got shape={class_ids.shape}"
            )
        if class_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "class_ids length must match number of samples in x, "
                f"got {class_ids.shape[0]} and {x.shape[0]}"
            )
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f"alpha must be finite and > 0, got {self.alpha}")

        dim = x.shape[1]
        self.reset(dim, dtype=float_cpu())

        u_ids = np.unique(class_ids)
        self.num_classes = len(u_ids)

        d = np.zeros((self.num_classes, x.shape[0]), dtype=float_cpu())
        delta = np.zeros((self.num_classes,) + x.shape, dtype=float_cpu())
        for i, class_id_i in enumerate(u_ids):
            idx_i = class_ids == class_id_i

            mu_i = np.mean(x[idx_i, :], axis=0)
            self.mu += mu_i

            x_i = x[idx_i]
            tree = BallTree(x_i)
            k_i = min(self.K, x_i.shape[0])
            d_i, NN_i = tree.query(x, k=k_i, dualtree=True, sort_results=True)
            d[i] = d_i[:, -1]
            for l in range(x.shape[0]):
                delta[i, l] = x[l] - np.mean(x_i[NN_i[l]], axis=0)

        d = d**self.alpha
        for i, class_id_i in enumerate(u_ids):
            idx_i = (class_ids == class_id_i).nonzero()[0]
            N_i = len(idx_i)
            w_i = 0
            Sb_i = np.zeros(self.Sb.shape, dtype=float_cpu())

            for j in range(self.num_classes):
                denom = d[i] + d[j]
                w_ij = np.divide(
                    np.minimum(d[i], d[j]),
                    denom,
                    out=np.zeros_like(denom, dtype=float_cpu()),
                    where=denom > 0,
                )
                for l in idx_i:
                    S = np.outer(delta[j, l], delta[j, l])
                    if i == j:
                        self.Sw += S / N_i
                    else:
                        Sb_i += w_ij[l] * S
                        w_i += w_ij[l]
            if w_i > 0:
                self.Sb += Sb_i / w_i

        if normalize:
            self.normalize()

    def normalize(self) -> None:
        """Normalizes nearest-neighbor accumulated statistics by `num_classes`."""
        self.mu /= self.num_classes
        self.Sb /= self.num_classes
        self.Sw /= self.num_classes

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {"K": self.K, "alpha": self.alpha}
        base_config = super(NSbSw, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
