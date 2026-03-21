"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import scipy.linalg as la
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from numpy.linalg import matrix_rank

from ...utils.misc import PathLike
from ..hyper_np_model import HyperNPModel


class PCA(HyperNPModel):
    """Class to do principal component analysis.

    Attributes:
      mu: data mean vector
      T: PCA projection.
      update_mu: whether or not to update the mean when training.
      update_T: whether or not to update T when training.
      pca_dim: pca dimension (optional).
      pca_var_r: pca variance ratio to retain, overrides pca_dim (optional).
      pca_min_dim: minimum dimension of PCA when using pca_var_r.
      whiten: whitens the data after PCA.

    Example:
      ```python
      import numpy as np
      from hyperion.np.transforms import PCA

      rng = np.random.default_rng(1234)
      x = rng.standard_normal((4000, 256))

      pca = PCA(pca_dim=128, whiten=True)
      pca.fit(x)
      x_pca = pca.predict(x)
      print(x_pca.shape)  # (4000, 128)
      ```
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        update_mu: bool = True,
        update_T: bool = True,
        pca_dim: Optional[int] = None,
        pca_var_r: Optional[float] = None,
        pca_min_dim: int = 2,
        whiten: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        self.update_mu = update_mu
        self.update_T = update_T
        self.pca_dim = pca_dim
        self.pca_var_r = pca_var_r
        self.pca_min_dim = pca_min_dim
        self.whiten = whiten

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Applies the transformation to the data.

        Args:
          x: data samples.

        Returns:
          Transformed data samples.
        """
        if self.T is None:
            raise ValueError(
                "PCA projection matrix T is None. Fit the model with update_T=True "
                "or load precomputed parameters before calling predict."
            )
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D array, got shape={x.shape}")
        if x.shape[1] != self.T.shape[0]:
            raise ValueError(
                "x feature dimension must match T input dimension, "
                f"got {x.shape[1]} and {self.T.shape[0]}"
            )
        if self.mu is not None and self.mu.shape[0] != x.shape[1]:
            raise ValueError(
                "mu dimension must match x feature dimension, "
                f"got {self.mu.shape[0]} and {x.shape[1]}"
            )
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)

    @staticmethod
    def get_pca_dim_for_var_ratio(
        x: np.ndarray, var_r: float = 1, min_dim: int = 2
    ) -> int:
        if var_r == 1:
            rank = matrix_rank(x)
            if rank <= min_dim:
                # it may have failed, let's try the cov
                rank = matrix_rank(np.dot(x.T, x))
        else:
            sv = la.svd(x, compute_uv=False)
            Ecc = np.cumsum(sv**2)
            Ecc = Ecc / Ecc[-1]
            rank = np.searchsorted(Ecc, var_r, side="left") + 1

        rank = max(min_dim, rank)
        return rank

    def fit(
        self,
        x: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        S: Optional[np.ndarray] = None,
    ) -> None:
        """Trains the model.

        Args:
          x: training data samples with shape (num_samples, x_dim).
          mu: precomputed mean.
          S: precomputed total covariance.
        """
        if x is not None:
            if x.ndim != 2:
                raise ValueError(f"x must be a 2D array, got shape={x.shape}")
            if x.shape[0] == 0:
                raise ValueError("x must have at least one sample")
            if x.shape[1] == 0:
                raise ValueError("x must have at least one feature")
            mu = np.mean(x, axis=0)
            delta = x - mu
            S = np.dot(delta.T, delta) / x.shape[0]
        elif self.update_T and S is None:
            raise ValueError("S must be provided when x is None and update_T=True.")

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
                    np.searchsorted(var_r, self.pca_var_r, side="left") + 1,
                    self.pca_min_dim,
                )

            if self.whiten:
                # the projected features will be whitened
                # do not whiten dimensions with eigenvalue equal to 0.
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
                if self.pca_dim <= 0:
                    raise ValueError(f"pca_dim must be > 0, got {self.pca_dim}")
                if self.pca_dim > V.shape[1]:
                    raise ValueError(
                        "pca_dim must be <= input feature dimension, "
                        f"got pca_dim={self.pca_dim} and x_dim={V.shape[1]}"
                    )
                V = V[:, : self.pca_dim]

            self.T = V

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration dict."""
        config = {
            "update_mu": self.update_mu,
            "update_T": self.update_T,
            "pca_dim": self.pca_dim,
            "pca_var_r": self.pca_var_r,
            "pca_min_dim": self.pca_min_dim,
            "whiten": self.whiten,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_params(self, f: h5py.File) -> None:
        """Saves the model parameters into the file.

        Args:
          f: file handle.
        """
        params = {"mu": self.mu, "T": self.T}
        self._save_params_from_dict(f, params)

    @classmethod
    def load_params(cls, f: h5py.File, config: Dict[str, Any]) -> "PCA":
        """Initializes the model from the configuration and loads the model
        parameters from file.

        Args:
          f: file handle.
          config: configuration dictionary.

        Returns:
          Model object.
        """
        param_list = ["mu", "T"]
        params = cls._load_params_to_dict(f, config["name"], param_list)
        return cls(
            mu=params["mu"],
            T=params["T"],
            **config,
        )

    @classmethod
    def load_mat(cls, file_path: PathLike) -> "PCA":
        with h5py.File(file_path, "r") as f:
            mu = np.asarray(f["mu"], dtype="float32")
            T = np.asarray(f["T"], dtype="float32")
            return cls(mu, T)

    def save_mat(self, file_path: PathLike) -> None:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("mu", data=self.mu)
            f.create_dataset("T", data=self.T)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        valid_args = (
            "update_mu",
            "update_T",
            "name",
            "pca_dim",
            "pca_var_r",
            "pca_min_dim",
            "whiten",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--update-mu",
            default=True,
            action=ActionYesNo,
            help=("updates centering parameter"),
        )
        parser.add_argument(
            "--update-T",
            default=True,
            action=ActionYesNo,
            help=("updates whitening parameter"),
        )
        parser.add_argument(
            "--whiten",
            default=False,
            action=ActionYesNo,
            help=("whitens the data after projection"),
        )

        parser.add_argument(
            "--pca-dim", default=None, type=int, help=("output dimension of PCA")
        )

        parser.add_argument(
            "--pca-var-r",
            default=None,
            type=float,
            help=("proportion of variance to keep when choosing the PCA dimension"),
        )

        parser.add_argument(
            "--pca-min-dim", default=2, type=int, help=("min. output dimension of PCA")
        )

        parser.add_argument("--name", dest="name", default="pca")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_args = add_class_args
